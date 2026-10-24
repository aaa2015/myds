#!/usr/bin/env python3
"""Frozen 16-run same-context MoE recovery gate with exact service restoration.

This controller never builds, edits core source, cleans native raw output, or
changes production configuration. --prepare performs only inherited read-only
preflight and CPU identity checks. Only explicit --run stops the service and
invokes the two native diagnostics inside the inherited exclusive window.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import sys
sys.dont_write_bytecode = True
import run_moe_workspace_window as workspace

need, sha, write_json = workspace.need, workspace.sha, workspace.write_json
FLAG = workspace.FLAG
RECOVERY_PLAN_SHA = 'f277f1d9f07e7ce3580b64a5f4a4153bde0a75dd70bc7d7fbe1f93304c666720'
SCENARIOS = ('deterministic', 'async', 'allocation', 'shared-pso')
TEMPERATURES = ('0', '0.7')
SETTINGS = (0, 1)


def frozen_matrix(plan):
    need(plan['schema_version'] == 2 and plan['before_recovery_gpu'] is True and
         plan['candidate_flag'] == FLAG and plan['settings'] == [0, 1] and
         plan['same_build'] is True and plan['temperatures'] == [0, 0.7] and
         plan['recover_tokens'] == 8 and plan['max_seq_len'] == 327680 and
         plan['seed'] == 42 and plan['production51settings_preserved'] is True and
         plan['no_manual_rng_reset'] is True and plan['executions'] == 16,
         'Frozen recovery configuration differs')
    need(plan['scenarios'] == [dict(kind='deterministic', checkpoint=1, delay_ms=0),
                              dict(kind='async', checkpoint=1, delay_ms=0),
                              dict(kind='allocation', ordinal=1, real_oom=False),
                              dict(kind='shared-pso', mask=8, expected='failclosed before MoE dispatch')],
         'Frozen recovery scenarios differ')
    need(plan['warm_request'] == plan['recover_request'] == 'requests/pf7k.json' and
         plan['failure_request'] == plan['cancel_request'] == 'requests/pf16k.json',
         'Frozen actual requests differ')
    return [dict(scenario=scenario, temperature=temperature, setting=setting)
            for scenario in SCENARIOS for temperature in TEMPERATURES for setting in SETTINGS]


def pair_name(scenario, temperature):
    return scenario + '-t' + temperature.replace('.', 'p')


class RecoveryWindow(workspace.WorkspaceWindow):
    def preflight(self):
        super().preflight()
        need(sha(self.a.recovery_plan) == RECOVERY_PLAN_SHA, 'Frozen recovery plan SHA differs')
        self.recovery_plan = json.loads(self.a.recovery_plan.read_text())
        self.matrix = frozen_matrix(self.recovery_plan)
        prerequisite = json.loads(self.a.component_review.read_text())
        need(prerequisite.get('evidence_valid') is True and prerequisite.get('component_gate_pass') is True,
             'Independent valid real-weight component gate is required before recovery')
        self.requests = {}
        for name in ('warm', 'failure', 'recover', 'cancel'):
            path = self.a.inputs / Path(self.recovery_plan[name + '_request']).name
            need(path.is_file() and sha(path) == self.recovery_plan[name + '_request_sha256'],
                 'Frozen original request differs: ' + name)
            self.requests[name] = path
        required = ['bin/test_moe_workspace_cancel', 'bin/test_moe_workspace_recovery',
                    'tools/test_moe_workspace_cancel.py', 'tools/test_moe_workspace_recovery.py',
                    'tools/test_moe_workspace_cancel.c', 'tools/test_moe_workspace_recovery.c']
        for name in required:
            path = self.a.stage / name
            need(path.is_file() and str(path) in self.input_hashes,
                 'Recovery executable/source is absent from frozen stage inventory: ' + name)
        need(str(Path(__file__).resolve()) in self.input_hashes, 'Execute the frozen recovery controller')
        self.input_hashes[str(self.a.component_review)] = sha(self.a.component_review)
        self.input_hashes[str(self.a.recovery_plan)] = sha(self.a.recovery_plan)
        write_json(self.a.out / 'stage-inputs.json', self.input_hashes)
        write_json(self.a.out / 'recovery-protocol.json', self.recovery_plan)
        write_json(self.a.out / 'recovery-matrix.json', self.matrix)
        self.summary.update(component='actual_prefill_moe_workspace_same_context_recovery',
                            recovery_run=False, recovery_gate_pass=False,
                            recovery_plan_sha256=sha(self.a.recovery_plan),
                            component_review_sha256=sha(self.a.component_review),
                            recovery_executions_required=16, recovery_pair_reports_required=8)
        self.arm_records, self.pair_reports = [], []

    def native_command(self, scenario, temperature, directory):
        common = ['--run', '--model', self.a.model, '--out-dir', directory,
                  '--scenario', scenario, '--temperature', temperature, '--recover-tokens', '8']
        if scenario in ('deterministic', 'async'):
            return [self.a.stage / 'bin/test_moe_workspace_cancel', *common,
                    '--cancel-request', self.requests['cancel'], '--recover-request', self.requests['recover'],
                    '--checkpoint', '1', '--delay-ms', '0']
        return [self.a.stage / 'bin/test_moe_workspace_recovery', *common,
                '--warm-request', self.requests['warm'], '--failure-request', self.requests['failure'],
                '--recover-request', self.requests['recover']]

    def reader_command(self, scenario, off, on, output):
        common = ['--off', off, '--on', on, '--recover-request', self.requests['recover'], '--out', output]
        if scenario in ('deterministic', 'async'):
            return ['/usr/bin/python3', self.a.stage / 'tools/test_moe_workspace_cancel.py', *common,
                    '--cancel-request', self.requests['cancel']]
        return ['/usr/bin/python3', self.a.stage / 'tools/test_moe_workspace_recovery.py', *common,
                '--warm-request', self.requests['warm'], '--failure-request', self.requests['failure']]

    def component(self):
        self.summary.update(recovery_run=True, fullmodel_run=True, performance_measured=False)
        for name in ('arms', 'arm-environments', 'reports'):
            (self.a.out / name).mkdir()
        for scenario in SCENARIOS:
            for temperature in TEMPERATURES:
                pair = pair_name(scenario, temperature)
                directories = []
                for setting in SETTINGS:
                    label = pair + ('-on' if setting else '-off')
                    directory = self.a.out / 'arms' / label
                    need(not directory.exists(), 'Native raw output directory must be new')
                    env = dict(self.environment, **{FLAG: str(setting)})
                    actual = {k: v for k, v in env.items() if k.startswith('MYDS_')}
                    expected = dict(self.before['effective_myds_env'], **{FLAG: str(setting)})
                    need(actual == expected, 'Recovery must retain all51 production settings except its explicit flag')
                    write_json(self.a.out / 'arm-environments' / (label + '.json'), actual)
                    argv = self.native_command(scenario, temperature, directory)
                    self.run_command(label, argv, env=env, timeout=1800)
                    summary_path = directory / 'summary.json'
                    record = json.loads(summary_path.read_text())
                    need(record.get('status') == 'complete' and record.get('fullmodel_recovery_within_arm_completed') is True and
                         record.get('scenario') == scenario and record.get('workspace_mode') == setting and
                         type(record.get('temperature')) in (int, float) and
                         math.isclose(record['temperature'], float(temperature), abs_tol=1e-6) and
                         record.get('recovery_tokens') == 8 and record.get('same_context') is True and
                         record.get('rng_manually_reset') is False and record.get('extra_gpu_waits_inserted') == 0,
                         'Native recovery scope did not complete exactly as requested: ' + label)
                    directories.append(directory)
                    self.arm_records.append(dict(label=label, scenario=scenario, temperature=temperature,
                                                 setting=setting, raw_directory=str(directory),
                                                 native_summary_sha256=sha(summary_path)))
                    write_json(self.a.out / 'recovery-arms.json', self.arm_records)
                output = self.a.out / 'reports' / (pair + '.json')
                self.run_command(pair + '-reader', self.reader_command(scenario, *directories, output),
                                 env=dict(self.environment, **{FLAG: '0'}), timeout=1200)
                report = json.loads(output.read_text())
                need(report.get('evidence_valid') is True and report.get('correctness_pass') is True and
                     report.get('performance_measured') is False and report.get('scenario') == scenario and
                     type(report.get('temperature')) in (int, float) and
                     math.isclose(report['temperature'], float(temperature), abs_tol=1e-6) and
                     report.get('recovery_tokens') == 8,
                     'Independent original-raw recovery pair reader failed: ' + pair)
                self.pair_reports.append(dict(pair=pair, scenario=scenario, temperature=temperature,
                                              path=str(output), sha256=sha(output), evidence_valid=True,
                                              correctness_pass=True))
                write_json(self.a.out / 'recovery-pairs.json', self.pair_reports)
        need(len(self.arm_records) == 16 and len(self.pair_reports) == 8,
             'Complete16 executions and8 original-raw pair reviews are required')
        self.summary.update(recovery_run=True, recovery_pairs_pass=True,
                            recovery_executions_completed=16, recovery_pair_reports_completed=8,
                            fullmodel_run=True, performance_measured=False)

    def run(self):
        code = super().run()
        if self.a.prepare:
            return code
        passed = (code == 0 and self.summary.get('status') == 'component_complete_review_required' and
                  'error' not in self.summary and
                  self.summary.get('recovery_pairs_pass') is True and
                  self.summary.get('restoration', {}).get('restored') is True and
                  len(getattr(self, 'arm_records', [])) == 16 and len(getattr(self, 'pair_reports', [])) == 8)
        result = dict(schema_version=1, evidence_valid=passed, recovery_gate_pass=passed,
                      production_adoption=False, performance_measured=False,
                      decision='eligible_for_separate_fullmodel_gate' if passed else 'reject_incomplete_or_failed_recovery_window',
                      recovery_plan_sha256=self.summary.get('recovery_plan_sha256'),
                      component_review_sha256=self.summary.get('component_review_sha256'),
                      arm_executions=getattr(self, 'arm_records', []), pair_reports=getattr(self, 'pair_reports', []),
                      original_pid=self.before['pid'] if self.before else None,
                      restoration=self.summary.get('restoration'),
                      limits=['Original native raw outputs are preserved; independent pair readers consume them directly.',
                              'Asynchronous interrupted partial live capacity is outside byte comparison; complete recovery state is compared.',
                              'Allocation failure is deterministic NULL injection, not actual memory pressure or OOM.',
                              'Recovery correctness does not establish fullmodel latency or memory gains.'])
        write_json(self.a.out / 'recovery-review.json', result)
        self.summary.update(status='recovery_passed_review_required' if passed else 'failed', recovery_gate_pass=passed)
        write_json(self.a.out / 'summary.json', self.summary)
        self.event('recovery_decision_complete', status=self.summary['status'], recovery_gate_pass=passed)
        return 0 if passed else 1


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('stage', 'gold-stage', 'model', 'out', 'plan', 'inputs', 'component-review', 'recovery-plan'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--port', type=int, default=8187)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    return p.parse_args(argv)


def main():
    a = arguments(); need(sys.platform == 'darwin', 'macOS required')
    for name in ('stage', 'gold_stage', 'model', 'out', 'plan', 'inputs', 'component_review', 'recovery_plan'):
        path = getattr(a, name); need(path.is_absolute(), 'Absolute paths required')
        setattr(a, name, path.resolve())
    for stage in (a.stage, a.gold_stage):
        need(Path('/tmp').resolve() in stage.parents, 'Separate tmp stages required')
    a.out.mkdir(parents=True, exist_ok=False); os.chmod(a.out, 0o700); signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try: os.setsid()
    except PermissionError: pass
    with (a.out / 'runner.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2); os.dup2(null.fileno(), 0)
    return RecoveryWindow(a).run()


if __name__ == '__main__':
    sys.exit(main())
