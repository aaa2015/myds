#!/usr/bin/env python3
"""Isolated successful same-context MoE growth diagnostic; never a timing cohort.

The inherited exclusive WorkspaceWindow owns stop, monitoring, drain and finally
restoration. --prepare remains read-only. This new controller neither builds nor
modifies a frozen stage, production configuration, or any native raw payload.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
sys.dont_write_bytecode = True
import run_moe_workspace_window as workspace

need, sha, write_json = workspace.need, workspace.sha, workspace.write_json
FLAG = workspace.FLAG
REUSE_PLAN_SHA = 'f4473d7b634f5b63502e53f3ac38ad8a0bde8edcd5b98e344cce842d889fd40e'
ORDER = ('A1', 'B', 'A2')
STEPS = ('warm', 'grow', 'retain')


def frozen_plan(plan):
    need(plan['schema_version'] == 1 and plan['frozen_before_reuse_gpu'] is True and
         plan['prior_fullmodel_plan_sha256'] == workspace.PLAN_SHA and plan['candidate_flag'] == FLAG and
         plan['order'] == list(ORDER) and plan['flags'] == {'A1': 0, 'B': 1, 'A2': 0} and
         plan['sequences_per_arm'] == 1 and plan['total_generations'] == 9 and
         plan['same_context_within_sequence'] is True and plan['temperature'] == 0 and
         plan['max_new_tokens'] == 8 and plan['seed'] == 42 and plan['max_seq_len'] == 327680 and
         plan['production51env_and_cache_settings_preserved'] is True and plan['manual_rng_reset'] is False and
         plan['memory_sampling_interval_ms'] == 10, 'Frozen successful-reuse protocol differs')
    expected = [dict(name='warm', request='pf7k.json', expected_prompt_tokens=6917, expected_B=8192),
                dict(name='grow', request='pf16k.json', expected_prompt_tokens=15187, expected_B=16384),
                dict(name='retain', request='pf7k.json', expected_prompt_tokens=6917, expected_B=16384)]
    need(plan['generation_steps_per_sequence'] == expected, 'Frozen same-context warm/grow/retain sequence differs')
    gate = plan['adoption_constraint']
    need(gate['reject_if_B_grow_observed_peak_device_or_phys_exceeds_conservative_A1_A2_by_bytes'] == 134217728 and
         gate['missing_peak_coverage_is_not_a_pass'] is True and
         gate['no_fullmodel_time_threshold_relaxation_or_new_performance_samples'] is True and
         plan['CPU_shadow_formula_is_not_actual_memory_evidence'] is True,
         'Frozen actual-peak coverage gate differs')
    return expected


def native_completion(directory, setting):
    data = [json.loads(line) for line in (directory/'summary.jsonl').read_text().splitlines() if line.strip()]
    metadata = [r for r in data if r.get('kind') == 'metadata']
    samples = [r for r in data if r.get('kind') == 'sample']
    complete = [r for r in data if r.get('kind') == 'complete']
    need(len(metadata) == len(complete) == 1 and data[-1] == complete[0] and
         metadata[0]['workspace_mode'] == setting and metadata[0]['testing_api_used'] is False and
         metadata[0]['same_context'] is True and metadata[0]['manual_cache_or_rng_reset'] is False and
         metadata[0]['extra_gpu_waits_inserted'] == 0 and metadata[0]['performance_measured'] is False,
         'Native normal-API same-context scope incomplete')
    need([(r['run'], r['step']) for r in samples] == list(enumerate(STEPS)) and
         all(r['generation_ok'] is True and r['raw_state_exported'] is True and r['output_tokens'] == 8 and
             r['stop_reason'] == 'length' and r['same_context'] is True for r in samples),
         'All three complete raw successful stages are required')
    need(complete[0]['three_generations_completed'] is True and
         complete[0]['actual_C2048_B8192_to16384_grow'] is True and
         complete[0]['actual_B16384_retained_after_short'] is True and
         complete[0]['performance_measured'] is False and complete[0]['adoption_gate_pass'] is False,
         'Native actual growth/retention must complete without a native adoption claim')
    return metadata[0], complete[0]


class ReuseWindow(workspace.WorkspaceWindow):
    def preflight(self):
        super().preflight()
        need(sha(self.a.reuse_plan) == REUSE_PLAN_SHA, 'Frozen reuse plan SHA differs')
        self.reuse_plan = json.loads(self.a.reuse_plan.read_text())
        frozen_plan(self.reuse_plan)
        for path, key in ((self.a.component_review, 'component_gate_pass'),
                          (self.a.recovery_review, 'recovery_gate_pass')):
            result = json.loads(path.read_text())
            need(result.get('evidence_valid') is True and result.get(key) is True,
                 'Independent frozen prerequisite failed: '+key)
            self.input_hashes[str(path)] = sha(path)
        for name, field in (('pf7k.json', 'pf7k_request_sha256'), ('pf16k.json', 'pf16k_request_sha256')):
            need(sha(self.a.inputs/name) == self.reuse_plan[field], 'Original reuse request differs')
        for name in ('tools/bench_moe_workspace_reuse.c', 'tools/moe_workspace_reuse.mk',
                     'tools/run_moe_workspace_reuse_window.py', 'tools/review_moe_workspace_reuse_artifacts.py',
                     'bin/bench_moe_workspace_reuse'):
            need(str(self.a.stage/name) in self.input_hashes, 'Reuse source/bin is not frozen: '+name)
        need(str(Path(__file__).resolve()) in self.input_hashes, 'Execute the actual frozen reuse controller')
        binary = self.a.stage/'bin/bench_moe_workspace_reuse'
        symbols = workspace.safety.command(['/usr/bin/nm', '-gU', str(binary)])
        forbidden = ('myds_metal_moe_workspace_test_control', 'myds_v41_moe_workspace_probe_create',
                     'myds_v41_moe_workspace_cancel_control')
        need(not any(name in symbols for name in forbidden), 'Reuse binary contains a TESTING/fault-control API')
        write_json(self.a.out/'normal-api-symbol-check.json', dict(binary_sha256=sha(binary),
                   forbidden_symbols=list(forbidden), forbidden_symbols_present=False, nm_stdout=symbols))
        self.input_hashes[str(self.a.reuse_plan)] = sha(self.a.reuse_plan)
        write_json(self.a.out/'stage-inputs.json', self.input_hashes)
        write_json(self.a.out/'reuse-protocol.json', self.reuse_plan)
        self.summary.update(component='successful_same_context_moe_workspace_growth_reuse',
            reuse_plan_sha256=sha(self.a.reuse_plan), reuse_run=False, performance_measured=False,
            reuse_growth_gate_pass=False, adoption_requires_independent_shadow_peak_coverage=True)
        self.arm_records = []

    def component(self):
        self.summary.update(reuse_run=True, fullmodel_run=True, performance_measured=False)
        (self.a.out/'arms').mkdir(); (self.a.out/'arm-environments').mkdir()
        for name in ORDER:
            setting = self.reuse_plan['flags'][name]
            directory = self.a.out/'arms'/name
            env = dict(self.environment, **{FLAG: str(setting)})
            expected = dict(self.before['effective_myds_env'], **{FLAG: str(setting)})
            need({k:v for k,v in env.items() if k.startswith('MYDS_')} == expected,
                 'All51 settings must remain intact except the explicit workspace flag')
            write_json(self.a.out/'arm-environments'/(name+'.json'), expected)
            argv = [self.a.stage/'bin/bench_moe_workspace_reuse', '--run', '--model', self.a.model,
                    '--warm-request', self.a.inputs/'pf7k.json', '--grow-request', self.a.inputs/'pf16k.json',
                    '--retain-request', self.a.inputs/'pf7k.json', '--out-dir', directory]
            self.run_command(name, argv, env=env, timeout=1800)
            metadata, complete = native_completion(directory, setting)
            need(metadata['environment'] == expected, 'Actual native MYDS environment differs')
            need(Path(metadata['actual_metallib_path']).resolve(strict=True) ==
                 (self.a.stage/'bin/myds.metallib').resolve(strict=True), 'Actual native library differs')
            self.arm_records.append(dict(name=name, setting=setting, raw_directory=str(directory),
                native_summary_sha256=sha(directory/'summary.jsonl'),
                sampling_coverage_proven=complete['sampling_coverage_proven']))
            write_json(self.a.out/'reuse-arms.json', self.arm_records)
        self.summary.update(reuse_generations_completed=9, reuse_sequences_completed=3,
                            reuse_native_completed=True, independent_raw_review_required=True)

    def run(self):
        code = super().run()
        if self.a.prepare:
            return code
        completed = (code == 0 and 'error' not in self.summary and self.summary.get('status') == 'component_complete_review_required' and
                     self.summary.get('reuse_native_completed') is True and
                     self.summary.get('restoration', {}).get('restored') is True and
                     len(getattr(self, 'arm_records', [])) == 3)
        self.summary.update(status='reuse_complete_independent_review_required' if completed else 'failed',
            reuse_growth_gate_pass=False, performance_measured=False, production_adoption=False)
        write_json(self.a.out/'summary.json', self.summary)
        self.event('reuse_decision_complete', status=self.summary['status'], reuse_growth_gate_pass=False,
                   independent_raw_review_required=True)
        return 0 if completed else 1


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stage', 'gold-stage', 'model', 'out', 'plan', 'inputs', 'reuse-plan',
                 'component-review', 'recovery-review'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--port', type=int, default=8187)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true'); mode.add_argument('--run', action='store_true')
    return parser.parse_args(argv)


def main():
    args = arguments(); need(sys.platform == 'darwin', 'macOS required')
    for name in ('stage', 'gold_stage', 'model', 'out', 'plan', 'inputs', 'reuse_plan',
                 'component_review', 'recovery_review'):
        path = getattr(args, name); need(path.is_absolute(), 'Absolute paths required')
        setattr(args, name, path.resolve())
    for stage in (args.stage, args.gold_stage):
        need(Path('/tmp').resolve() in stage.parents, 'Separate tmp stage required')
    args.out.mkdir(parents=True, exist_ok=False); os.chmod(args.out, 0o700)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try: os.setsid()
    except PermissionError: pass
    with (args.out/'runner.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2); os.dup2(null.fileno(), 0)
    return ReuseWindow(args).run()


if __name__ == '__main__':
    sys.exit(main())
