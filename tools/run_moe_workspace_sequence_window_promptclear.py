#!/usr/bin/env python3
"""Normal localgrow probe or one frozen full-sequence ABA cohort.

Builds and source edits are external. The inherited isolated window owns lock,
process/memory monitoring and finally restoration. All native raw stays intact.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sys
sys.dont_write_bytecode = True
import run_moe_workspace_window_promptclear as workspace
import review_moe_workspace_fullmodel_promptclear as review

need, sha, write_json = workspace.need, workspace.sha, workspace.write_json
FLAG = workspace.FLAG
CLEAR_FLAG = workspace.CLEAR_FLAG
ORDER = (('A1', 0), ('B', 1), ('A2', 0))
NATIVE = 'bin/bench_moe_workspace_fullmodel_promptclear'
CORE = ('src/myds_v41.c', 'src/myds_metal.m', 'include/myds_internal.h', 'include/myds_metal.h')
REQUIRED = (NATIVE, 'tools/bench_moe_workspace_fullmodel_promptclear.c',
            'tools/moe_workspace_fullmodel_promptclear.mk', 'tools/moe_workspace_fullmodel_getters.inc',
            'tools/moe_workspace_sequence_state_localgrow.inc', 'tools/moe_workspace_layout_promptclear.inc',
            'tools/moe_workspace_memory.inc', 'tools/bench_moe_workspace_fullmodel.c',
            'tools/run_moe_workspace_sequence_window_promptclear.py',
            'tools/review_moe_workspace_fullmodel_promptclear.py','tools/moe_prompt_lifetime_diagnostic.inc')
FORBIDDEN = ('myds_metal_moe_workspace_test_control', 'myds_v41_moe_workspace_probe_',
             'myds_v41_moe_workspace_cancel_control', 'myds_v41_prompt_workspace_test_', 'myds_v41_prompt_workspace_probe_')


def sequence_plan(plan):
    need(plan['prompt_view_clear_flag'] == CLEAR_FLAG and plan['fullmodel']['prompt_view_clear_flags']=={'A1':0,'B':1,'A2':0}, 'Frozen new clear flag/arms')
    f = plan['fullmodel']; p = f['performance']; a = f['adoption']; c = f['correctness']; n = plan['normal_success_probe']
    need(plan['implementation_version'] == 4 and plan['flag'] == FLAG and plan['default_off'] is True and
         f['max_seq_len'] == 327680 and f['max_new_tokens'] == 32 and f['temperature'] == 0 and f['seed'] == 42 and
         f['fresh_context_per_sequence'] is True and f['same_context_three_stages'] is True and
         [s['stage'] for s in f['sequence']] == ['setup', 'grow', 'retain'] and
         [s['request'] for s in f['sequence']] == ['pf7k', 'pf16k', 'pf7k'], 'Unique same-context v3 scope differs')
    need(c['order'] == ['gold', 'off', 'on'] and c['warmup'] == 1 and c['formal_samples'] == 1 and
         c['total_generations'] == 18 and p['order'] == ['A1', 'B', 'A2'] and
         p['flags'] == dict(ORDER) and p['warmup'] == 1 and p['formal_samples'] == 3 and
         p['repeat_limit'] == 1 and p['total_generations'] == 36 and
         p['maximum_absolute_prefill_and_decode_control_drift_percent'] == 2 and
         p['maximum_conservative_time_regression_percent'] == 1 and p['memory_sampling_interval_ms'] == 10 and
         p['physical_memory_control_drift_limit_bytes'] == 268435456, 'Frozen counts/time/physical control gates differ')
    need(a['minimum_conservative_peak_footprint_reduction_bytes'] == 268435456 and
         a['minimum_conservative_prefill_time_saved_percent_strict'] == 2 and
         a['max_footprint_regression_bytes'] == 134217728 and a['required_primary_workloads'] == ['grow', 'retain'] and
         a['setup_nonregression_required'] is True and a['setup_material_gain_threshold'] is False and
         a['total_sequence_wall_control_drift_limit_percent'] == 2 and
         a['total_sequence_wall_regression_limit_percent'] == 1 and
         a['known_growth_lifetime_checkpoint_required'] is True and a['localgrow_non_moe_identity_reuse_required'] is True,
         'Frozen independent per-stage gain and complete-sequence gates differ')
    need(n['order'] == ['A1', 'B', 'A2'] and n['sequences_per_arm'] == 1 and n['max_new_tokens'] == 8 and
         n['performance_measured'] is False and n['required_before_formal'] is True and
         n['all_state_bytes_equal'] is True and n['localgrow_id_reuse_and_checkpoint_required'] is True,
         'Frozen prerequisite normal probe differs')


class SequenceWindow(workspace.WorkspaceWindow):
    def preflight(self):
        super().preflight(); sequence_plan(self.plan)
        self.arm_records = []; self.arm_reviews = []
        self.requests = {'setup': self.a.inputs/'pf7k.json', 'grow': self.a.inputs/'pf16k.json',
                         'retain': self.a.inputs/'pf7k.json'}
        need(str(Path(__file__).resolve()) in self.input_hashes and
             str(Path(review.__file__).resolve()) in self.input_hashes, 'Execute the actual frozen controller and reader')
        for stage in (self.a.stage, self.a.gold_stage):
            for name in (REQUIRED if stage == self.a.stage else tuple(n for n in REQUIRED if n not in ('tools/run_moe_workspace_sequence_window_promptclear.py','tools/review_moe_workspace_fullmodel_promptclear.py'))):
                need(str(stage/name) in self.input_hashes and (stage/name).is_file(), 'Required normal source/bin not frozen: '+str(stage/name))
        self.native_identity = {}
        for label, stage in (('candidate', self.a.stage), ('gold', self.a.gold_stage)):
            symbols = workspace.safety.command(['/usr/bin/nm', '-gU', str(stage/NATIVE)])
            need(not any(n in symbols for n in FORBIDDEN), 'Normal binary exposes TESTING/fault/cancel API: '+label)
            if label == 'candidate':
                need(all(n in symbols for n in ('myds_v41_prompt_workspace_growth_observer_set', 'myds_v41_prefill_prompt_workspace_snapshot', 'myds_v41_prefill_moe_workspace_layout',
                     'myds_v41_prefill_moe_workspace_binding', 'myds_v41_moe_workspace_growth_observer_set')),
                     'Candidate normal identity/observer APIs absent')
            self.native_identity[label] = {name: sha(stage/name) for name in (*CORE, NATIVE, 'bin/myds.metallib')}
            write_json(self.a.out/(label+'-normal-symbols.json'), dict(binary_sha256=sha(stage/NATIVE),
                       forbidden_symbols=list(FORBIDDEN), forbidden_symbols_present=False, nm_stdout=symbols))
        prerequisites = {}
        for path, key in ((self.a.component_review, 'component_gate_pass'), (self.a.recovery_review, 'recovery_gate_pass'), (self.a.causal_review, 'causal_lifetime_gate_pass')):
            value = review.load(path)
            need(value.get('evidence_valid') is True and value.get(key) is True, 'Valid independent prerequisite required: '+key)
            if key == 'component_gate_pass': workspace.validate_component_prerequisite(value,self.a.stage,self.summary,self.before)
            elif key == 'recovery_gate_pass': need(value.get('implementation_version')==4 and value.get('plan_sha256')==workspace.PLAN_SHA and value.get('recovery_plan_sha256') == workspace.PLAN_SHA and value.get('component_review_sha256')==sha(self.a.component_review), 'Recovery belongs to this exact frozen plan/component prerequisite')
            else:
                need(value.get('mode')=='causal_diagnostic' and value.get('implementation_version')==4 and value.get('plan_sha256')==workspace.PLAN_SHA and
                     value.get('native_identity',{}).get('candidate')==self.native_identity['candidate'] and
                     value.get('production_environment')==self.before['effective_myds_env'] and value.get('model')==self.summary['model'] and
                     value.get('model_header_sha256')==self.summary['model_header_sha256'], 'Causal proof exact build/model/environment differs')
            workspace.freeze_independent_long(value, 'full_report' if key=='causal_lifetime_gate_pass' else 'long_report', self.input_hashes)
            self.input_hashes[str(path)] = sha(path); prerequisites[key] = dict(path=str(path), sha256=sha(path))
        if self.a.mode == 'full':
            need(self.a.probe_review is not None, 'Full mode requires prior normal probe review')
            value = review.load(self.a.probe_review)
            need(value.get('evidence_valid') is True and value.get('growth_gate_pass') is True and
                 value.get('mode') == 'probe' and value.get('plan_sha256') == workspace.PLAN_SHA and
                 value.get('native_identity') == self.native_identity and
                 {k:v.get('sha256') for k,v in value.get('prerequisites', {}).items()} ==
                 {k:v['sha256'] for k,v in prerequisites.items()} and value.get('production_environment') == self.before['effective_myds_env'] and
                 value.get('model') == self.summary['model'] and
                 value.get('model_header_sha256') == self.summary['model_header_sha256'],
                 'Prior complete normal probe must bind this exact plan/build/model/environment/prerequisites')
            self.input_hashes[str(self.a.probe_review)] = sha(self.a.probe_review)
        else: need(self.a.probe_review is None, 'Probe mode does not consume or replace another probe review')
        self.prerequisites = prerequisites
        write_json(self.a.out/'stage-inputs.json', self.input_hashes)
        self.summary.update(component='normal_moe_workspace_promptclear_sequence_v4', mode=self.a.mode,
            sequence_run=False, fullmodel_run=False, performance_measured=False, correctness_pass=False,
            growth_gate_pass=self.a.mode == 'full', performance_gate_pass=False, prerequisites=prerequisites,
            prior_probe_growth_gate_pass=self.a.mode == 'full',
            native_identity=self.native_identity, normal_observer_cpu_cost_in_wall=True)

    def arm(self, phase, label, setting, gold=False):
        repeats = {'probe': 0, 'correctness': 1, 'performance': 3}[phase]
        directory = self.a.out/'arms'/(phase+'-'+label)
        need(not directory.exists(), 'Preserve raw: native output directory must be new')
        stage = self.a.gold_stage if gold else self.a.stage
        env = dict(self.environment, **{FLAG: str(setting), CLEAR_FLAG: str(setting)})
        expected = dict(self.before['effective_myds_env'], **{FLAG: str(setting), CLEAR_FLAG: str(setting)})
        need({k: v for k, v in env.items() if k.startswith('MYDS_')} == expected,
             'Preserve all51 effective production settings except explicit candidate flag')
        write_json(self.a.out/'arm-environments'/(phase+'-'+label+'.json'), expected)
        argv = [stage/NATIVE, '--run', '--model', self.a.model, '--warm-request', self.requests['setup'],
                '--grow-request', self.requests['grow'], '--retain-request', self.requests['retain'],
                '--out-dir', directory, '--phase', phase, '--repeats', str(repeats)]
        saved = self.a.stage; self.a.stage = stage
        try: self.run_command(phase+'-'+label, argv, env=env, timeout=3600)
        finally: self.a.stage = saved
        result = review.read_arm(directory, phase, setting, self.requests, expected, gold=gold, expected_model=self.a.model)
        need(Path(result['metadata']['actual_metallib_path']).resolve(strict=True) == (stage/'bin/myds.metallib').resolve(strict=True),
             'Actual first-priority library differs from the expected native stage')
        self.arm_reviews.append(result)
        record = dict(label=label, phase=phase, setting=setting, gold=gold, directory=str(directory),
                      native_summary_sha256=sha(directory/'summary.jsonl'), generations=3*(repeats+1),
                      native_binary_sha256=sha(stage/NATIVE), metallib_sha256=sha(stage/'bin/myds.metallib'))
        self.arm_records.append(record)
        write_json(directory/'controller-review.json', result); write_json(self.a.out/'arms.json', self.arm_records)
        return result

    def component(self):
        self.summary.update(sequence_run=True, fullmodel_run=True, performance_measured=False)
        (self.a.out/'arms').mkdir(); (self.a.out/'arm-environments').mkdir()
        if self.a.mode == 'probe':
            arms = [self.arm('probe', label, setting) for label, setting in ORDER]
            self.raw_comparisons = [review.compare_arms(arms[0], arm) for arm in arms]
            self.gate = review.growth_gate(*arms)
            need(type(self.gate.get('growth_gate_pass')) is bool, 'Reader must explicitly decide the growth gate')
            self.summary.update(correctness_pass=True, probe_generations_completed=9,
                                growth_gate_pass=self.gate['growth_gate_pass'], growth=self.gate)
        else:
            gold = self.arm('correctness', 'gold', 0, gold=True)
            controls = [self.arm('correctness', label, setting) for label, setting in (('off', 0), ('on', 1))]
            self.raw_comparisons = [review.compare_arms(gold, arm) for arm in (gold, *controls)]
            self.summary.update(correctness_pass=True, correctness_generations_completed=18)
            write_json(self.a.out/'correctness-gate.json', dict(correctness_pass=True, comparisons=self.raw_comparisons))
            self.summary['performance_measured'] = True
            arms = [self.arm('performance', label, setting) for label, setting in ORDER]
            self.raw_comparisons += [review.compare_arms(gold, arm) for arm in arms]
            self.gate = review.performance_gate(*arms)
            need(type(self.gate.get('performance_gate_pass')) is bool, 'Reader must explicitly decide the performance gate')
            self.summary.update(performance_gate_pass=self.gate['performance_gate_pass'], performance=self.gate,
                                performance_generations_completed=36, total_generations_completed=54)
        self.summary['normal_sequence_native_complete'] = True

    def run(self):
        code = super().run()
        if self.a.prepare: return code
        complete = (code == 0 and 'error' not in self.summary and self.summary.get('status') == 'component_complete_review_required' and
                    self.summary.get('normal_sequence_native_complete') is True and self.summary.get('restoration', {}).get('restored') is True and
                    len(getattr(self, 'arm_records', [])) == (3 if self.a.mode == 'probe' else 6))
        key = 'growth_gate_pass' if self.a.mode == 'probe' else 'performance_gate_pass'
        passed = complete and self.summary.get(key) is True
        status = (self.a.mode+'_passed_review_required' if passed else self.a.mode+'_rejected_gain_or_controls') if complete else 'failed'
        self.summary.update(status=status, evidence_valid=complete, production_adoption=False,
                            **{key: passed})
        report = dict(schema_version=1, mode=self.a.mode, evidence_valid=complete, production_adoption=False,
            plan_sha256=self.summary.get('plan_sha256'), model=self.summary.get('model'),
            model_header_sha256=self.summary.get('model_header_sha256'),
            production_environment=self.before.get('effective_myds_env') if self.before else None,
            native_identity=getattr(self, 'native_identity', {}), prerequisites=getattr(self, 'prerequisites', {}),
            arms=getattr(self, 'arm_records', []), raw_comparisons=getattr(self, 'raw_comparisons', []),
            gate=getattr(self, 'gate', {}), restoration=self.summary.get('restoration'),
            performance_measured=self.summary.get('performance_measured', False), **{key: passed},
            limits=['No automatic deployment/adoption; a valid negative result remains a rejected gate.',
                    'Full original native raw is preserved. Reader recomputes state, actual provider, layout bindings, memory and gates.',
                    'Observed peaks include synchronous lifetime events; requested-byte formulas are not physical evidence.'])
        if 'error' in self.summary: report['error'] = self.summary['error']
        write_json(self.a.out/('probe-review.json' if self.a.mode == 'probe' else 'fullmodel-review.json'), report)
        write_json(self.a.out/'summary.json', self.summary)
        self.event('sequence_decision_complete', status=status, evidence_valid=complete, production_adoption=False, **{key: passed})
        return 0 if complete else 1


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('stage', 'gold-stage', 'model', 'out', 'plan', 'inputs', 'component-review', 'recovery-review', 'causal-review'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--probe-review', type=Path); p.add_argument('--mode', choices=('probe', 'full'), required=True)
    p.add_argument('--port', type=int, default=8187)
    run = p.add_mutually_exclusive_group(required=True)
    run.add_argument('--prepare', action='store_true'); run.add_argument('--run', action='store_true')
    a = p.parse_args(argv)
    need(a.mode != 'full' or a.probe_review is not None, 'Full mode requires --probe-review')
    return a


def main():
    a = arguments(); need(sys.platform == 'darwin', 'macOS required')
    for name in ('stage', 'gold_stage', 'model', 'out', 'plan', 'inputs', 'component_review', 'recovery_review', 'causal_review', 'probe_review'):
        path = getattr(a, name)
        if path is not None:
            need(path.is_absolute(), 'Absolute paths required'); setattr(a, name, path.resolve())
    need(all(Path('/tmp').resolve() in p.parents for p in (a.stage, a.gold_stage)), 'Separate tmp stages required')
    a.out.mkdir(parents=True, exist_ok=False); os.chmod(a.out, 0o700); signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try: os.setsid()
    except PermissionError: pass
    with (a.out/'runner.log').open('ab', buffering=0) as log, open(os.devnull, 'rb') as null:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2); os.dup2(null.fileno(), 0)
    return SequenceWindow(a).run()


if __name__ == '__main__': raise SystemExit(main())
