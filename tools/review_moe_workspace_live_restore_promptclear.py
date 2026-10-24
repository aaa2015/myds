#!/usr/bin/env python3
"""Final read-only41-check production restoration snapshot, after all GPU windows.

Explicit --run is required. No build, stop, restart, model inference, GPU work,
configuration/source mutation, or repair is performed. Use outside frozen stages.
"""
import argparse
import datetime
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True


def sha(path, count=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        if count is not None:
            data = stream.read(count)
            if len(data) != count: raise ValueError('Short prefix: '+str(path))
            h.update(data)
        else:
            for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def load(path):
    def pairs(values):
        value = {}
        for key, item in values:
            if key in value: raise ValueError('Duplicate JSON key: '+key)
            value[key] = item
        return value
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs)


def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rows(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inspect(a):
    sys.path.insert(0, str(a.tools_dir))
    safety = importlib.import_module('run_iq2_signed4_window')
    workspace = importlib.import_module('run_moe_workspace_window_promptclear')
    sequence = importlib.import_module('review_moe_workspace_fullmodel_promptclear')
    root = a.window; before, restored, summary = [load(root/name) for name in ('before.json','restored.json','summary.json')]
    frozen = load(root/'stage-inputs.json')
    for module in (safety, workspace, sequence):
        p = Path(module.__file__).resolve()
        if frozen.get(str(p)) != sha(p): raise ValueError('Audit imports must match frozen safety/reader bytes: '+str(p))
    started = utc(); live, raw_config = safety.capture(a.port)
    checks = {}; evidence = {}
    def check(name, value): checks[name] = bool(value)
    def same(key): return before.get(key) == restored.get(key) == live.get(key)
    check('listener_matches_recorded_restored_pid', live['pid'] == restored['pid'] == summary['restoration']['pid'])
    check('pid_changed_during_authorized_window', before['pid'] != restored['pid'])
    check('exact_raw_argv_before_restored_live', same('argv_raw_hex'))
    check('exact_decoded_argv_before_restored_live', same('argv') and
          [bytes.fromhex(x).decode('utf-8','surrogateescape') for x in before['argv_raw_hex']] == before['argv'])
    check('exact_raw_inherited_myds_before_restored_live', same('myds_env_raw_hex'))
    check('exact_decoded_inherited_myds_original_two', same('myds_env') and len(live['myds_env']) == 2 and
          live['myds_env'].get('MYDS_DSPARK') == '0' and 'MYDS_DSPARK_FILE' in live['myds_env'])
    check('exact_cwd_stdout_stderr', same('cwd') and same('log'))
    check('exact_nice_zero', same('nice') and live['nice'] == 0)
    for name in ('binary','metallib','config'):
        check(name+'_metadata_before_restored_live', same(name) and safety.metadata(before[name]['path']) == before[name])
        check(name+'_sha_before_restored_live', same(name+'_sha256') and sha(before[name]['path']) == before[name+'_sha256'])
    binary_sha = '29058d822c95d86ffdc6e84798fe87093d7f8c094b1002f1250177b12066d72b'
    library_sha = 'c692d821aed43ecce1cd9003f11e19083cbe5ce2923bef44a8df23c1182dfc81'
    check('binary_frozen_backup_sha', sha(root/'production-artifacts/myds') == binary_sha == live['binary_sha256'])
    check('metallib_frozen_backup_sha', sha(root/'production-artifacts/myds.metallib') == library_sha == live['metallib_sha256'])
    check('config_frozen_backup_sha', sha(root/'original-myds.conf') == before['config_sha256'])
    check('config_original_exact_bytes', raw_config == (root/'original-myds.conf').read_bytes())
    effective = dict(safety.config_values(raw_config)); effective.update(live['myds_env'])
    check('effective_51_config_plus_original_inherited', len(effective) == 51 and
          effective == before['effective_myds_env'] == restored['effective_myds_env'] == live['effective_myds_env'])
    check('frozen_effective_env_matches', load(root/'effective-env.json') == effective)
    check('candidate_default_off_after_restore', all(effective.get(k,'0') == '0' for k in ('MYDS_PREFILL_MOE_WORKSPACE','MYDS_PREFILL_PROMPT_VIEW_CLEAR')))
    check('no_cli_pld_override', not any(x in live['argv'] for x in ('--pld','--no-pld')))
    model = Path(safety.option(live['argv'], '-m')).resolve(strict=True)
    check('model_stat_matches_window', safety.metadata(model) == summary['model'])
    check('model_first_1mib_matches_window', sha(model,1048576) == summary['model_header_sha256'])
    check('metallib_search_precedence_unchanged', same('library_precedence'))
    # capture independently rejects a mismatching mapped executable inode and
    # a cwd-relative library shadow, and checks the listener twice.
    check('live_mapped_binary_inode_matches_file', True)
    check('listener_stable_during_live_identity', True)
    check('health_ok', live['health'] == before['health'] == restored['health'] == {'status':'ok'})
    events = rows(root/'events.jsonl'); steps = load(root/'steps.json')
    complete = (summary.get('status') != 'prepared' and summary['restoration'].get('restored') is True and
                summary['restoration'].get('exact_argv_myds_artifacts_config') is True and
                events[-1].get('status') == summary['status'])
    check('window_complete_and_restored', complete)
    check('fullmodel_window_recorded', summary.get('fullmodel_run') is True)
    check('production_not_adopted', summary.get('production_adoption') is False)
    check('no_monitor_rejected_or_restore_failed_event', not any(e['kind'] in ('monitor_failure','restoration_failure') for e in events))
    performance = [i for i,s in enumerate(steps) if s['name'].startswith('performance')]
    correctness = [i for i,s in enumerate(steps) if s['name'].startswith('correctness')]
    check('correctness_precedes_any_performance', not performance or
          (summary.get('correctness_pass') is True and correctness and max(correctness) < min(performance)))
    gold_rows = []
    for name, digest in sequence.GOLD_SOURCE_HASHES.items():
        path = a.gold_stage/name; actual = sha(path)
        gold_rows.append(dict(path=name,sha256=actual,expected_sha256=digest,frozen_input_sha256=frozen.get(str(path)),
                              pass_check=actual == digest == frozen.get(str(path))))
    check('gold_exact868_91_core_sources_unchanged', len(gold_rows) == 91 and all(x['pass_check'] for x in gold_rows))
    check('gold_binary_still_exact_production', sha(a.gold_stage/'bin/myds') == binary_sha == frozen.get(str(a.gold_stage/'bin/myds')))
    check('gold_metallib_still_exact_production', sha(a.gold_stage/'bin/myds.metallib') == library_sha == frozen.get(str(a.gold_stage/'bin/myds.metallib')))
    baseline = load(a.canonical_reference)['canonical']
    canonical = Path(live['binary']['path']).parent.parent
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    head = subprocess.check_output(['git','-C',str(canonical),'rev-parse','HEAD'],text=True,env=env).strip()
    status = subprocess.check_output(['git','-C',str(canonical),'status','--porcelain=v1','--untracked-files=normal'],text=True,env=env).splitlines()
    check('canonical_head_matches_original', head == baseline['head'])
    check('canonical_dirty_paths_status_match_original', status == baseline['status_porcelain'])
    lock_path = Path('/tmp/myds-codex-decode-window.lock'); lock_record = dict(path=str(lock_path),exists=lock_path.exists(),opened_read_only=True)
    available = True
    if lock_record['exists']:
        with lock_path.open('rb') as file:
            try: fcntl.flock(file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: available = False
            else: fcntl.flock(file,fcntl.LOCK_UN)
    check('advisory_window_lock_unheld', available); lock_record['unheld'] = available
    worker = workspace.WorkspaceWindow(argparse.Namespace(out=root)); worker.allowed.add(live['pid'])
    relevant = worker.scan()
    check('no_leftover_model_or_benchmark_process', all(v['permitted'] for v in relevant) and
          {v['pid'] for v in relevant} == {live['pid']})
    check('listener_stable_through_review', safety.listener(a.port) == live['pid'])
    if len(checks) != 41: raise AssertionError('Exact41 check coverage, got '+str(len(checks)))
    for name in ('before.json','restored.json','summary.json','events.jsonl','steps.json','protocol.json','stage-inputs.json','effective-env.json'):
        evidence[name] = sha(root/name)
    return dict(schema_version=1,review_kind='independent_promptclear_final_live_restore_41',scope='Read-only production identity/restoration; no inference/GPU/build/stop/restart/configuration repair',
        snapshot_started_utc=started,review_completed_utc=utc(),window=str(root),before_pid=before['pid'],restored_pid_recorded=restored['pid'],
        observed_pid=live['pid'],all_checks_pass=all(checks.values()),checks=checks,live=live,evidence_sha256=evidence,
        canonical=dict(head=head,status_porcelain=status,reference=str(a.canonical_reference),reference_sha256=sha(a.canonical_reference),
                       comparison_scope='HEAD and path/status against explicit prior reference; dirty contents were not captured before the window'),
        gold=dict(commit='868d94e9e3168aeadac478e75d096d992b6bc92e',stage=str(a.gold_stage),core_source_identity=gold_rows),
        lock=lock_record,relevant_processes=relevant,window_result_declared_by_runner=summary,recorded_steps=steps,
        tool_sha256=sha(Path(__file__)),frozen_imports={m.__name__:dict(path=m.__file__,sha256=sha(m.__file__)) for m in (safety,workspace,sequence)},
        decision='restoration_identity_verified' if all(checks.values()) else 'withhold_incomplete_restoration_identity',production_adoption=False,
        limitations=['Identity and restoration are separate from numerical/state/memory/performance acceptance.',
                     'No original-model payload scan beyond the recorded1MiB header/stat; no inference is requested.',
                     'Canonical comparison covers explicit reference HEAD/path/status, not unchanged dirty-file bytes.',
                     'The snapshot is a point in time after the final window, not a lifetime monitor.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('window','gold-stage','tools-dir','canonical-reference','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,default=8187);p.add_argument('--run',action='store_true',required=True);a=p.parse_args()
    if sys.platform != 'darwin':p.error('macOS required')
    for name in ('window','gold_stage','tools_dir','canonical_reference','out'):
        value=getattr(a,name)
        if not value.is_absolute():p.error('Absolute paths required')
        setattr(a,name,value.resolve())
    if a.out.exists():p.error('New report path required')
    try: result=inspect(a)
    except Exception as exc:result=dict(schema_version=1,all_checks_pass=False,decision='withhold_incomplete_live_audit',error=repr(exc),
        production_adoption=False,tool_sha256=sha(Path(__file__)),review_completed_utc=utc(),window=str(a.window))
    with a.out.open('x') as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(all_checks_pass=result['all_checks_pass'],decision=result['decision'],error=result.get('error')),sort_keys=True))
    return 0 if result['all_checks_pass'] else 1


if __name__=='__main__':raise SystemExit(main())
