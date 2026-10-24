#!/usr/bin/env python3
"""CPU-only false-accept controls for the diagnostic allocation-site selector.

This does not initialize Metal, call the real allocator, or prove GPU recovery.
The native test's --run remains a separate exclusive-window operation.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stage = args.stage.resolve(strict=True)
    source = Path(__file__).with_name('test_moe_workspace_allocations.c').resolve(strict=True)
    # Include the actual selector implementation and replace only the entrypoint.
    wrapper = '''#define MYDS_MOE_WORKSPACE_TESTING 1
#define MYDS_MOE_WS_SELECTION_ONLY 1
#define main ws_original_main
#include %s
#undef main
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    int scenario=atoi(argv[1]);
    if(scenario==0){selection_host_test();return 0;}
    myds_metal_moe_workspace_allocation_t t[5]={0};
    for(unsigned i=0;i<5;i++){
        t[i].ordinal=i+1;t[i].success=1;t[i].kind=2;t[i].index=(int32_t)i;
        strcpy(t[i].site,"view_x");
    }
    if(scenario==1)t[3].ordinal=t[2].ordinal;
    if(scenario==2)t[0].ordinal=2;
    if(scenario==3)t[2].success=0;
    if(scenario==4)memset(t[2].site,'x',sizeof(t[2].site));
    if(scenario==5)t[2].site[0]=0;
    uint64_t selected[15]={0};size_t n=representatives(t,5,selected);
    if(scenario==6)require(n==3&&selected[0]==1&&selected[1]==3&&selected[2]==5,"odd count representative indices");
    if(scenario==7){n=representatives(t,1,selected);require(n==1&&selected[0]==1,"singleton owner");}
    if(scenario==8){n=representatives(t,2,selected);require(n==2&&selected[0]==1&&selected[1]==2,"two views");}
    if(scenario==9){n=representatives(t,4,selected);require(n==3&&selected[0]==1&&selected[1]==3&&selected[2]==4,"even count representatives");}
    return 0;
}
''' % json.dumps(str(source))
    results = []
    with tempfile.TemporaryDirectory(prefix='myds-moe-ws-cpu-') as temp:
        root = Path(temp); code = root / 'control.c'; binary = root / 'control'
        code.write_text(wrapper)
        command = ['/usr/bin/clang', '-O2', '-std=c11', '-Wall', '-Wextra', '-Werror',
                   '-I' + str(stage / 'include'), '-I' + str(stage / 'third_party'),
                   str(code), '-o', str(binary)]
        compiled = subprocess.run(command, capture_output=True, text=True)
        if compiled.returncode:
            raise SystemExit('CPU compile failed:\n' + compiled.stderr)
        names = ['positive_interleaved_sites', 'reject_duplicate_ordinal', 'reject_nonrelative_ordinal',
                 'reject_failed_dry_trace', 'reject_unterminated_site', 'reject_empty_site',
                 'positive_odd_count', 'positive_singleton', 'positive_two_views', 'positive_even_count']
        for index, name in enumerate(names):
            run = subprocess.run([str(binary), str(index)], capture_output=True, text=True)
            expected = 1 if 1 <= index <= 5 else 0
            passed = run.returncode == expected
            results.append(dict(name=name, expected_exit=expected, actual_exit=run.returncode,
                                passed=passed, stdout=run.stdout, stderr=run.stderr))
    report = dict(schema_version=1, scope='cpu_allocation_selector_controls_only',
                  metal_initialized=False, model_or_gpu_executed=False,
                  real_allocator_failure_tested=False, compile_returncode=compiled.returncode,
                  compile_stderr=compiled.stderr, controls=results,
                  all_pass=all(x['passed'] for x in results))
    if args.out:
        with args.out.open('x') as stream:
            json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps(report, indent=2))
    return 0 if report['all_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
