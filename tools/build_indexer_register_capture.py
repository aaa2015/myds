#!/usr/bin/env python3
"""CPU-build an independent correctness-only real-capture register-only replay.

The old replay is kept byte exact in the snapshot. Only reversible edits below
produce a separate unit. No binary, GPU API, service, or full model is run.
"""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import subprocess

ORIGINAL = 'tools/bench_indexer_k_capture.m'
GENERATED = 'tools/bench_indexer_register_capture.m'
ORIGINAL_SHA = '31f63df2a6fcbe3bd0d12a76723828c7e8f1a51cd981fa4f58da2941b669cefe'
SYMBOL = 'kernel_myds_indexer_scores_register_only_bf16'
CORE_SHA = 'c403c38f2d254d0d4e113afad0e27e32ddf6684d0faeee2d70cac9d9001303f5'


def sha(b):
    return hashlib.sha256(b).hexdigest()


def descriptor(p):
    b = Path(p).read_bytes()
    return dict(path=str(Path(p).resolve()), bytes=len(b), sha256=sha(b))


def inverse(generated, edits):
    for e in reversed(edits):
        a, b = e['before'].encode(), e['after'].encode()
        if generated.count(b) != 1:
            raise ValueError('unique inverse anchor: ' + e['name'])
        generated = generated.replace(b, a, 1)
    return generated


def generate(original):
    if sha(original) != ORIGINAL_SHA:
        raise ValueError('frozen v4 original replay SHA required')
    out, edits = original, []
    def edit(name, before, after):
        nonlocal out
        a,b = before.encode(),after.encode()
        if not a or out.count(a) != 1 or not b:
            raise ValueError('unique nonempty edit anchor: '+name)
        out=out.replace(a,b,1);edits.append(dict(name=name,before=before,after=after))
    edit('register subject', '@"probe":@"indexer_k_resident_real_capture",',
         '@"probe":@"indexer_register_only_real_capture",@"correctness_only":@YES,@"timing_collected":@NO,\n'
         '        @"selected_symbol":@"'+SYMBOL+'",@"candidate_dynamic_threadgroup_bytes":@0,\n'
         '        @"adoption_eligible":@NO,@"production_registered":@NO,')
    edit('no samples or warmup', 'BOOL host_only=NO;unsigned samples=9;', 'BOOL host_only=NO;const unsigned samples=0;')
    edit('reject samples CLI', '        else if(!strcmp(key,"--samples"))samples=number(v,9,9);\n',
         '        /* No timing/samples option exists in this correctness-only binary. */\n')
    edit('zero warmups', '@"samples":@(samples),@"warmups":@1,', '@"samples":@(samples),@"warmups":@0,')
    edit('BF16 exact captured scope', '    f.bf16=[c[@"cache_bf16"]boolValue];need(x->index_row_stride==(f.bf16?256u:512u),"real cache stride");',
         '    f.bf16=[c[@"cache_bf16"]boolValue];need(f.bf16&&x->index_row_stride==256u,"register replay BF16 cache stride");')
    edit('actual exact captured provider only', '    need(tiled||exact||direct,"captured actual production provider");',
         '    need(exact&&!tiled&&!direct,"captured actual exact provider only");')
    edit('candidate no TG encoder binding', '    [e setThreadgroupMemoryLength:provider?21504:f->smem atIndex:0];',
         '    if(provider!=2u)[e setThreadgroupMemoryLength:provider?21504:f->smem atIndex:0];')
    edit('score completion witnesses', '    need(cb.status==MTLCommandBufferStatusCompleted,"real replay GPU completed");return cb;',
         '    static unsigned ordinal;ordinal++;manifest[@"score_gpu_completion_count"]=@(ordinal);\n'
         '    emit(@{@"kind":@"real_score_completion",@"ordinal":@(ordinal),@"layer":@(f->layer),@"provider":@(provider),\n'
         '        @"status":@(cb.status),@"completed":cb.status==MTLCommandBufferStatusCompleted?@YES:@NO,\n'
         '        @"error":cb.error?cb.error.localizedDescription:(id)[NSNull null]});\n'
         '    need(cb.status==MTLCommandBufferStatusCompleted,"real replay GPU completed");return cb;')
    edit('consumer full post score bytes', '        emit(@{@"kind":@"real_topk",',
         '        raw([artifact_prefix stringByAppendingString:@"-topk-score-post.f32"],myds_metal_buffer_data(scores),sb);\n'
         '        emit(@{@"kind":@"real_topk",')
    edit('register PSO and complete pipeline bindings',
         '        psos[ci][2]=real_index_pso(d,probe,[@"kernel_myds_indexer_scores_k_resident_"stringByAppendingString:suffix],21504);',
         '        psos[ci][2]=real_index_pso(d,probe,@"'+SYMBOL+'",0);\n'
         '        need(psos[ci][2].staticThreadgroupMemoryLength==0,"candidate truly zero static and dynamic TG");\n'
         '        NSString *symbols[]={f->provider,@"kernel_dsv4_indexer_scores_tiled_bf16",@"'+SYMBOL+'"};\n'
         '        for(unsigned p=0;p<3;p++)emit(@{@"kind":@"real_pipeline",@"layer":@(f->layer),@"provider":@(p),\n'
         '            @"symbol":symbols[p],@"library_sha256":file_sha(p?probe_path:native_path),\n'
         '            @"argument_bytes":@(p?72:f->native_args.length),@"threads":@[@32,@4,@1],\n'
         '            @"grid":@[@(p?(f->args.n_comp+31)/32:f->grid[0]),@(p?(f->args.n_tokens+7)/8:f->grid[1]),@1],\n'
         '            @"dynamic_threadgroup_bytes":@(p==2?0:p==1?21504:f->smem),\n'
         '            @"static_threadgroup_bytes":@(psos[ci][p].staticThreadgroupMemoryLength)});')
    edit('actual GPU input preimages',
         '        for(unsigned p=0;p<3;p++) {\n            (void)real_index_run',
         '        all_numerical&=real_index_inputs(f,@"phase0_pre_dispatch");\n'
         '        for(unsigned p=0;p<3;p++) {\n            (void)real_index_run')
    timing = out[out.index(b'    real_timing_phase=YES;'):].decode()
    edit('unconditional return before all timing; timing phase absent', timing,
         '    manifest[@"timing_collected"]=@NO;\n'
         '    myds_metal_cleanup();finish(@"correctness_only_complete");return 0;\n}}\n')
    if inverse(out,edits)!=original:
        raise ValueError('complete old replay inverse mismatch')
    if any(x in out for x in (b'@"real_timing"',b'@"real_summary"',b'GPUStartTime',b'GPUEndTime',b'for(unsigned sample=')):
        raise ValueError('timing must be entirely absent')
    return out,dict(schema_version=1,kind='register_capture_correctness_reversible_source_transform',
                    exact_edits=edits,edit_count=len(edits),original_sha256=sha(original),
                    generated_sha256=sha(out),inverse_restored_sha256=sha(original),
                    inverse_restores_original_byte_for_byte=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--emit-only',action='store_true')
    a=p.parse_args();root=a.source_root.resolve(strict=True);out=a.out_dir.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError('new private output directory outside source required')
    original=(root/ORIGINAL).read_bytes();generated,proof=generate(original)
    out.mkdir(parents=True);snapshot=out/'source'
    paths=[f for directory in ('src','include','third_party') for f in (root/directory).rglob('*')
           if f.is_file() and f.suffix in ('.c','.h','.m','.inc')]
    paths += [root/ORIGINAL,root/'tools/bench_attention_multihead.m']
    for path in sorted(set(paths)):
        b=path.read_bytes();dest=snapshot/path.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
    for name in ('build_indexer_register_capture.py','review_indexer_k_capture.py'):
        src=Path(__file__).resolve().parent/name;dest=snapshot/'tools'/name;dest.write_bytes(src.read_bytes())
    if sha((snapshot/'tools/review_indexer_k_capture.py').read_bytes())!=CORE_SHA:
        raise ValueError('frozen independent core SHA')
    (snapshot/GENERATED).write_bytes(generated)
    (out/'source-transform.json').write_text(json.dumps(proof,indent=2)+'\n')
    (out/'generated-source.diff').write_text(''.join(difflib.unified_diff(original.decode().splitlines(True),generated.decode().splitlines(True),fromfile=ORIGINAL,tofile=GENERATED)))
    inventory={str(f.relative_to(snapshot)):dict(bytes=f.stat().st_size,sha256=sha(f.read_bytes())) for f in sorted(snapshot.rglob('*')) if f.is_file()}
    common=['xcrun','clang','-O3','-Wall','-Wextra','-mmacosx-version-min=26.0','-I'+str(snapshot/'include'),'-I'+str(snapshot/'third_party')]
    commands=[];objects=[]
    for rel in ('src/myds_v41_gpu.c','src/myds_gguf.c','src/myds_log.c','src/myds_configfile.c','src/myds_metal.m',GENERATED):
        obj=out/(Path(rel).stem+'.o');flags=['-fobjc-arc','-Wno-deprecated-declarations'] if rel.endswith('.m') else ['-std=c11']
        commands.append(common+flags+['-c',str(snapshot/rel),'-o',str(obj)]);objects.append(str(obj))
    binary=out/'bench_indexer_register_capture'
    commands.append(['xcrun','clang','-mmacosx-version-min=26.0',*objects,'-framework','Foundation','-framework','Metal','-lm','-lpthread','-o',str(binary)])
    manifest=dict(schema_version=1,kind='indexer_register_only_real_capture_build',subject='indexer_register_only_head_control_v2',
                  selected_symbol=SYMBOL,correctness_only=True,timing_present=False,topk_api='myds_v41_gpu_topk_causal',
                  source_inventory=inventory,source_inventory_sha256=sha(json.dumps(inventory,sort_keys=True,separators=(',',':')).encode()),
                  original_seed=ORIGINAL,generated_source=GENERATED,source_transform=descriptor(out/'source-transform.json'),
                  original_replay_sha256=sha(original),generated_source_sha256=sha(generated),commands=commands,
                  compile_only_pass=False,gpu_work_submitted=False,binary_executed=False,production_registered=False,
                  compilation_skipped=a.emit_only,compile_environment={k:os.environ.get(k) for k in ('DEVELOPER_DIR','SDKROOT','CPATH','C_INCLUDE_PATH','LIBRARY_PATH')})
    target=out/'build-manifest.json'
    try:
        if not a.emit_only:
            with (out/'compile.log').open('w') as log:
                for command in commands:
                    log.write(json.dumps(command)+'\n');log.flush();subprocess.run(command,stdout=log,stderr=log,check=True)
            manifest.update(compile_only_pass=True,binary=descriptor(binary))
        if (root/ORIGINAL).read_bytes()!=original:
            raise ValueError('original replay changed during build')
    except Exception as e:
        manifest['build_error']=repr(e);raise
    finally:
        target.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(build_manifest=descriptor(target),compile_only_pass=manifest['compile_only_pass'],gpu_work_submitted=False),indent=2))


if __name__=='__main__':main()
