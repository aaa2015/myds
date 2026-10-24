#!/usr/bin/env python3
"""CPU-only steady Output_A protocol and pre-GPU gate rejection controls.

All GGUF weights, captures and claimed correctness records here are synthetic
parser fixtures, not model evidence. Every invocation includes --host-only;
the tests assert that no Metal device is created. No GPU work is submitted.
"""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def string(value):
    data=value.encode()
    return struct.pack('<Q',len(data))+data


def main():
    binary=Path(sys.argv[1] if len(sys.argv)>1 else 'bin/bench_output_a_fixed_steady').resolve()
    check=subprocess.run([str(binary),'--cpu-self-test'],capture_output=True,text=True,check=True)
    assert json.loads(check.stdout)['rejected_invalid_bindings']==8
    print(check.stdout.strip())
    with tempfile.TemporaryDirectory(prefix='myds-output-a-host-') as temp:
        root=Path(temp);model=root/'sparse-cpu-fixture.gguf';weight_bytes=18874368
        header=struct.pack('<IIQQ',0x46554747,3,1,1)
        header+=string('general.architecture')+struct.pack('<I',8)+string('deepseek41')
        header+=string('blk.0.attn_output_a.weight')+struct.pack('<IQQIQ',2,4096,8192,12,0)
        header+=b'\0'*(-len(header)%32)
        with model.open('wb') as stream:
            stream.write(header);stream.truncate(len(header)+weight_bytes)
        x=struct.pack('<32768I',*([0x3f800000]*32768));(root/'input.f32').write_bytes(x)
        capture=dict(schema_version=1,capture_kind='myds_decode_output_a',
            model_bytes=model.stat().st_size,model_header_bytes=len(header),model_header_sha256=digest(header),
            source=dict(kind='synthetic_cpu_only_parser_control_not_inference'),
            cases=[dict(layer=0,position=0,call_ordinal=1,need_logits=True,ordinary_decode_allowed=True,
                trace_on=False,attention_rope_consumed=True,compressed=True,weight_type=12,
                boundary='post_inverse_rope_bf16_before_output_a',observer_commit_wait=True,
                observer_batch_restored=True,performance_from_capture=False,
                input_f32=dict(path='input.f32',bytes=len(x),sha256=digest(x)))])
        probe=root/'probe.metallib';production=root/'production.metallib'
        probe.write_bytes(b'CPU-only-not-a-library');production.write_bytes(b'CPU-only-not-production')
        capture_path=root/'capture.json';capture_path.write_text(json.dumps(capture))

        def run(label,value,expected,extra=()):
            capture_path.write_text(json.dumps(value));out=root/label
            result=subprocess.run([str(binary),'--model',str(model),'--capture-json',str(capture_path),
                '--probe-metallib',str(probe),'--production-metallib',str(production),'--out-dir',str(out),
                '--host-only','--samples','15','--reps','256',*map(str,extra)],capture_output=True,text=True)
            assert (result.returncode==0)==expected,(label,result.returncode,result.stdout,result.stderr)
            path=out/'artifact-manifest.json';record=json.loads(path.read_text()) if path.exists() else None
            if record:
                assert record['status']==('host_only' if expected else 'failed')
                assert 'device' not in record and record['host_invalid_bindings_rejected']==8
            if expected:
                complete=json.loads(result.stdout.splitlines()[-1])
                assert complete['host_only'] and not complete['metal_device_created']
            print(json.dumps(dict(case=label,passed=True,expected_success=expected,gpu_executed=False)))
            return out,record

        positive,record=run('valid-host-only',capture,True)
        assert record['samples']==15 and record['repetitions_per_cb']==256
        for label,extra in [('old-samples-rejected',('--samples','9')),
                            ('old-reps-rejected',('--reps','8')),
                            ('wrong-samples-rejected',('--samples','14')),
                            ('wrong-reps-rejected',('--reps','255'))]:
            run(label,capture,False,extra)
        for label,mutate in [
            ('wrong-capture-kind',lambda c:c.update(capture_kind='myds_decode_qkv')),
            ('wrong-boundary',lambda c:c['cases'][0].update(boundary='q_rope')),
            ('trace-rejected',lambda c:c['cases'][0].update(trace_on=True)),
            ('missing-logits',lambda c:c['cases'][0].update(need_logits=False)),
            ('nonordinary',lambda c:c['cases'][0].update(ordinary_decode_allowed=False)),
            ('wrong-observer-ordinal',lambda c:c['cases'][0].update(call_ordinal=2)),
            ('wrong-rf-flag',lambda c:c['cases'][0].update(attention_rope_consumed=2)),
            ('wrong-weight-type',lambda c:c['cases'][0].update(weight_type=13)),
            ('wrong-input-bytes',lambda c:c['cases'][0]['input_f32'].update(bytes=128)),
            ('wrong-input-sha',lambda c:c['cases'][0]['input_f32'].update(sha256='0'*64)),
            ('wrong-prefix',lambda c:c.update(model_header_sha256='0'*64)),
            ('wrong-model-size',lambda c:c.update(model_bytes=c['model_bytes']+32)),
            ('wrong-layer',lambda c:c['cases'][0].update(layer=1)),
            ('duplicate-input',lambda c:c['cases'].append(copy.deepcopy(c['cases'][0]))),
        ]:
            bad=copy.deepcopy(capture);mutate(bad);run(label,bad,False)
        for label,bits in [('nonfinite',0x7f800000),('nonbf16',0x3f800001)]:
            data=struct.pack('<I',bits)+x[4:];(root/(label+'.f32')).write_bytes(data)
            bad=copy.deepcopy(capture);bad['cases'][0]['input_f32']=dict(path=label+'.f32',bytes=len(data),sha256=digest(data))
            run(label,bad,False)

        # Construct a synthetic complete record solely to test the host gate
        # verifier. GPU replay still requires all 12 actual ordinary cases.
        capture_path.write_text(json.dumps(capture));gate_dir=root/'cpu-claimed-correctness';gate_dir.mkdir()
        gate=copy.deepcopy(record);gate.update(status='complete',phase='correctness',full_f32_bit_exact=True,
            output_a_checked=True,all_gpu_inputs_unchanged=True,timing_command_buffers=0,parity=[],input_integrity=[],files=[])
        payloads={}
        def add(name,data):
            payloads[name]=data;(gate_dir/name).write_bytes(data)
            gate['files'].append(dict(path=name,bytes=len(data),sha256=digest(data)))
        add('capture.json',(positive/'capture.json').read_bytes())
        add('args136.bin',(positive/'args136.bin').read_bytes())
        add('capture-case-000-input.f32',x);add('input-case-000.guards',b'\xa5'*128)
        weights=b'\0'*weight_bytes;add('weight-layer-0-output-a.q4k',weights);add('weight-layer-0-output-a.guards',b'\xa5'*128)
        for provider in range(3):
            add(f'parity-case-000-provider-{provider}-output.f32',b'\0'*32768)
            add(f'parity-case-000-provider-{provider}-output.guards',b'\xa5'*128)
            gate['parity'].append(dict(kind='parity',phase='correctness',mode='concurrent',case=0,layer=0,position=0,
                provider=provider,output_bytes=32768,mismatches=0,full_f32_bit_exact=True,finite_guards_written=True,
                bf16_postround=True,all_gpu_inputs_unchanged=True,completed_cb=True))
            gate['input_integrity'].append(dict(case=0,provider=provider,all_gpu_inputs_unchanged=True,
                input_sha256=digest(x),expected_input_sha256=digest(x),weight_sha256=digest(weights),
                expected_weight_sha256=digest(weights),input_guards_intact=True,weight_guards_intact=True))
        gate_path=gate_dir/'artifact-manifest.json'
        def gate_run(label,value,expected):
            gate_path.write_text(json.dumps(value))
            return run(label,capture,expected,('--correctness-manifest',gate_path,
                '--correctness-manifest-sha256',digest(gate_path.read_bytes())))
        gate_run('valid-cpu-gate-parser-only',gate,True)
        for label,mutate in [
            ('wrong-gate-phase',lambda g:g.update(phase='performance')),
            ('wrong-gate-binary',lambda g:g.update(binary_sha256='0'*64)),
            ('wrong-gate-library',lambda g:g.update(probe_metallib_sha256='0'*64)),
            ('missing-provider',lambda g:g['parity'].pop()),
            ('duplicate-provider',lambda g:g['parity'][2].update(provider=1)),
            ('mismatch-claim',lambda g:g['parity'][1].update(mismatches=1)),
            ('duplicate-input-proof',lambda g:g['input_integrity'][2].update(provider=1)),
            ('unbound-input-proof',lambda g:g['input_integrity'][1].update(input_sha256='0'*64,expected_input_sha256='0'*64)),
            ('missing-full-output',lambda g:g.update(files=[r for r in g['files'] if r['path']!='parity-case-000-provider-2-output.f32'])),
        ]:
            bad=copy.deepcopy(gate);mutate(bad);gate_run(label,bad,False)
        for label,name,changed in [
            ('updated-sha-output-mismatch','parity-case-000-provider-1-output.f32',struct.pack('<I',0x3f800000)+b'\0'*(32768-4)),
            ('updated-sha-output-guard','parity-case-000-provider-1-output.guards',b'\0'+b'\xa5'*127),
            ('updated-sha-input-guard','input-case-000.guards',b'\0'+b'\xa5'*127),
            ('updated-sha-weight-guard','weight-layer-0-output-a.guards',b'\0'+b'\xa5'*127),
            ('updated-sha-args','args136.bin',b'\0'+payloads['args136.bin'][1:]),
        ]:
            bad=copy.deepcopy(gate);(gate_dir/name).write_bytes(changed)
            for r in bad['files']:
                if r['path']==name:r.update(bytes=len(changed),sha256=digest(changed))
            gate_run(label,bad,False);(gate_dir/name).write_bytes(payloads[name])
        run('mutually-exclusive-phases',capture,False,('--correctness-only','--correctness-manifest',gate_path,
            '--correctness-manifest-sha256',digest(gate_path.read_bytes())))
        with model.open('r+b') as stream:stream.truncate(model.stat().st_size-32)
        bad=copy.deepcopy(capture);bad['model_bytes']-=32;run('tensor-eof',bad,False)


if __name__=='__main__':main()
