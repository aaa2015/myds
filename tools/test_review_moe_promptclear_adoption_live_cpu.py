#!/usr/bin/env python3
"""Small CPU-only negative controls for the post-adoption audit byte readers."""
import argparse,hashlib,importlib.util,io,json,struct,sys,tarfile,tempfile
from pathlib import Path
sys.dont_write_bytecode=True

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();source=Path(__file__).with_name('review_moe_promptclear_adoption_live.py');spec=importlib.util.spec_from_file_location('live_cpu',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);passed=[]
    def good(name,fn):fn();passed.append(name)
    def bad(name,fn):
        try:fn()
        except (ValueError,KeyError):passed.append(name)
        else:raise AssertionError('Invalid bytes accepted: '+name)
    with tempfile.TemporaryDirectory(prefix='post-adoption-readonly-CPU-') as t:
        root=Path(t);f=root/'original';f.write_bytes(b'CPU original');ref=m.desc(f)
        head=struct.pack('i',1)+b'/CPU/myds\0\0\0'+b'myds\0';env=b'A=1\0B=2\0\0'
        expected={b'A'.hex():b'1'.hex(),b'B'.hex():b'2'.hex()};digest=hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest()
        good('KERN_environment_stops_at_first_empty_before_trailing_duplicate_storage',lambda:m.need(m.environment_digest_from_blob(head+env+b'A=not-env\0BROKEN-STORAGE\0')==dict(count=2,sha256=digest,non_myds_names_or_values_recorded=False),'Ignore trailing storage after environment terminator'))
        bad('reject_duplicate_environment_before_terminator',lambda:m.environment_digest_from_blob(head+b'A=1\0A=2\0\0'))
        bad('reject_missing_environment_terminator',lambda:m.environment_digest_from_blob(head+b'A=1'))
        good('strict_reference_positive',lambda:m.reference(ref));bad('reject_boolean_extent',lambda:m.reference(dict(ref,bytes=True)));bad('reject_changed_sha',lambda:m.reference(dict(ref,sha256='0'*64)));bad('reject_extra_reference_field',lambda:m.reference(dict(ref,unexpected=1)))
        link=root/'link';link.symlink_to(f);bad('reject_reference_symlink',lambda:m.reference(dict(ref,path=str(link))))
        j=root/'duplicate.json';j.write_text('{"k":1,"k":2}');bad('reject_duplicate_json_key',lambda:m.load(j));j.write_text('{"k":NaN}');bad('reject_nonfinite_json',lambda:m.load(j))
        def fixture(name,variant):
            release=root/name;d=release/'executed';d.mkdir(parents=True);rows=[];archives={};manifest={'executed':{}}
            for kind,key,logical,content in [('source.tar.gz','source_archive','stages/0/src/a.c',b'CPU source'),('artifacts.tar.gz','artifacts_archive','stages/0/bin/a',b'CPU fake binary')]:
                path=d/kind;original='/CPU/'+logical;expected=hashlib.sha256(content).hexdigest();row=dict(original_path=original,tar_path=logical,archive=kind,bytes=len(content),sha256=expected);rows.append(row)
                with tarfile.open(path,'w:gz') as tar:
                    n='../escape'if variant=='escape'and key=='source_archive'else logical;info=tarfile.TarInfo(n);info.size=len(content)
                    if variant=='symlink'and key=='source_archive':info.type=tarfile.SYMTYPE;info.linkname='/CPU/target';info.size=0;tar.addfile(info)
                    else:tar.addfile(info,io.BytesIO(b'BAD source'if variant=='payload'and key=='source_archive'else content))
                archives[kind]=dict(bytes=path.stat().st_size,sha256=m.sha(path),members=1);manifest['executed'][key]=dict(sha256=m.sha(path))
            inp={r['original_path']:r['sha256'] for r in rows};m.write_new(d/'stage-inputs.json',inp);mapping=dict(status='complete',stage_inputs_sha256=m.sha(d/'stage-inputs.json'),files=rows,archives=archives);m.write_new(d/'archive-bindings.json',mapping);manifest['executed']['input_map']=dict(sha256=m.sha(d/'stage-inputs.json'));return release,manifest
        r,manifest=fixture('positive','normal');good('complete_archived_source_and_bin_positive',lambda:m.need(m.archive_sources(r,manifest)[1]['original_frozen_inputs']==2,'coverage'))
        for name in ('escape','symlink','payload'):
            r,manifest=fixture(name,name);bad('reject_rehashed_archive_'+name,lambda r=r,manifest=manifest:m.archive_sources(r,manifest))
    result=dict(schema_version=1,all_controls_pass=True,scope='temporary_CPU_byte_readers_only',control_count=len(passed),controls=passed,tool_sha256=m.sha(source),test_sha256=m.sha(__file__),GPU_or_model_or_network_or_service_executed=False)
    m.write_new(a.out,result);print(json.dumps(result,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
