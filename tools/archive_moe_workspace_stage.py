#!/usr/bin/env python3
"""Archive exact executed stage inputs without modifying a stage or window.

Every archived byte must match the original window's stage-inputs.json. Source,
binary and external-input paths are retained in a reversible mapping. Creation
requires a new directory; verification never extracts or executes any payload.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def load(path):return json.loads(path.read_text())
def write_new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')


def verify(directory):
    manifest=load(directory/'archive-bindings.json');original=load(directory/'stage-inputs.json')
    need(manifest['schema_version']==1 and manifest['status']=='complete' and
         sha(directory/'stage-inputs.json')==manifest['stage_inputs_sha256'],'Original input-map binding differs')
    entries=manifest['files'];need(len(entries)==len(original) and {r['original_path']:r['sha256'] for r in entries}==original,
                                'Exact original path/SHA input coverage differs')
    targets={};verified=0
    for entry in entries:
        key=(entry['archive'],entry['tar_path']);need(key not in targets,'Duplicate archive logical member');targets[key]=entry
    reports=[]
    for archive in ('source.tar.gz','artifacts.tar.gz'):
        path=directory/archive;record=manifest['archives'][archive]
        need(sha(path)==record['sha256'] and path.stat().st_size==record['bytes'],'Archive physical bytes/SHA differs')
        names=set()
        with tarfile.open(path,'r:*') as tar:
            for member in tar:
                need(member.isfile() and not member.issym() and not member.islnk() and
                     not Path(member.name).is_absolute() and '..' not in Path(member.name).parts,'Only bounded regular members allowed')
                need(member.name not in names,'Duplicate tar member');names.add(member.name)
                expected=targets.get((archive,member.name));need(expected is not None,'Unexpected tar member')
                stream=tar.extractfile(member);need(stream is not None,'Unreadable archived member');h=hashlib.sha256();count=0
                for block in iter(lambda:stream.read(1048576),b''):h.update(block);count+=len(block)
                need(count==member.size==expected['bytes'] and h.hexdigest()==expected['sha256'],'Original member bytes/SHA mismatch')
                verified+=1
        expected={name for kind,name in targets if kind==archive}
        need(names==expected and len(names)==record['members'],'Missing archived member')
        reports.append(dict(path=str(path),sha256=record['sha256'],bytes=record['bytes'],members_verified=len(names)))
    need(verified==len(original),'Incomplete original member verification')
    return dict(schema_version=1,archive_verification_passed=True,stage_inputs_sha256=manifest['stage_inputs_sha256'],
                original_logical_inputs_verified=verified,archives=reports,source_or_window_modified=False,
                payload_extracted=False,gpu_or_model_executed=False)


def create(window,directory):
    window=window.resolve(strict=True);source=window/'stage-inputs.json';bindings=load(source)
    need(type(bindings) is dict and bindings and all(Path(p).is_absolute() and type(h) is str and len(h)==64 for p,h in bindings.items()),
         'Absolute original frozen input map required')
    summary=load(window/'summary.json')
    need(summary['restoration']['restored'] is True and summary['status']!='prepared','Completed/restored actual window required')
    stage_roots=sorted({Path(p).parent for p in bindings if Path(p).name=='Makefile'},key=str)
    need(stage_roots,'Original stage root coverage missing')
    entries=[]
    for ordinal,(name,digest) in enumerate(sorted(bindings.items())):
        p=Path(name);need(p.is_file() and sha(p)==digest,'Frozen original input changed or unavailable: '+name)
        matches=[stage for stage in stage_roots if stage in p.parents]
        need(len(matches)<=1,'Overlapping stage roots')
        if matches:
            stage=matches[0];relative=p.relative_to(stage);kind='artifacts.tar.gz' if relative.parts[0]=='bin' else 'source.tar.gz'
            target=f'stages/stage-{stage_roots.index(stage):03d}/'+relative.as_posix()
        else:kind='source.tar.gz';target=f'external/input-{ordinal:05d}/'+p.name
        st=p.stat();entries.append(dict(original_path=name,resolved_path=str(p.resolve(strict=True)),archive=kind,tar_path=target,
            bytes=st.st_size,sha256=digest,device=st.st_dev,inode=st.st_ino,mtime_ns=st.st_mtime_ns))
    directory.mkdir(parents=False,exist_ok=False)
    shutil.copyfile(source,directory/'stage-inputs.json')
    records={}
    for kind in ('source.tar.gz','artifacts.tar.gz'):
        members=[r for r in entries if r['archive']==kind];path=directory/kind
        with tarfile.open(path,'x:gz',format=tarfile.PAX_FORMAT) as tar:
            for entry in members:
                p=Path(entry['original_path']);st=p.stat();info=tarfile.TarInfo(entry['tar_path'])
                info.size=entry['bytes'];info.mode=st.st_mode&0o777;info.mtime=st.st_mtime
                with p.open('rb') as stream:tar.addfile(info,stream)
        records[kind]=dict(bytes=path.stat().st_size,sha256=sha(path),members=len(members))
    for name,digest in bindings.items():need(sha(Path(name))==digest,'Original input changed while archiving: '+name)
    need(sha(source)==sha(directory/'stage-inputs.json'),'Original input-map changed while archiving')
    write_new(directory/'archive-bindings.json',dict(schema_version=1,status='complete',original_window=str(window),
        original_stage_roots=[str(s) for s in stage_roots],stage_inputs_original_path=str(source),
        stage_inputs_sha256=sha(source),files=entries,archives=records,
        source_or_window_modified=False,model_backed_weight_payload_exported=False))
    result=verify(directory);result['all_original_input_paths_rehashed_after_archive']=True
    write_new(directory/'archive-validation.json',result);return result


def self_test():
    controls=[]
    with tempfile.TemporaryDirectory(prefix='moe-stage-archive-CPU-') as tmp:
        root=Path(tmp);window=root/'window';window.mkdir();paths=[]
        for name in ('candidate','gold'):
            stage=root/name
            for relative,content in [('Makefile',b'CPU fixture Makefile'),('src/a.c',b'CPU fixture source '+name.encode()),('bin/myds',b'CPU fake binary '+name.encode())]:
                p=stage/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content);paths.append(p)
        external=root/'plan.json';external.write_text('{"synthetic_CPU":true}');paths.append(external)
        write_new(window/'stage-inputs.json',{str(p):sha(p) for p in paths})
        write_new(window/'summary.json',dict(status='failed',restoration=dict(restored=True)))
        archive=root/'archive';report=create(window,archive);need(report['original_logical_inputs_verified']==7,'Synthetic coverage')
        controls.append('complete_two_stage_source_bin_external_mapping_positive')
        paths[0].write_bytes(b'changed after window freeze')
        try:create(window,root/'invalid-new-archive')
        except ValueError:controls.append('reject_changed_original_source_before_creation')
        else:raise AssertionError('Changed original accepted')
        need(verify(archive)['archive_verification_passed'],'Portable originals no longer needed');controls.append('portable_archive_verifies_without_unchanged_originals')
        m=load(archive/'archive-bindings.json');m['files'][0]['sha256']='0'*64
        (archive/'archive-bindings.json').write_text(json.dumps(m))
        try:verify(archive)
        except ValueError:controls.append('reject_changed_original_path_SHA_mapping')
        else:raise AssertionError('Tampered mapping accepted')
    return dict(schema_version=1,scope='synthetic_CPU_archive_controls',all_controls_pass=True,controls=controls,
                gpu_or_model_executed=False,production_evidence_reviewed=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--window',type=Path);m.add_argument('--verify-existing',type=Path);m.add_argument('--self-test',action='store_true')
    p.add_argument('--out',type=Path);p.add_argument('--report',type=Path);a=p.parse_args()
    try:
        if a.self_test:result=self_test()
        elif a.window:need(a.out is not None,'--window requires new --out directory');result=create(a.window,a.out)
        else:result=verify(a.verify_existing)
    except Exception as exc:result=dict(archive_verification_passed=False,error=repr(exc),source_or_window_modified=False,gpu_or_model_executed=False)
    if a.report:write_new(a.report,result)
    print(json.dumps(result,indent=2,sort_keys=True));return 0 if result.get('archive_verification_passed') or result.get('all_controls_pass') else 1


if __name__=='__main__':raise SystemExit(main())
