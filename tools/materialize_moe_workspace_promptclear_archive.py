#!/usr/bin/env python3
"""Lossless CPU evidence materialization with immutable existing payload reuse.

Existing files are read and optionally hardlinked. Their bytes, mode and mtime
are never changed. Link creation necessarily changes inode link count/ctime.
Original per-logical-path metadata stays in raw-files.json even when an existing
inode with a different mtime supplies identical bytes and the required mode.
"""
import argparse,hashlib,importlib.util,json,os,shutil,stat,tarfile,tempfile
from pathlib import Path,PurePosixPath

def need(v,m):
    if not v:raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def verifier():
    spec=importlib.util.spec_from_file_location('promptclear_lossless_verifier',Path(__file__).with_name('archive_attention_multihead_evidence.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def destination(root,name):
    p=PurePosixPath(name);need(name and not p.is_absolute()and str(p)==name and all(x not in ('','.','..')for x in p.parts),'Safe original logical archive path required')
    q=root.joinpath(*p.parts);need(q.is_relative_to(root),'Logical path escaped new output');return q

def mode(record):
    value=record['source_stat']['mode'];need(type(value)is int and 0<=value<=0o7777,'Original regular-file mode required');return value

def materialize(archive_dir,output,reuse_roots=()):
    archive_dir=Path(archive_dir).resolve(strict=True);output=Path(output).resolve()
    need(not output.exists()and not output.is_relative_to(archive_dir),'Separate new output directory required')
    manifest=json.loads((archive_dir/'raw-files.json').read_text());records=manifest['files']
    need(records and all(r['storage']in ('tar_regular','tar_hardlink')for r in records),'All logical raw files must have physical lossless members')
    by_path={r['path']:r for r in records};need(len(by_path)==len(records),'Unique original logical paths')
    verified=verifier().verify_archive(archive_dir/'raw.tar.gz',manifest)
    reuse={};old={};inode_sha={}
    for source in reuse_roots:
        source=Path(source).resolve(strict=True);need(source.is_dir()and not output.is_relative_to(source)and not source.is_relative_to(output),'Separate existing reuse tree')
        for p in sorted(source.rglob('*')):
            need(not p.is_symlink(),'Reuse tree may not contain symlinks')
            if not p.is_file():continue
            s=p.stat();key=(s.st_dev,s.st_ino)
            if key not in inode_sha:inode_sha[key]=sha(p)
            digest=inode_sha[key];m=stat.S_IMODE(s.st_mode)
            old[key]=dict(path=p,bytes=s.st_size,sha256=digest,mode=m,mtime_ns=s.st_mtime_ns)
            reuse.setdefault((s.st_size,digest,m),p)
    output.mkdir(parents=True,exist_ok=False);emitted=set();content={};reused=0;mode_copies=0
    def private_mode(path,wanted):
        need(path.stat().st_nlink==1,'Mode can only be applied to a new private output inode');os.chmod(path,wanted)
    with tarfile.open(archive_dir/'raw.tar.gz','r|gz')as tar:
        for member in tar:
            name=member.name;need(name in by_path and name not in emitted,'Unexpected/repeated logical member');r=by_path[name];wanted=mode(r)
            need(stat.S_IMODE(member.mode)==wanted,'Archived member mode differs from recorded original')
            target=destination(output,name);target.parent.mkdir(parents=True,exist_ok=True);key=(r['bytes'],r['sha256'],wanted)
            prior=content.get(key)or reuse.get(key)
            if member.islnk():
                need(r['storage']=='tar_hardlink'and member.linkname==r['linkname']and member.linkname in emitted,'Original backward hardlink identity')
                source=destination(output,member.linkname)
                if prior is not None:os.link(prior,target);reused+=int((prior.stat().st_dev,prior.stat().st_ino)in old)
                elif stat.S_IMODE(source.stat().st_mode)==wanted:os.link(source,target)
                else:
                    with source.open('rb')as a,target.open('xb')as b:shutil.copyfileobj(a,b,1048576)
                    private_mode(target,wanted);mode_copies+=1
            else:
                need(member.isfile()and r['storage']=='tar_regular'and member.size==r['bytes'],'Original regular payload identity')
                stream=tar.extractfile(member);need(stream is not None,'Readable original member');h=hashlib.sha256();count=0
                if prior is not None:
                    os.link(prior,target);reused+=int((prior.stat().st_dev,prior.stat().st_ino)in old);sink=None
                else:sink=target.open('xb')
                try:
                    for block in iter(lambda:stream.read(1048576),b''):
                        h.update(block);count+=len(block)
                        if sink is not None:sink.write(block)
                finally:
                    if sink is not None:sink.close()
                need(count==r['bytes']and h.hexdigest()==r['sha256'],'Every original physical payload bytes/SHA')
                if prior is None:private_mode(target,wanted)
            content.setdefault(key,target);emitted.add(name)
    need(emitted==by_path.keys(),'Complete logical output coverage')
    inodes={};mappings=[];new_allocated=referenced_allocated=0
    for r in records:
        p=destination(output,r['path']);need(p.is_file()and not p.is_symlink(),'Original regular output');s=p.stat();key=(s.st_dev,s.st_ino)
        if key not in inodes:
            inodes[key]=sha(p);referenced_allocated+=s.st_blocks*512
            if key not in old:new_allocated+=s.st_blocks*512
        need(s.st_size==r['bytes']and inodes[key]==r['sha256']and stat.S_IMODE(s.st_mode)==mode(r),'Final original logical bytes/SHA/mode')
        mappings.append(dict(path=r['path'],source_path=r['source_path'],bytes=s.st_size,sha256=r['sha256'],mode=mode(r),device=s.st_dev,inode=s.st_ino,existing_inode_reused=key in old,original_mtime_ns=r['source_stat']['mtime_ns'],materialized_inode_mtime_ns=s.st_mtime_ns))
    for key,v in old.items():
        s=v['path'].stat();need((s.st_dev,s.st_ino)==key and s.st_size==v['bytes']and stat.S_IMODE(s.st_mode)==v['mode']and s.st_mtime_ns==v['mtime_ns']and sha(v['path'])==v['sha256'],'Existing reuse bytes/mode/mtime must remain unchanged')
    return dict(schema_version=1,evidence_valid=True,scope='CPU_lossless_promptclear_evidence_materialization',gpu_or_model_executed=False,source_data_modified=False,existing_payload_bytes_modes_mtimes_unchanged=True,existing_inode_chmod_or_utime_called=False,archive_sha256=sha(archive_dir/'raw.tar.gz'),raw_files_sha256=sha(archive_dir/'raw-files.json'),archive_verification=verified,original_logical_files_verified=len(records),logical_bytes=sum(r['bytes']for r in records),local_unique_inodes=len(inodes),new_materialized_allocated_bytes=new_allocated,materialized_total_referenced_allocated_bytes=referenced_allocated,existing_payload_hardlinks_reused=reused,mode_mismatch_private_copies=mode_copies,original_to_local_file_identity=mappings,limits=['Hardlink creation changes link count and ctime; bytes/mode/mtime of existing sources are not changed.','Original per-logical-path metadata remains in the verified manifest. Shared materialized inode mtime may differ from that logical original; no timestamp equality is claimed.','Evidence byte verification/materialization does not execute the model or prove correctness/performance.'])

def self_test():
    checks=[];existing=verifier()
    with tempfile.TemporaryDirectory(prefix='promptclear-no-touch-CPU-')as t:
        root=Path(t);raw=root/'raw';reuse=root/'reuse';raw.mkdir();reuse.mkdir()
        for n,m in (('a',0o644),('b',0o600),('c',0o644)):
            p=raw/n;p.write_bytes(b'CPU-only identical payload');os.chmod(p,m)
        prior=reuse/'old';prior.write_bytes(b'CPU-only identical payload');os.chmod(prior,0o644);os.utime(prior,ns=(123000000000,123000000000));before=(prior.stat().st_mode,prior.stat().st_mtime_ns,sha(prior))
        existing.create(existing.parse_windows(['fixture='+str(raw)]),root/'archive',True)
        value=materialize(root/'archive',root/'output',[reuse]);need(value['evidence_valid']and value['original_logical_files_verified']==3,'All original bytes verified');checks.append('all_original_bytes_SHA_modes_verified')
        need((prior.stat().st_mode,prior.stat().st_mtime_ns,sha(prior))==before,'No old metadata/payload mutation');checks.append('existing_inode_bytes_mode_mtime_unchanged')
        p=root/'output/raw/fixture';need((p/'a').stat().st_ino==prior.stat().st_ino==(p/'c').stat().st_ino,'Compatible old inode reused');checks.append('compatible_mode_existing_hardlink_reused')
        need(stat.S_IMODE((p/'b').stat().st_mode)==0o600 and(p/'b').stat().st_ino!=prior.stat().st_ino,'Mode mismatch gets own inode');checks.append('different_mode_copied_without_chmod_old_inode')
        for name in ('../escape','/absolute','raw/../escape'):
            try:destination(root/'output',name)
            except ValueError:checks.append('reject_unsafe_'+name)
            else:raise AssertionError('Unsafe path accepted')
        r=root/'archive/raw-files.json';original=r.read_bytes();v=json.loads(original);v['files'][0]['sha256']='0'*64;r.write_text(json.dumps(v))
        try:materialize(root/'archive',root/'bad-output',[reuse])
        except ValueError:checks.append('reject_changed_manifest_SHA')
        else:raise AssertionError('Changed archive accepted')
        r.write_bytes(original);(reuse/'link').symlink_to(prior)
        try:materialize(root/'archive',root/'symlink-output',[reuse])
        except ValueError:checks.append('reject_reuse_symlink')
        else:raise AssertionError('Symlink reuse accepted')
    return dict(schema_version=1,scope='bounded_CPU_no_touch_materializer_controls',gpu_or_model_executed=False,all_controls_pass=True,checks=checks)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive-dir',type=Path);p.add_argument('--out-dir',type=Path);p.add_argument('--reuse-root',type=Path,action='append',default=[]);p.add_argument('--report',type=Path,required=True);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    try:
        if a.self_test:need(a.archive_dir is None and a.out_dir is None,'Separate CPU self-test');value=self_test()
        else:need(a.archive_dir is not None and a.out_dir is not None,'Archive and new output required');value=materialize(a.archive_dir,a.out_dir,a.reuse_root)
    except Exception as e:value=dict(schema_version=1,evidence_valid=False,gpu_or_model_executed=False,error=repr(e))
    with a.report.open('x')as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in value.items()if k not in ('original_to_local_file_identity','archive_verification')},indent=2));return 0 if value.get('evidence_valid',value.get('all_controls_pass',False))else 1
if __name__=='__main__':raise SystemExit(main())
