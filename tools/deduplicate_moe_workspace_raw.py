#!/usr/bin/env python3
"""CPU-only SHA plus full-byte checked atomic hardlink dedup of immutable raw."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import uuid


def need(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def equal(a,b):
    with a.open('rb') as left,b.open('rb') as right:
        while True:
            x,y=left.read(1048576),right.read(1048576)
            if x!=y:return False
            if not x:return True


def key(st):return st.st_dev,st.st_ino
def identity(path):
    s=path.stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,blocks=s.st_blocks,
                nlink=s.st_nlink,mode=stat.S_IMODE(s.st_mode),uid=s.st_uid,gid=s.st_gid,mtime_ns=s.st_mtime_ns)


def deduplicate(target,references):
    target=target.resolve(strict=True);references=[p.resolve(strict=True) for p in references]
    need(target.is_dir() and references and all(p.is_dir() and not target.is_relative_to(p) and
         not p.is_relative_to(target) for p in references),'Separate immutable target/reference trees required')
    free_before=os.statvfs(target).f_bavail*os.statvfs(target).f_frsize
    candidates={};digests={};before={};expected={};records=[];comparisons=set()
    def files(root):
        for p in sorted(root.rglob('*')):
            need(not p.is_symlink(),'No symlink permitted in immutable raw trees')
            if p.is_file():yield p
    for root in references:
        for p in files(root):
            s=p.stat();k=key(s)
            if k not in digests:digests[k]=sha(p)
            candidates.setdefault((s.st_size,digests[k],stat.S_IMODE(s.st_mode),s.st_uid,s.st_gid),p)
    target_files=list(files(target))
    for p in target_files:
        s=p.stat();k=key(s);before.setdefault(k,dict(stat=identity(p),paths=[]))['paths'].append(str(p))
        if k not in digests:digests[k]=sha(p)
        expected[str(p)]=(s.st_size,digests[k])
        reference=candidates.get((s.st_size,digests[k],stat.S_IMODE(s.st_mode),s.st_uid,s.st_gid))
        if reference is None or key(reference.stat())==k:continue
        pair=(k,key(reference.stat()))
        if pair not in comparisons:
            need(equal(p,reference),'Same SHA/size failed independent complete byte comparison')
            comparisons.add(pair)
        previous=identity(p);ref=identity(reference)
        temporary=p.with_name('.'+p.name+'.dedup-'+uuid.uuid4().hex)
        try:
            os.link(reference,temporary)
            need(identity(temporary)['inode']==ref['inode'] and identity(p)==previous,
                 'Immutable input changed before atomic replacement')
            os.replace(temporary,p)
        finally:
            if temporary.exists():temporary.unlink()
        records.append(dict(path=str(p),reference=str(reference),bytes=s.st_size,sha256=digests[k],
                            original=previous,replacement=identity(p),full_memcmp_verified=True))
    final={};referenced=set()
    for p in target_files:
        s=p.stat();k=key(s);referenced.add(k)
        if k not in final:final[k]=sha(p)
        need((s.st_size,final[k])==expected[str(p)],'Final original path/bytes/SHA differs')
    released=[v for k,v in before.items() if k not in referenced and len(v['paths'])==v['stat']['nlink']]
    return dict(schema_version=1,evidence_valid=True,CPU_only=True,GPU_or_model_executed=False,
        source_content_changed=False,all_original_target_paths_and_full_SHA_retained=True,
        target_root=str(target),reference_roots=list(map(str,references)),target_files_verified=len(target_files),
        replaced_logical_files=len(records),unique_full_memcmp_pairs=len(comparisons),replacements=records,
        target_unique_inodes_before=len(before),target_unique_inodes_after=len(final),
        released_previous_inodes=len(released),released_inode_allocated_bytes=sum(v['stat']['blocks']*512 for v in released),
        disk_available_bytes_before=free_before,disk_available_bytes_after=os.statvfs(target).f_bavail*os.statvfs(target).f_frsize,
        limits='No evidence bytes or logical paths removed; only identical immutable payloads share existing inodes. Original remote/tar metadata remains archived.')


def self_test():
    with tempfile.TemporaryDirectory(prefix='moe-raw-hardlink-CPU-') as name:
        root=Path(name);ref=root/'ref';target=root/'target';ref.mkdir();target.mkdir()
        (ref/'a').write_bytes(b'immutable exact payload'*100)
        (target/'x').write_bytes((ref/'a').read_bytes());os.link(target/'x',target/'alias')
        (target/'different').write_bytes(b'immutable other payload'*100)
        result=deduplicate(target,[ref])
        need(result['replaced_logical_files']==2 and result['unique_full_memcmp_pairs']==1 and
             result['released_previous_inodes']==1 and key((ref/'a').stat())==key((target/'x').stat())==key((target/'alias').stat()),
             'Atomic SHA/full-memcmp hardlink positive failed')
        (target/'bad-symlink').symlink_to(ref/'a')
        try:deduplicate(target,[ref])
        except ValueError:pass
        else:raise AssertionError('Symlink negative accepted')
        return dict(schema_version=1,all_pass=True,CPU_only=True,GPU_or_model_executed=False,
                    controls=['full_bytes_identical_atomic_hardlink','alias_inodes_and_original_paths_preserved',
                              'different_payload_retained','one_full_memcmp_for_shared_physical_source','reject_symlink'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--target-root',type=Path)
    p.add_argument('--reference-root',type=Path,action='append',default=[]);p.add_argument('--report',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--apply',action='store_true');mode.add_argument('--self-test',action='store_true')
    a=p.parse_args();result=self_test() if a.self_test else deduplicate(a.target_root,a.reference_root)
    with a.report.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='replacements'}))


if __name__=='__main__':main()
