#!/usr/bin/env python3
"""Lossless CPU-only archive of explicitly named attention evidence windows.

  --window window-1=/absolute/window --window diagnostic-1=/absolute/diagnostic
  --out-dir NEW --verify

Each input file retains a raw/NAME/relative logical path, byte count and SHA256.
Only raw-capture/layer-N-kv-f16.bin may be virtual: its actual SHA must match
both its capture descriptor and an independently expanded dictionary/row-ID
stream. All other files remain tar members, with cross-window identical content
stored as tar hardlinks. No source data is modified or removed. --verify checks
every tar member and reconstructs every virtual stream without materializing it,
then rehashes the actual sources. --verify-existing DIR verifies copied archives
without requiring the original source paths. This tool never calls Metal/models.
"""
import argparse
import copy
import datetime
import hashlib
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import re
import stat
import struct
import tarfile
import tempfile

ROW_BYTES = 1024


def need(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(path):
    value=Path(path).stat()
    return dict(device=value.st_dev,inode=value.st_ino,bytes=value.st_size,
                mtime_ns=value.st_mtime_ns,mode=stat.S_IMODE(value.st_mode))


def logical_path(value):
    path=PurePosixPath(value)
    need(value and not path.is_absolute() and all(p not in ('','..','.') for p in path.parts),
         'Unsafe archive logical path: '+value)
    need(str(path)==value,'Noncanonical logical path: '+value)
    return value


def reconstruct_sha(dictionary, ids, count, row_bytes=ROW_BYTES):
    need(row_bytes==ROW_BYTES and type(count) is int and count>0,'Invalid KV reconstruction shape')
    dbytes,ibytes=Path(dictionary).stat().st_size,Path(ids).stat().st_size
    need(dbytes>0 and dbytes%row_bytes==0 and ibytes==count*4,'Invalid dictionary/row-ID byte lengths')
    digest=hashlib.sha256()
    with Path(dictionary).open('rb') as df,Path(ids).open('rb') as inf:
        with mmap.mmap(df.fileno(),0,access=mmap.ACCESS_READ) as dm,mmap.mmap(inf.fileno(),0,access=mmap.ACCESS_READ) as im:
            view=memoryview(dm)
            try:
                for offset in range(0,ibytes,4):
                    row=struct.unpack_from('<I',im,offset)[0]
                    need(row<dbytes//row_bytes,'KV row ID outside dictionary')
                    digest.update(view[row*row_bytes:(row+1)*row_bytes])
            finally:
                view.release()
    return digest.hexdigest()


def parse_windows(arguments):
    result=[]
    for value in arguments:
        name,separator,path=value.partition('=')
        if name.startswith('raw/'):
            name=name[4:]
        need(separator and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name),'Expected --window NAME=/absolute/path')
        source=Path(path)
        need(source.is_absolute() and source.is_dir(),'Input window must be an existing absolute directory')
        source=source.resolve(strict=True)
        need(all(name!=w['name'] and source!=w['source'] and
                 not source.is_relative_to(w['source']) and not w['source'].is_relative_to(source)
                 for w in result),'Duplicate/overlapping window mapping')
        result.append(dict(name=name,source=source,prefix='raw/'+name))
    need(result,'At least one explicit window required')
    return result


def scan(windows):
    records=[]; inode_hashes={}
    for window in windows:
        source=window['source']
        for path in sorted(source.rglob('*')):
            need(not path.is_symlink(),'Symlink evidence requires explicit handling: '+str(path))
            if path.is_dir():
                continue
            need(path.is_file(),'Unsupported evidence file type: '+str(path))
            before=fingerprint(path)
            key=(before['device'],before['inode'],before['bytes'],before['mtime_ns'])
            digest=inode_hashes.get(key)
            if digest is None:
                digest=inode_hashes[key]=sha(path)
            need(fingerprint(path)==before,'Evidence changed while scanning: '+str(path))
            records.append(dict(path=logical_path(window['prefix']+'/'+path.relative_to(source).as_posix()),
                                source_path=str(path),source_stat=before,bytes=before['bytes'],sha256=digest,
                                storage='tar_regular'))
    need(records,'Input windows contain no files')
    return records,len(inode_hashes)


def mark_virtual(records, windows):
    by_source={r['source_path']:r for r in records}
    marked=0
    for window in windows:
        candidates=[r for r in records if Path(r['source_path']).parent==window['source']/'raw-capture' and
                    re.fullmatch(r'layer-[0-9]+-kv-f16\.bin',Path(r['source_path']).name)]
        if not candidates:
            continue
        capture_path=window['source']/'packed-capture/capture.json'
        need(capture_path.is_file(),'Expanded KV has no retained packed capture: '+str(capture_path))
        capture=load(capture_path)
        cases=capture.get('cases')
        need(isinstance(cases,list),'Packed capture cases required')
        for record in candidates:
            layer=int(re.fullmatch(r'layer-([0-9]+)-kv-f16\.bin',Path(record['source_path']).name)[1])
            matching=[c for c in cases if c.get('layer')==layer]
            need(len(matching)==1,'Expanded KV must match exactly one packed case')
            case=matching[0]
            rows,keys=case.get('rows'),case.get('keys')
            need(type(rows) is int and type(keys) is int and rows>0 and keys>0,'Positive captured KV shape required')
            count=rows*keys
            need(record['bytes']==count*ROW_BYTES,'Original expanded KV byte count differs from capture')
            refs={}
            for field,label in (('kv_unique_f16','dictionary'),('kv_row_ids_u32','row_ids')):
                desc=case.get(field)
                need(isinstance(desc,dict) and isinstance(desc.get('path'),str),'Dictionary/ID descriptor required')
                path=Path(desc['path'])
                if not path.is_absolute():
                    path=capture_path.parent/path
                path=path.resolve(strict=True)
                need(str(path) in by_source,'Virtual KV dependency is outside mapped windows: '+str(path))
                physical=by_source[str(path)]
                need(physical['bytes']==desc.get('bytes') and physical['sha256']==desc.get('sha256'),
                     'Packed KV dependency descriptor SHA/bytes mismatch')
                refs[label]=dict(archive_path=physical['path'],bytes=physical['bytes'],sha256=physical['sha256'])
            dictionary=by_source[str((Path(case['kv_unique_f16']['path']) if Path(case['kv_unique_f16']['path']).is_absolute()
                else capture_path.parent/case['kv_unique_f16']['path']).resolve())]
            ids=by_source[str((Path(case['kv_row_ids_u32']['path']) if Path(case['kv_row_ids_u32']['path']).is_absolute()
                else capture_path.parent/case['kv_row_ids_u32']['path']).resolve())]
            reconstructed=reconstruct_sha(dictionary['source_path'],ids['source_path'],count)
            need(record['sha256']==reconstructed==case.get('kv_expanded_sha256'),
                 'Actual expanded KV, dictionary expansion and capture SHA do not all agree')
            record.update(storage='lossless_kv_dictionary',row_bytes=ROW_BYTES,row_ids_format='uint32le',
                          row_count=count,capture_rows=rows,capture_keys=keys,layer=layer,
                          dictionary=refs['dictionary'],row_ids=refs['row_ids'],
                          actual_source_sha_equals_reconstruction_and_capture=True)
            marked+=1
    regular={}
    for record in records:
        if record['storage']=='lossless_kv_dictionary':
            continue
        key=(record['bytes'],record['sha256'])
        if key in regular:
            record.update(storage='tar_hardlink',linkname=regular[key])
        else:
            regular[key]=record['path']
    return marked


def archive_files(path,records,level):
    with tarfile.open(path,'w:gz',compresslevel=level,format=tarfile.PAX_FORMAT) as archive:
        for record in records:
            need(fingerprint(record['source_path'])==record['source_stat'],'Evidence changed before tar write')
            if record['storage']=='lossless_kv_dictionary':
                continue
            info=tarfile.TarInfo(record['path'])
            info.mode=record['source_stat']['mode'];info.mtime=record['source_stat']['mtime_ns']//1000000000
            if record['storage']=='tar_hardlink':
                info.type=tarfile.LNKTYPE;info.linkname=record['linkname'];info.size=0
                archive.addfile(info)
            else:
                info.size=record['bytes']
                with Path(record['source_path']).open('rb') as stream:
                    archive.addfile(info,stream)
            need(fingerprint(record['source_path'])==record['source_stat'],'Evidence changed during tar write')


def verify_archive(archive_path,manifest):
    records=manifest['files'];by_path={r['path']:r for r in records}
    need(len(by_path)==len(records),'Duplicate logical manifest path')
    for record in records:
        logical_path(record['path'])
        need(type(record['bytes']) is int and record['bytes']>=0 and
             re.fullmatch(r'[0-9a-f]{64}',record['sha256']),'Invalid logical byte/SHA metadata')
    def canonical(path):
        seen=set()
        while True:
            need(path in by_path and path not in seen,'Invalid/cyclic archive dependency')
            seen.add(path);row=by_path[path]
            if row['storage']=='tar_regular':
                return path
            need(row['storage']=='tar_hardlink','Virtual dependency cannot itself be virtual')
            path=logical_path(row['linkname'])
    required={}
    virtual=[r for r in records if r['storage']=='lossless_kv_dictionary']
    for record in virtual:
        need(record['row_bytes']==ROW_BYTES and record['row_ids_format']=='uint32le' and
             record['row_count']*ROW_BYTES==record['bytes'],'Invalid virtual KV shape')
        for label in ('dictionary','row_ids'):
            ref=record[label];logical_path(ref['archive_path'])
            need(ref['archive_path'] in by_path,'Missing virtual dependency')
            physical=by_path[ref['archive_path']]
            need((ref['bytes'],ref['sha256'])==(physical['bytes'],physical['sha256']),
                 'Virtual dependency SHA/bytes mismatch')
            required[canonical(ref['archive_path'])]=None
    actual={};regular_bytes=0
    with tempfile.TemporaryDirectory(prefix='attention-archive-verify-') as temporary:
        with tarfile.open(archive_path,'r|gz') as archive:
            for member in archive:
                name=logical_path(member.name)
                need(name in by_path and name not in actual,'Unexpected/duplicate tar member')
                record=by_path[name]
                if member.islnk():
                    need(record['storage']=='tar_hardlink' and member.linkname==record['linkname'] and
                         member.linkname in actual and member.size==0,'Unexpected/forward tar hardlink')
                    digest,size=actual[member.linkname]
                else:
                    need(member.isfile() and record['storage']=='tar_regular' and member.size==record['bytes'],
                         'Unexpected tar entry type/length')
                    digest,size=hashlib.sha256(),0
                    cache=None
                    if name in required:
                        cache=Path(temporary)/str(len([x for x in required.values() if x is not None]))
                        required[name]=cache
                    sink=cache.open('xb') if cache else None
                    try:
                        stream=archive.extractfile(member)
                        for block in iter(lambda:stream.read(1048576),b''):
                            digest.update(block);size+=len(block)
                            if sink:sink.write(block)
                    finally:
                        if sink:sink.close()
                    digest=digest.hexdigest();regular_bytes+=size
                need((digest,size)==(record['sha256'],record['bytes']),'Tar logical file byte/SHA mismatch: '+name)
                actual[name]=(digest,size)
        need(set(actual)=={r['path'] for r in records if r['storage']!='lossless_kv_dictionary'},
             'Tar omitted required physical logical files')
        for record in virtual:
            dpath=required[canonical(record['dictionary']['archive_path'])]
            ipath=required[canonical(record['row_ids']['archive_path'])]
            need(dpath is not None and ipath is not None,'Missing cached archive reconstruction dependency')
            digest=reconstruct_sha(dpath,ipath,record['row_count'],record['row_bytes'])
            need(digest==record['sha256'],'Archived virtual KV reconstruction SHA mismatch: '+record['path'])
    return dict(verified=True,logical_files_verified=len(records),physical_tar_members_verified=len(actual),
                virtual_kv_streams_verified=len(virtual),logical_bytes=sum(r['bytes'] for r in records),
                virtual_reconstructed_bytes=sum(r['bytes'] for r in virtual),physical_regular_payload_bytes=regular_bytes,
                hardlink_members_verified=sum(r['storage']=='tar_hardlink' for r in records),
                archive_bytes=Path(archive_path).stat().st_size,archive_sha256=sha(archive_path),
                verification='Every physical logical member byte/SHA checked; every virtual KV re-expanded from archived dictionary/IDs and SHA checked without writing expanded bytes')


def verify_sources(records):
    cache={}
    for record in records:
        need(fingerprint(record['source_path'])==record['source_stat'],'Source metadata changed after archive')
        st=record['source_stat'];key=(st['device'],st['inode'],st['bytes'],st['mtime_ns'])
        digest=cache.get(key)
        if digest is None:digest=cache[key]=sha(record['source_path'])
        need(digest==record['sha256'],'Actual source SHA changed after archive')
        need(fingerprint(record['source_path'])==record['source_stat'],'Source changed during final actual SHA check')
    return len(cache)


def create(windows,out,verify=False,level=1):
    out=Path(out).resolve()
    need(not out.exists() and all(not out.is_relative_to(w['source']) for w in windows),
         'Archive output must be new and outside every source window')
    out.mkdir(parents=True)
    try:
        records,scanned=scan(windows);virtual=mark_virtual(records,windows)
        manifest=dict(schema_version=1,kind='attention_multihead_lossless_raw_archive',gpu_work_submitted=False,
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            windows=[dict(name=w['name'],source_root=str(w['source']),archive_prefix=w['prefix']) for w in windows],
            logical_files=len(records),logical_bytes=sum(r['bytes'] for r in records),virtual_kv_files=virtual,
            storage_note='Tar hardlinks preserve duplicate logical payloads; virtual KV exists only as exact archived dictionary+uint32le row IDs. Source files are retained unchanged.',files=records)
        write_json(out/'raw-files.json',manifest)
        archive_files(out/'raw.tar.gz',records,level)
        if verify:
            result=verify_archive(out/'raw.tar.gz',manifest)
            result['source_unique_inodes_rehashed_after_archive']=verify_sources(records)
            result['actual_source_bytes_sha_rechecked_after_archive']=True
        else:
            result=dict(verified=False,reason='--verify not requested',archive_sha256=sha(out/'raw.tar.gz'))
        result.update(schema_version=1,raw_files_sha256=sha(out/'raw-files.json'),source_unique_inodes_scanned=scanned,
                      source_data_modified=False,gpu_work_submitted=False)
        write_json(out/'archive-validation.json',result)
        return result
    except BaseException as exc:
        path=out/'archive-validation.json'
        if not path.exists():write_json(path,dict(schema_version=1,verified=False,error=repr(exc),source_data_modified=False))
        raise


def self_test():
    with tempfile.TemporaryDirectory(prefix='attention-archive-selftest-') as temporary:
        base=Path(temporary);w=base/'window';d=base/'diagnostic'
        (w/'raw-capture').mkdir(parents=True);packed=w/'packed-capture';packed.mkdir();d.mkdir()
        dictionary=packed/'unique.f16';ids=packed/'ids.u32le';expanded=w/'raw-capture/layer-2-kv-f16.bin'
        dictionary.write_bytes(b''.join(bytes([row])*ROW_BYTES for row in range(8)))
        values=[(i*7)%8 for i in range(1024)]
        ids.write_bytes(struct.pack('<1024I',*values))
        expanded.write_bytes(b''.join(bytes([row])*ROW_BYTES for row in values))
        descriptor=lambda p:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
        write_json(packed/'capture.json',dict(cases=[dict(layer=2,rows=2,keys=512,kv_unique_f16=descriptor(dictionary),
            kv_row_ids_u32=descriptor(ids),kv_expanded_sha256=sha(expanded))]))
        (w/'same.bin').write_bytes(b'identical content across separate windows\n')
        (d/'copy.bin').write_bytes((w/'same.bin').read_bytes());(d/'empty.bin').write_bytes(b'')
        windows=parse_windows(['window-1='+str(w),'diagnostic-1='+str(d)])
        result=create(windows,base/'archive',True)
        need(result['virtual_kv_streams_verified']==1 and result['hardlink_members_verified']>=1,'Positive fixture did not exercise virtual/hardlink storage')
        manifest=load(base/'archive/raw-files.json');rejected=[]
        def reject(name,call):
            try:call()
            except (ValueError,OSError,KeyError,tarfile.TarError):rejected.append(name)
            else:raise AssertionError('Negative control accepted: '+name)
        original=expanded.read_bytes();expanded.write_bytes(b'X'+original[1:])
        reject('actual_expanded_KV_tamper',lambda:mark_virtual(scan(windows)[0],windows));expanded.write_bytes(original)
        original_ids=ids.read_bytes();ids.write_bytes(struct.pack('<I',9999)+original_ids[4:])
        reject('source_row_ID_tamper',lambda:mark_virtual(scan(windows)[0],windows));ids.write_bytes(original_ids)
        bad=copy.deepcopy(manifest);next(r for r in bad['files'] if r['storage']=='lossless_kv_dictionary')['sha256']='0'*64
        reject('virtual_reconstruction_digest_tamper',lambda:verify_archive(base/'archive/raw.tar.gz',bad))
        bad=copy.deepcopy(manifest);next(r for r in bad['files'] if r['storage']=='lossless_kv_dictionary')['row_count']+=1
        reject('virtual_reconstruction_count_tamper',lambda:verify_archive(base/'archive/raw.tar.gz',bad))
        bad=copy.deepcopy(manifest);next(r for r in bad['files'] if r['storage']=='tar_hardlink')['linkname']='raw/missing'
        reject('hardlink_target_tamper',lambda:verify_archive(base/'archive/raw.tar.gz',bad))
        # Copied archives remain independently verifiable when all input paths
        # in the manifest have become unavailable; only archive members are used.
        for r in manifest['files']:r['source_path']='/unavailable/original'
        verify_archive(base/'archive/raw.tar.gz',manifest)
        return dict(self_test_passed=True,gpu_work_submitted=False,fixture_expanded_rows=1024,
                    fixture_expanded_bytes=1024*ROW_BYTES,positive_archive=result,rejected_negative_controls=rejected,
                    offline_verification_without_original_sources=True,scope='Synthetic CPU archive tests only; no model/correctness/performance claim')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window',action='append',default=[])
    parser.add_argument('--out-dir',type=Path);parser.add_argument('--verify',action='store_true')
    parser.add_argument('--gzip-level',type=int,choices=range(1,10),default=1)
    parser.add_argument('--verify-existing',type=Path);parser.add_argument('--report',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        need(not args.window and args.out_dir is None and args.verify_existing is None,'Self-test is a separate mode')
        result=self_test()
    elif args.verify_existing:
        need(not args.window and args.out_dir is None,'Verify-existing is a separate mode')
        manifest=load(args.verify_existing/'raw-files.json')
        result=verify_archive(args.verify_existing/'raw.tar.gz',manifest)
        result.update(schema_version=1,raw_files_sha256=sha(args.verify_existing/'raw-files.json'),
                      original_source_paths_required=False,gpu_work_submitted=False)
    else:
        need(args.out_dir is not None,'--out-dir NEW is required')
        result=create(parse_windows(args.window),args.out_dir,args.verify,args.gzip_level)
    if args.report:write_json(args.report,result)
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
