#!/usr/bin/env python3
"""CPU-only full-buffer review of direct pack fixture, never inference."""
import argparse,hashlib,json,math,pathlib,posixpath,struct
P=pathlib.Path
p=argparse.ArgumentParser();p.add_argument('--raw',type=P,required=True);p.add_argument('--stdout',type=P,required=True);p.add_argument('--source-identity',type=P,required=True);p.add_argument('--golden-metallib',type=P,required=True);p.add_argument('--candidate-metallib',type=P,required=True);p.add_argument('--out',type=P,required=True);a=p.parse_args()
def need(x,m):
 if not x:raise ValueError(m)
def parse(b):
 def pairs(xs):
  d={}
  for k,v in xs:need(k not in d,'duplicate JSON '+k);d[k]=v
  return d
 def f(x):
  v=float(x);need(math.isfinite(v),'nonfinite JSON');return v
 def reject(x):raise ValueError('nonfinite JSON '+x)
 return json.loads(b,object_pairs_hook=pairs,parse_float=f,parse_constant=reject)
def sha(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):need(p.is_file()and not p.is_symlink(),'ordinary file required '+str(p));return sha(p.read_bytes())
def fnv(b):
 h=14695981039346656037
 for x in b:h=((h^x)*1099511628211)&0xffffffffffffffff
 return '%016x'%h
def canon(p):
 need(type(p)is str and p.startswith('/'),'absolute path');p=posixpath.normpath(p);return '/private'+p if p=='/tmp' or p.startswith('/tmp/')else p
need(not a.out.exists(),'never overwrite prior evidence');need(a.raw.is_dir()and not a.raw.is_symlink(),'ordinary raw directory')
m=parse((a.raw/'artifact-manifest.json').read_bytes());source=parse(a.source_identity.read_bytes());lines=[x for x in a.stdout.read_bytes().splitlines()if x.strip()];need(all(x.strip().startswith(b'{')for x in lines),'non-JSON original stdout');records=[parse(x)for x in lines]
need(m['schema_version']==1 and m['status']=='passed'and m['fixture_not_fullmodel']is True,'complete fixture manifest')
identity=[r for r in records if r['kind']=='library_identity'];result=[r for r in records if r['kind']=='result'];neg=[r for r in records if r['kind']=='predispatch_rejections'];parity=[r for r in records if r['kind']=='pack_parity']
need(len(identity)==len(result)==len(neg)==1 and len(parity)==96 and len(m['cases'])==96,'complete unique record inventory')
need({r['kind']for r in records}=={'library_identity','result','predispatch_rejections','pack_parity'},'unknown record')
need(result[0]['pass']is True and result[0]['cases']==96 and result[0]['fixture_not_fullmodel']is True,'complete result')
golden=file_sha(a.golden_metallib);candidate=file_sha(a.candidate_metallib);sd=file_sha(a.source_identity);i=identity[0]
need(golden=='e17018b547d24e558c64c61b3493d343ac3bd595397e207bb1a763d9ad544d57','immutable adopted NBHC+RF production golden')
need(m['golden_metallib']['sha256']==i['golden_sha256']==golden and m['candidate_metallib']['sha256']==i['candidate_sha256']==candidate and m['source_identity']['sha256']==i['source_identity_sha256']==sd,'library/source wholebyte SHA binding')
need(canon(m['golden_metallib']['path'])==canon(i['golden_path'])and canon(m['candidate_metallib']['path'])==canon(i['candidate_path'])==canon(i['loaded_runtime_path']),'actual loaded library binding')
need(type(m['args_bytes'])is int and type(m['dynamic_bytes'])is int and all(type(x)is int for x in m['threads'])and m['args_bytes']==16 and m['threads']==[128,1,1]and m['dynamic_bytes']==0 and m['candidate_symbol']==i['candidate_symbol']=='kernel_myds_dsv41_direct_pack_fuse'and m['golden_symbols']==['kernel_dsv41_copy_window_f32_to_f16','kernel_dsv41_get_rows_bf16_to_f16'],'PSO/geometry ABI')
need(source['compile_only']is True and source['GPU_run']is False and source['toolchain']['all_clang_isysroot_explicit']is True and len(source['input_sha256'])>20,'original compile-only source freeze')
blobs={}
for n,v in m['files'].items():
 need(type(n)is str and P(n).name==n and n not in('.','..')and type(v['bytes'])is int,'safe ordinary payload descriptor')
 f=a.raw/n;data=f.read_bytes();need(f.is_file()and not f.is_symlink()and len(data)==v['bytes']and sha(data)==v['sha256']and fnv(data)==v['fnv1a64'],'complete payload SHA/FNV '+n)
 need(v['view_offset']==16 and v['view_bytes']==len(data)-32 and data[:16]==data[-16:]==b'\xa5'*16,'full storage view/canary '+n);blobs[n]=data
need({f.name for f in a.raw.iterdir()}==set(blobs)|{'artifact-manifest.json'},'no omitted/unexpected raw files')
expected=[]
for concurrent in range(2):
 for nr in (1,33,128):
  for selected in (0,1,33,512):
   for start in (0,127):
    for invalid in range(2):expected.append((nr,selected,start,invalid,concurrent))
used=set();total_bytes=0
for k,(c,r,geometry)in enumerate(zip(m['cases'],parity,expected)):
 nr,selected,start,invalid,concurrent=geometry;source_rows=selected+19 if selected else 0
 need(all(type(c[t])is int for t in ('case','n_raw','selected_rows','raw_start','source_rows','invalid_ids','concurrent'))and type(r['case'])is int and type(c['case'])is int and c['case']==r['case']==k and tuple(c[t]for t in ('n_raw','selected_rows','raw_start','invalid_ids','concurrent'))==geometry and c['source_rows']==source_rows,'exact cases/order/geometry')
 for field in ('full_pack_bytes_equal','finite_active_half','poison_free_active','inputs_full_storage_same','canaries'):need(r[field]is True,'native full parity/guards '+field)
 need(all(type(r[t])is int for t in ('full_storage_bytes','reference_dispatches','candidate_dispatches','requests','encoded','failures'))and r['full_storage_bytes']==655392 and r['reference_dispatches']==1+bool(selected)and r['candidate_dispatches']==r['requests']==r['encoded']==1 and r['failures']==0,'actual candidate dispatch/counters')
 wf,cf,idf,of,pf=[c[t]for t in ('window','compressed','ids','golden','candidate')];need(len({wf,cf,idf,of,pf})==5 and not used.intersection((wf,cf,idf,of,pf)),'no reused logical casefiles');used.update((wf,cf,idf,of,pf))
 w,cbytes,ids,g,b=[blobs[n][16:-16]for n in (wf,cf,idf,of,pf)];need(len(w)==262144 and len(cbytes)==max(1,source_rows)*1024 and len(ids)==max(1,selected)*4 and len(g)==len(b)==655360,'exact complete input/output sizes')
 ew=bytearray()
 for j in range(65536):
  bits=0x3e800001+(j%1024)*67
  if j%7==0:bits^=0x80000000
  if j%19==0:bits=0x80000000
  ew+=struct.pack('<I',bits)
 ec=b''.join(struct.pack('<H',0x3e80+j%128)for j in range(max(1,source_rows)*512));ei=[]
 for j in range(max(1,selected)):
  ei.append((-1 if invalid else 0)if not selected else(-1 if j%5==0 else source_rows if j%5==1 else 2147483647 if j%5==2 else -2147483648 if j%5==3 else j%source_rows)if invalid else(j*53+7)%source_rows)
 need(w==ew and cbytes==ec and ids==struct.pack('<'+'i'*len(ei),*ei),'complete seeded input/guard preservation')
 oracle=bytearray(b'\xa5'*655360)
 for row in range(nr):
  rr=(start+row)%128
  for col in range(512):oracle[(row*512+col)*2:(row*512+col+1)*2]=struct.pack('<e',struct.unpack_from('<f',w,(rr*512+col)*4)[0])
 for row in range(selected):
  rr=ei[row]
  for col in range(512):
   x=struct.unpack('<f',struct.pack('<I',struct.unpack_from('<H',cbytes,(rr*512+col)*2)[0]<<16))[0]if 0<=rr<source_rows else 0.
   at=((nr+row)*512+col)*2;oracle[at:at+2]=struct.pack('<e',x)
 need(g==b==oracle,'CPU independent full row/conversion oracle')
 active=b[:(nr+selected)*1024];need(all((v[0]&0x7c00)!=0x7c00 and v[0]!=0xa5a5 for v in struct.iter_unpack('<H',active)),'finite/poisonfree complete active output')
 total_bytes+=len(g)+len(b)
n=neg[0];need(type(n['rejects'])is int and n['rejects']==12 and n['zero_dispatch_delta']is True and n['full_output_storage_same']is True and n['missing_selected_symbol_rejected']is True,'predispatch/missingPSO rejection gate');nf=n['poison_storage_file'];need(nf in blobs and blobs[nf]==b'\xa5'*655392,'complete rejected storage poison');used.add(nf);need(used==set(blobs),'complete referenced payload inventory')
r={'schema_version':1,'cpu_only':True,'pass':True,'reviewer_sha256':file_sha(P(__file__)),'manifest_sha256':file_sha(a.raw/'artifact-manifest.json'),'original_stdout_sha256':file_sha(a.stdout),'source_identity_sha256':sd,'golden_metallib_sha256':golden,'candidate_metallib_sha256':candidate,'cases':96,'full_payload_files':len(blobs),'full_output_bytes_compared':total_bytes,'all_complete_pack_storage_exact_golden_and_CPU_conversion_oracle':True,'all_seeded_inputs_fullbytes_recomputed':True,'canaries_and_unused_poison_exact':True,'active_output_finite_and_poison_free':True,'native_predispatch_rejections':12,'missing_selected_PSO_gate_pass':True,'scope':['Seeded finite conversion fixtures and original half pack kernels; no complete model/attention/reduction state proof.','No component timings are measured by this fixture; production samebuild native A-B-A performance and process/runtime authority are separate gates.','Whole local library files are independently hashed. Candidate source/build matching deployment must also be bound by the outer controller.']}
a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
