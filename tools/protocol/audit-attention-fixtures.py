#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);result=[];previous=None
paths=sorted(root.glob('attention-*.json'),key=lambda p:json.loads(p.read_text())['base_pos']+json.loads(p.read_text())['c_start'])
for meta_path in paths:
 meta=json.loads(meta_path.read_text());prefix=meta_path.with_suffix('');rows=meta['rows'];pos=meta['base_pos']+meta['c_start'];nc=meta['n_comp'];ratio=meta['ratio']
 assert meta['layer']==14 and meta['heads']==64 and meta['head_dim']==512 and meta['top_k']==512 and meta['carry_rows']==127
 assert meta['compressed_bf16'] and meta['use_gather'] and rows in (512,294) and ratio==2
 assert nc==(pos+rows)//ratio
 sizes={'q-f32':rows*64*512*4,'kv-f32':rows*512*4,'comp':nc*512*2,'ids-i32':rows*512*4,'sinks-f32':256,'carry-f16':127*512*2,'heads-f32':rows*64*512*4}
 item={'prefix':str(prefix),'metadata':meta,'files':{}}
 for name,n in sizes.items():
  p=Path(str(prefix)+'-'+name+'.bin');assert p.stat().st_size==n,(p,n,p.stat().st_size)
  h=hashlib.sha256()
  with p.open('rb') as f:
   while block:=f.read(1024*1024):h.update(block)
  item['files'][name]={'bytes':n,'sha256':h.hexdigest()}
 stats={}
 for name in ('q-f32','kv-f32','sinks-f32','heads-f32'):
  p=Path(str(prefix)+'-'+name+'.bin');x=np.memmap(p,mode='r',dtype='<f4')
  assert np.isfinite(x).all(),name
  stats[name]={'min':float(x.min()),'max':float(x.max()),'rms':float(np.sqrt(np.mean(x.astype(np.float64)**2))), 'zeros':int(np.count_nonzero(x==0)), 'bf16_representable':bool(np.all((x.view('<u4') & 65535)==0))}
 comp=np.memmap(str(prefix)+'-comp.bin',mode='r',dtype='<u2');assert np.all((comp&0x7f80)!=0x7f80)
 stats['comp']={'finite':True,'zeros':int(np.count_nonzero((comp&0x7fff)==0))}
 ids=np.memmap(str(prefix)+'-ids-i32.bin',mode='r',dtype='<i4',shape=(rows,512));live_total=0;ignored_negative=0;ignored_nonnegative=0;bad_live=[];unique=[]
 for r in range(rows):
  count=(pos+r+1)//ratio;live=min(512,count);sel=ids[r,:live];live_total+=live
  if not np.all((sel>=0)&(sel<count)&(sel<nc)):bad_live.append(r)
  if len(set(int(v) for v in sel))!=live:unique.append(r)
  ignored_negative+=int(np.count_nonzero(ids[r,live:]<0));ignored_nonnegative+=int(np.count_nonzero(ids[r,live:]>=0))
 assert not bad_live and not unique,(bad_live,unique)
 item['ids']={'valid_live_picks':live_total,'causal_bound_checked':True,'unique_per_query':True,'ignored_negative':ignored_negative,'ignored_nonnegative':ignored_nonnegative,'live_min':min(512,(pos+1)//ratio),'live_max':min(512,(pos+rows)//ratio)}
 carry=np.memmap(str(prefix)+'-carry-f16.bin',mode='r',dtype='<f2',shape=(127,512));valid=min(pos,127)
 item['carry']={'valid_rows':valid,'all_finite':bool(np.isfinite(carry).all()),'valid_finite':bool(np.isfinite(carry[127-valid:]).all()) if valid else True,'ignored_leading_rows':127-valid}
 assert item['carry']['valid_finite']
 if previous:
  pm,pp=previous;assert pos==pm['base_pos']+pm['c_start']+pm['rows']
  kv=np.memmap(str(pp)+'-kv-f32.bin',mode='r',dtype='<f4',shape=(pm['rows'],512))
  wanted=kv[-127:].astype('<f2');diff=int(np.count_nonzero(wanted.view('<u2')!=carry.view('<u2')))
  item['carry']['previous_chunk_tail_bitwise_differences']=diff;assert diff==0
  previous_comp=Path(str(pp)+'-comp.bin').read_bytes();assert Path(str(prefix)+'-comp.bin').read_bytes()[:len(previous_comp)]==previous_comp
  item['compressed_prefix_matches_previous']=True
  assert Path(str(prefix)+'-sinks-f32.bin').read_bytes()==Path(str(pp)+'-sinks-f32.bin').read_bytes()
 else:item['carry']['first_chunk_not_read']=True
 item['values']=stats;result.append(item);previous=(meta,prefix)
assert len(result)==4
out=root.parent/'attention-fixture-audit.json';out.write_text(json.dumps({'pass':True,'n':4,'total_rows':sum(x['metadata']['rows'] for x in result),'fixtures':result},indent=2)+'\n');print(out)
for x in result:print(Path(x['prefix']).name,x['ids'],x['carry'])
