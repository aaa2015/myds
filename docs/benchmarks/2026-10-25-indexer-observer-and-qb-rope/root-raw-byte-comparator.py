import json,hashlib,struct,math,sys
from pathlib import Path
root,out=map(Path,sys.argv[1:3]);providers=sys.argv[3].split(',');runs=range(int(sys.argv[4]));b=root/providers[0]/'state/run-0'
m=json.loads((b/'manifest.json').read_text());names=[x['path'] for x in m['files']]
assert len(names)==len(set(names)) and all(Path(n).name==n for n in names)
base={n:(b/n).read_bytes() for n in names};gold=(root/providers[0]/'logits/run-0.f32').read_bytes()
report={'kind':'independent_root_raw_only_comparator','scope':'complete manifested state/output/token bytes and final logits; not provenance/schema certification','root':str(root),'rows':[],'failures':[]};total=0
for provider in providers:
 for i in runs:
  d=root/provider/'state'/('run-'+str(i));mm=json.loads((d/'manifest.json').read_text());ns=[x['path'] for x in mm['files']];bad=[];nb=0
  assert all(Path(n).name==n for n in ns)
  if ns!=names:bad.append('file list')
  for n in ns:
   raw=(d/n).read_bytes();nb+=len(raw)
   if raw!=base.get(n):bad.append(n)
  logits=(root/provider/'logits'/('run-'+str(i)+'.f32')).read_bytes();nb+=len(logits);total+=nb
  if logits!=gold:bad.append('final logits')
  if len(logits)%4 or not all(math.isfinite(x[0]) for x in struct.iter_unpack('<f',logits)):bad.append('nonfinite/truncated logits')
  report['rows'].append({'provider':provider,'run':i,'state_files':len(ns),'raw_bytes':nb,'logits_elements':len(logits)//4,'logits_sha256':hashlib.sha256(logits).hexdigest(),'all_equal':not bad,'differences':bad})
  if bad:report['failures'].append([provider,i,bad])
report['raw_bytes']=total;report['all_equal']=not report['failures'];out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'report':str(out),'rows':len(report['rows']),'state_files_per_row':len(names),'raw_bytes':total,'all_equal':report['all_equal'],'failure_rows':len(report['failures'])}))
