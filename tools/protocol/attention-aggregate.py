#!/usr/bin/env python3
"""Aggregate probe JSONL only; segmented stages have a different command graph."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
records=[];metadata=[];verify=[]
for source in a.inputs:
 for line in Path(source).read_text().splitlines():
  d=json.loads(line)
  if d['kind']=='timing':
   if not d['gpu_time_valid'] or d['gpu_ms']<0 or d['wall_ms']<0:raise SystemExit(f'Invalid time: {source}')
   d['source']=source;records.append(d)
  elif d['kind']=='metadata':metadata.append(d)
  elif d['kind'] in ('verify','capture_verify','reference'):
   if not d['pass']:raise SystemExit(f'Failed correctness: {source}: {d}')
   verify.append(d)
groups=defaultdict(list)
for d in records:groups[(d['source'],d['rows'],d['start'],d['ratio'],d['n_comp'],d['variant'],d['stage'])].append(d)
result=[]
for key,ds in sorted(groups.items()):
 source,rows,start,ratio,ncomp,variant,stage=key
 if len({d['iteration'] for d in ds})!=len(ds):raise SystemExit('Duplicate measured iterations')
 base={d['iteration']:d for d in groups.get((source,rows,start,ratio,ncomp,'vec',stage),[])}
 row=dict(source=source,rows=rows,start=start,ratio=ratio,n_comp=ncomp,variant=variant,stage=stage,n=len(ds))
 for metric in ('gpu_ms','wall_ms'):
  xs=[d[metric] for d in ds];row.update({metric+'_median':statistics.median(xs),metric+'_min':min(xs),metric+'_max':max(xs)})
  paired=[d[metric]-base[d['iteration']][metric] for d in ds if d['iteration'] in base]
  row[metric+'_paired_delta_median']=statistics.median(paired) if paired else None
 result.append(row)
root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
(root/'attention-summary.json').write_text(json.dumps(dict(metadata=metadata,correctness=verify,rows=result,scope='isolated component; segmented stage CBs change submission; no end-to-end projection'),indent=2)+'\n')
if result:
 with (root/'attention-stages.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
print(f'{len(records)} timings; {len(verify)} successful correctness records; {len(result)} summary rows')
