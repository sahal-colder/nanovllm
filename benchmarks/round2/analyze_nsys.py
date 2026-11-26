#!/usr/bin/env python3
"""Derive launch counts and real timeline bars from an exported Nsys SQLite."""
import argparse
import csv
import json
from pathlib import Path
import sqlite3

p=argparse.ArgumentParser(description=__doc__);p.add_argument('sqlite',type=Path);p.add_argument('--plot',action='store_true');a=p.parse_args()
assert a.sqlite.is_file()
c=sqlite3.connect('file:'+str(a.sqlite.resolve())+'?mode=ro',uri=True)
c.row_factory=sqlite3.Row
regions=c.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text IN ('decode_eager','decode_graph') ORDER BY start").fetchall()
assert len(regions)==2
summary={};traces={}
for region in regions:
    start,end,label=region
    api=c.execute('SELECT r.start,r.end,s.value AS name FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE r.start>=? AND r.end<=? ORDER BY r.start',(start,end)).fetchall()
    kernels=c.execute('SELECT k.start,k.end,s.value AS name,k.graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName WHERE k.start>=? AND k.end<=? ORDER BY k.start',(start,end)).fetchall()
    memsets=c.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start>=? AND end<=?',(start,end)).fetchall()
    launches=[x for x in api if x['name'].startswith('cudaLaunchKernel')]
    graphs=[x for x in api if x['name'].startswith('cudaGraphLaunch')]
    spans=sorted([(x['start'],x['end']) for x in list(kernels)+list(memsets)])
    first,last=spans[0];active=0
    for left,right in spans[1:]:
        if left<=last:last=max(last,right)
        else:active+=last-first;first,last=left,right
    active+=last-first;window=max(x[1] for x in spans)-min(x[0] for x in spans)
    key=label.removeprefix('decode_')
    summary[key]=dict(kernel_launch_api_calls=len(launches),graph_launch_api_calls=len(graphs),gpu_kernels=len(kernels),
                      device_memsets=len(memsets),distinct_graph_nodes=len(set(x['graphNodeId'] for x in kernels if x['graphNodeId'] is not None)),
                      launch_api_total_us=sum(x['end']-x['start'] for x in launches+graphs)/1000,
                      gpu_activity_window_us=window/1000,gpu_active_us=active/1000,gpu_gap_us=(window-active)/1000,
                      gpu_gap_percent=100*(window-active)/window,nvtx_range_wall_us=(end-start)/1000)
    traces[key]=(region,launches+graphs,kernels,memsets)
steps=summary['graph']['graph_launch_api_calls'];assert steps==5
assert summary['graph']['kernel_launch_api_calls']==0
assert summary['eager']['kernel_launch_api_calls']==summary['eager']['gpu_kernels']==summary['graph']['gpu_kernels']==51*steps
assert summary['graph']['distinct_graph_nodes']==51
summary['steps_per_mode']=steps
summary['host_launches_per_step']={'eager':51,'graph':1}
summary['scope']='Five instrumented steps/mode; kernel+device-memset activity union; gaps include event/synchronization and tracing overhead. Use unprofiled 100-sample CSV for latency.'
out=a.sqlite.parent
(out/'timeline-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (out/'timeline-summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['mode',*summary['eager']]);writer.writeheader()
    for key in ['eager','graph']:writer.writerow(dict(mode=key,**summary[key]))
print(json.dumps(summary,indent=2))
if a.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    colors={'gemv':'#2A7F62','rms':'#3979AB','attention':'#CE8A27','rope':'#8A5AA8','other':'#65758B'}
    fig,axes=plt.subplots(4,1,figsize=(13,5.4),sharex=True,gridspec_kw={'height_ratios':[.65,1,.65,1]},layout='constrained')
    for row,key in enumerate(['eager','graph']):
        region,api,kernels,memsets=traces[key];origin=region['start']
        ax=axes[2*row];ax.broken_barh([((x['start']-origin)/1e6,max((x['end']-x['start'])/1e6,.0003)) for x in api],(.2,.6),facecolors='#25374A')
        ax.set_ylabel(key+'\nHost API',rotation=0,ha='right',va='center');ax.set_ylim(0,1);ax.set_yticks([])
        ax.text(.995,.87,f'{len(api)} launch calls / 5 steps',transform=ax.transAxes,ha='right',fontsize=9)
        ax=axes[2*row+1]
        for kernel in kernels:
            name=kernel['name'];cat='gemv' if 'gemv' in name else 'rms' if 'rms' in name else 'attention' if 'attention' in name else 'rope' if 'rope' in name else 'other'
            ax.broken_barh([((kernel['start']-origin)/1e6,(kernel['end']-kernel['start'])/1e6)],(.15,.7),facecolors=colors[cat])
        ax.set_ylabel(key+'\nGPU kernels',rotation=0,ha='right',va='center');ax.set_ylim(0,1);ax.set_yticks([])
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.spines[['top','right','left']].set_visible(False)
    axes[-1].set_xlabel('Milliseconds from each NVTX range start')
    fig.suptitle('Nsight Systems: same 255 GPU kernels, 255 eager launches versus 5 Graph launches',fontsize=12)
    axes[1].legend(handles=[Patch(color=v,label=k) for k,v in colors.items()],loc='upper center',bbox_to_anchor=(.5,1.6),ncol=5,frameon=False,fontsize=8)
    fig.savefig(out/'decode-timeline.png',dpi=180)
    plt.close(fig)
