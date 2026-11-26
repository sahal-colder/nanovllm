#!/usr/bin/env python3
"""Validate final audit artifacts and publish tables derived only from real CSV."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);p.add_argument('--publish',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=a.results.resolve();rel=out.relative_to(root)
def read(path):return list(csv.DictReader(path.open()))
def number(x):return float(str(x).replace(',',''))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,r))+' |' for r in rows],'']
states={}
for phase,path in [('performance',out),('ncu',out/'ncu'),('decode',out),('validation',out)]:
    state=json.loads((path/(phase+'-status.json')).read_text());states[phase]=state
    assert state['status']=='complete' or (phase=='validation' and state['status']=='partial' and state.get('remaining')==['envelope-large-f16-racecheck']),phase
attention=[]
for f in sorted(out.glob('attention-*.csv')):
    rows=read(f);assert len(rows)==3
    assert {r['implementation'] for r in rows}=={'serial_online','split_online_vec4','unsplit_vec4'}
    for r in rows:
        b,h,kv,d,s=(int(r[k]) for k in ['batch','heads','kv_heads','head_dim','seq_len']);size=4 if r['dtype']=='f32' else 2
        kv_bytes=size*2*sum(s-3*i for i in range(b))*kv*d
        assert int(r['ideal_bytes'])==kv_bytes+size*2*b*h*d
        assert number(r['p95_us'])>=number(r['p50_us'])>0 and int(r['samples'])==100
        assert math.isclose(number(r['effective_GiB_s']),int(r['ideal_bytes'])/number(r['p50_us'])/1e-6/2**30,rel_tol=3e-6)
        assert int(r['flush_bytes'])==(134217728 if r['cache_policy']=='cold' else 0)
        r['kv_MiB']=kv_bytes/2**20;r['source']=str(f.relative_to(root));attention.append(r)
assert len(attention)==18
operators=[]
for cache in ['hot','cold']:
    rows=read(out/f'operators-{cache}.csv');assert len(rows)==15
    for r in rows:
        assert r['reference']=='cpu_f64' and r['cache_policy']==cache
        assert number(r['p95_us'])>=number(r['p50_us'])>0 and int(r['samples'])*int(r['inner'])==100
        assert int(r['flush_bytes'])==(134217728 if cache=='cold' else 0)
        r['source']=str((out/f'operators-{cache}.csv').relative_to(root));operators.append(r)
ncu=[]
units={'byte':1,'Kbyte':1e3,'Mbyte':1e6,'Gbyte':1e9,'byte/s':1,'Kbyte/s':1e3,'Mbyte/s':1e6,'Gbyte/s':1e9,'Tbyte/s':1e12,'ns':.001,'us':1,'ms':1000,'s':1e6,'%':1}
for r in read(out/'ncu/metrics.csv'):
    def metric(m):return number(r[m])*units[r[m+'_unit']]
    ncu.append(dict(case=r['case'],kernel=r['kernel'],dram_pct=metric('dram__throughput.avg.pct_of_peak_sustained_elapsed'),dram_GiB_s=metric('dram__bytes.sum.per_second')/2**30,read_MiB=metric('dram__bytes_read.sum')/2**20,write_MiB=metric('dram__bytes_write.sum')/2**20,occupancy_pct=metric('sm__warps_active.avg.pct_of_peak_sustained_active'),l2_hit_pct=metric('lts__t_sector_hit_rate.pct'),profiled_duration_us=metric('gpu__time_duration.sum')))
assert len(ncu)==28
for suffix in ['m0','m1','m2']:
    hot=next(r for r in ncu if r['case']=='attention-f32-hot-'+suffix and 'partition' in r['kernel'])
    cold=next(r for r in ncu if r['case']=='attention-f32-cold-'+suffix and 'partition' in r['kernel'])
    assert hot['read_MiB']<.1 and cold['read_MiB']>=16, 'L2 eviction hardware check'
decoder=[]
for name in ['small-f16','large-f16']:
    rows=read(out/f'decode-{name}.csv');assert len(rows)==6
    assert {(r['scope'],r['mode']) for r in rows}=={(s,m) for s in ['compute','full_cpu_greedy','full_gpu_greedy'] for m in ['eager','graph']}
    for r in rows:
        assert r['dtype']=='f16' and r['cache_policy']=='cold' and int(r['samples'])==100
        assert r['sampling_order']=='alternating_pairs'
        assert 0<number(r['p50_wall_us'])<=number(r['p95_wall_us'])
        assert 0<number(r['p50_device_us'])<=number(r['p95_device_us'])
        r['source']=str((out/f'decode-{name}.csv').relative_to(root));decoder.append(r)
    log=(out/f'decode-{name}.log').read_text();assert 'PASS audit decoder' in log
    assert log.count('PASS dynamic graph_eager_bitwise_equal=1')==5
ctest=(out/'ctest.stdout').read_text();assert '100% tests passed, 0 tests failed out of 8' in ctest
validation=[];large_race_log=out/'envelope-large-f16-racecheck.log'
large_race_text=large_race_log.read_text() if large_race_log.exists() else ''
large_race_pass='RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in large_race_text
# A recorded failure cannot be hidden by treating a completed summary as missing.
assert 'RACECHECK SUMMARY:' not in large_race_text or large_race_pass,large_race_log
remaining=[]
for tool in ['memcheck','racecheck']:
    for case in ['decoder','gemv-rms','attention-cold','attention-bf16-cold','envelope-f32','envelope-f16','envelope-bf16','envelope-large-f16']:
        if tool=='racecheck' and case=='envelope-large-f16' and not large_race_pass:
            remaining.append(f'{case}-{tool}');continue
        f=out/f'{case}-{tool}.log';txt=f.read_text()
        assert ('ERROR SUMMARY: 0 errors' if tool=='memcheck' else '0 errors, 0 warnings') in txt,f
        numeric=(out/f'{case}-{tool}-validation.log').read_text()
        assert 'FAIL ' not in numeric,(case,tool)
        if case.startswith('envelope-'):assert 'PASS audit decoder' in numeric,(case,tool)
        elif case=='gemv-rms':assert 'PASS 297 random/boundary cases' in numeric
        elif case=='decoder':assert 'PASS attention cases=90' in numeric and 'PASS decode layers=3' in numeric
        else:
            checked=read(out/f'{case}-{tool}.stdout')
            assert len(checked)==3 and {r['implementation'] for r in checked}=={'serial_online','split_online_vec4','unsplit_vec4'}
            assert all(int(r['samples'])==5 and r['cache_policy']=='cold' for r in checked)
        validation.append(str(f.relative_to(root)))
validation_state=states['validation']
if remaining:
    assert validation_state['status']=='partial','Incomplete Racecheck cannot be complete'
    assert validation_state.get('remaining')==remaining
    assert validation_state.get('completed_cases')==len(validation) and validation_state.get('total_cases')==16
else:
    assert validation_state['status']=='complete','All-case success requires a complete validation phase'
    assert validation_state.get('completed_cases')==16 and validation_state.get('total_cases')==16
    assert validation_state.get('remaining')==[]
resolution=None;resolution_dir=out/'racecheck-resolution'
if large_race_pass and resolution_dir.exists():
    resolution=json.loads((resolution_dir/'summary.json').read_text())
    assert resolution['status']=='complete' and resolution['sanitizer_matrix_completed']==16
    assert json.loads((resolution_dir/'resolution-status.json').read_text())['status']=='complete'
    identity=json.loads((resolution_dir/'identity.json').read_text())
    commands=json.loads((resolution_dir/'commands.json').read_text())
    assert [c['name'] for c in commands]==['ordinary','memcheck','racecheck']
    assert all('--validate-decode' in c['argv'] and not any(x.startswith(('--kernel-name','--launch-skip','--launch-count')) for x in c['argv']) for c in commands)
    assert '--racecheck-num-workers' in commands[-1]['argv']
    for name in ['ordinary','memcheck','racecheck']:
        txt=(resolution_dir/(name+'-validation.log')).read_text()
        assert 'FAIL ' not in txt and 'PASS audit decoder dtype=f16 layers=16 dynamic_steps=5' in txt
        assert txt.count('PASS dynamic graph_eager_bitwise_equal=1')==5
        assert txt.count('PASS validation scope=')==3 and 'KV_slots=16777216' in txt
        assert 'PASS validation_only=1 timing_samples=0 warmup_steps=0 flush_calls=0 kernel_filter=none' in txt
        assert not (resolution_dir/(name+'.stdout')).read_text()
    assert 'ERROR SUMMARY: 0 errors' in (resolution_dir/'memcheck.log').read_text()
    assert (resolution_dir/'racecheck.log').read_text()==large_race_text
    assert (resolution_dir/'racecheck-validation.log').read_text()==(out/'envelope-large-f16-racecheck-validation.log').read_text()
    resource_rows=[json.loads(x) for x in (resolution_dir/'telemetry.jsonl').read_text().splitlines()]
    capped=[cg for r in resource_rows for cg in r.get('process_cgroup',{}).values() if cg.get('memory.max')==str(6*2**30)]
    assert capped and all(cg['memory.swap.max']=='0' and cg['memory.swap.current']=='0' for cg in capped)
    assert all('oom_kill 0' in cg['memory.events'] for cg in capped)
    resolution['observed_cgroup_memory_peak_MiB']=max(int(cg['memory.peak'])/2**20 for cg in capped)
    resolution['observed_cgroup_swap_bytes']=0
nsys=json.loads((out/'nsys/timeline-summary.json').read_text());assert nsys['host_launches_per_step']=={'eager':51,'graph':1}
assert 'f16' in (out/'nsys/profile.stdout').read_text()
bf16_log=out/'rejected-decode-bf16/diagnostic.log'
bf16_failed=bf16_log.is_file() and 'FAIL dynamic transfer Graph CPU FP64' in bf16_log.read_text()
telemetry=[]
for name in ['telemetry.jsonl','ncu/telemetry.jsonl','decode-telemetry.jsonl','validation-telemetry.jsonl','nsys/telemetry.jsonl']+(['racecheck-resolution/telemetry.jsonl'] if resolution else []):
    data=[json.loads(x) for x in (out/name).read_text().splitlines()];assert data
    for r in data:
        assert not r['compute'] or r['allowed_root_pid'],r
        assert not any('VLLM::' in str(x) for x in r['compute']),r
    temperature=[int(r['gpu'].split(', ')[5]) for r in data]
    telemetry.append(dict(file=name,snapshots=len(data),sm_mem_MHz=sorted({tuple(r['gpu'].split(', ')[3:5]) for r in data}),min_temp=min(temperature),max_temp=max(temperature)))
env=json.loads((out/'environment.json').read_text())
if resolution:
    assert all(env['source_sha256'][k]==v for k,v in identity['unchanged_production_source_sha256'].items())
summary=dict(validation_status=validation_state['status'],validation_completed_cases=len(validation),validation_total_cases=16,validation_remaining=remaining,phase_states=states,large_full_racecheck_passed=large_race_pass,racecheck_resolution=resolution,environment=env,attention=attention,operators=operators,ncu=ncu,decoder=decoder,validation=validation,nsys=nsys,telemetry=telemetry,known_failure='16-layer BF16 dynamic decode exceeds unchanged CPU FP64 tolerance; eager and Graph bitwise identical at failing position; no BF16 large-model performance claim' if bf16_failed else '16-layer BF16 not part of this accepted performance matrix')
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
for name,rows in [('attention',attention),('operators',operators),('ncu',ncu),('decoder',decoder)]:
    with (out/f'summary-{name}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def att(dt,cache,impl):return next(r for r in attention if r['dtype']==dt and r['cache_policy']==cache and r['implementation']==impl)
def op(dt,kind,cache,impl):return next(r for r in operators if r['dtype']==dt and r['operator']==kind and r['cache_policy']==cache and r['implementation']==impl)
def dec(layers,scope,mode):return next(r for r in decoder if int(r['layers'])==layers and r['scope']==scope and r['mode']==mode)
def nmetric(case,merge=False):return next(r for r in ncu if r['case']==case and (('merge' in r['kernel'])==merge))
def pct(e,g,key):return 100*(1-number(g[key])/number(e[key]))
def f(x,d=2):return f'{number(x):.{d}f}'
text=['# CUDA 评测审计与实测报告（2026-11-26，Australia/Perth）','',
'本轮在用户释放GPU后完成正式重测。采用RTX 4060 Laptop GPU，所有GPU工作串行执行；每个阶段的进程守卫均未发现外部CUDA计算任务。此前并发VLLM干扰及失败阶段保留，但不进入正式性能结论。',
'',f'正式证据目录：`{rel}/`。性能、NCU、Decode均完成；现有{len(validation)}/16项sanitizer矩阵通过。'+('16层FP16完整链路的独立正确性入口已通过Racecheck；旧计时循环插桩中断记录仍保留。' if large_race_pass else '**额外16层整步Racecheck未完成，日志无完成摘要，不能算通过；不再自动重跑高开销任务。**')+'新结果与历史热缓存日志分开，以下除明确说明外均来自本轮真实实测。',
'','## 1. 评测审计与实测报告','','### 硬件、计时与正确性口径','',
'RTX 4060 Laptop GPU，24 SM、CC8.9、128-bit显存总线；nvidia-smi显存8188 MiB，L2为**33,554,432 bytes = 32 MiB**。Driver615.71.09，NVCC13.4.92，cuBLAS130800，NCU2026.3.1，NSys2026.3.2。',
'',
'未锁频；每秒保存进程、温度、SM/显存频率快照。实测存在2010/2580 MHz及7001/8001 MHz等动态频率状态，单次snapshot不能替代Kernel瞬时频率。表内延迟与NCU独立采样，不能直接用不同pass的时间反推同一次带宽。小幅差异不作为稳定优化承诺。',
'',
'冷缓存：每次调用前在同一stream完整读改写128 MiB驱逐区（4×L2）；flush位于start event之前，不计入算子延迟。允许同一次调用内缓存复用。Attention partition与merge之间不驱逐。热缓存只用于同地址warmup重复调用的对照。每方法100个event样本，报告P50/P95；有效GiB/s按逻辑唯一张量字节计算，区别于NCU真实DRAM字节。',
'',
'全量CTest **8/8**通过。算子、新增三路Attention、FP16/BF16超L2冷调用、三dtype小模型传输/greedy Graph的Memcheck/Racecheck均零错误、Racecheck零警告；'+('16层FP16独立正确性入口也通过普通数值、Memcheck与Racecheck，未过滤任何Kernel。' if large_race_pass else '16层FP16完整步的普通数值与Memcheck通过，额外整步Racecheck未完成。')+'完整16层FP16逐16,777,216个物理KV槽对照CPU参考，并检查5个动态步的Graph/eager logits逐位一致。',
'',
'CPU参考使用独立FP64算术并显式模拟存储舍入；验收是**容差内零校验失败，不是浮点零误差**。Attention atol/rtol：FP32=2e-5/3e-4，FP16=1e-3/2e-3，BF16=8e-3/1.6e-2；GEMV/RMS为1e-4/2e-4、1e-3/2e-3、8e-3/1.6e-2；完整前向为1e-4/3e-4、3e-3/8e-3、2e-2/5e-2。逐元素满足`abs_err <= atol + rtol*abs(ref)`，近零max_rel需与max_abs一起解释。',
'','### Attention：冷/热差异及Baseline','',
'历史73.73 µs的FP32 B1/H16/KV4/D128/S4096循环未显式驱逐，唯一KV恰为16 MiB，小于32 MiB L2；原62.60% DRAM和75.86% Occupancy来自另一轮`ncu --cache-control all`，两者不能组合为同一冷缓存测量。',
'']
rows=[]
for impl in ['serial_online','unsplit_vec4','split_online_vec4']:
    h,c=att('f32','hot',impl),att('f32','cold',impl)
    rows.append([impl,f(h['p50_us'])+'/'+f(h['p95_us']),f(c['p50_us'])+'/'+f(c['p95_us']),f(number(c['p50_us'])/number(h['p50_us']),3)+'×',f(c['effective_GiB_s'])])
text+=table(['FP32 B1/S4096','hot P50/P95 µs','cold P50/P95 µs','cold/hot P50','cold有效GiB/s'],rows)
h,c=att('f32','hot','split_online_vec4'),att('f32','cold','split_online_vec4');hot=nmetric('attention-f32-hot-m1');cold=nmetric('attention-f32-cold-m1')
text += [f'分段路径清L2后P50由{f(h["p50_us"])}增至{f(c["p50_us"])} µs，增加{f(number(c["p50_us"])-number(h["p50_us"]))} µs / {f(100*(number(c["p50_us"])/number(h["p50_us"])-1))}%。NCU应用重放中，partition热缓存DRAM读取{f(hot["read_MiB"],3)} MiB，驱逐后{f(cold["read_MiB"],3)} MiB，交叉验证驱逐生效。',
'',
'Baseline精确定义：`serial_online`每query head一个warp、4 warps/CTA，整条序列不分段；不是成熟Attention库，也不是“每序列恰好一个block”。新增`unsplit_vec4`保留向量布局、只去掉分段，与`split_online_vec4`比较可隔离分段并行度及合并的净收益。',
'',
'FP16/BF16新形状：B2/H16/KV4/D128，长度[32768,32765]，GQA比4，唯一KV **127.994 MiB**，远超L2；Q/KV/output按对应dtype存储，均FP32累加。页表乱序、页尾投毒。',
'']
rows=[]
for dt in ['f32','f16','bf16']:
    b,u,s=(att(dt,'cold',i) for i in ['serial_online','unsplit_vec4','split_online_vec4'])
    rows.append([dt,'B1/S4096' if dt=='f32' else 'B2/S32768',f(b['kv_MiB'],3),f(b['p50_us']),f(u['p50_us']),f(s['p50_us'])+'/'+f(s['p95_us']),f(number(b['p50_us'])/number(s['p50_us']))+'×',f(number(u['p50_us'])/number(s['p50_us']))+'×',f(b['effective_GiB_s'])+'/'+f(u['effective_GiB_s'])+'/'+f(s['effective_GiB_s'])])
text+=table(['dtype','形状H16/KV4/D128','KV MiB','标量未分段P50 µs','向量未分段P50 µs','分段P50/P95 µs','vs标量','vs向量未分段','有效GiB/s 标量/向量/分段'],rows)
rows=[]
for dt in ['f32','f16','bf16']:
    checked=[r for r in attention if r['dtype']==dt and r['cache_policy']=='cold']
    rows.append([dt,max(int(r['batch'])*int(r['heads'])*int(r['head_dim']) for r in checked),f'{max(number(r["max_abs_err"]) for r in checked):.9g}',f'{max(number(r["max_rel_err"]) for r in checked):.9g}','全部三路通过混合容差'])
text+=table(['dtype','每路输出元素','三路最大abs error','三路最大rel error','CPU FP64判定'],rows)
rows=[]
for dt in ['f32','f16','bf16']:
    for method,label,merge in [(0,'scalar unsplit',False),(2,'vec4 unsplit',False),(1,'partition',False),(1,'merge',True)]:
        r=nmetric(f'attention-{dt}-cold-m{method}',merge)
        rows.append([dt,label,f(r['dram_pct']),f(r['dram_GiB_s']),f(r['occupancy_pct']),f(r['read_MiB'],3),f(r['l2_hit_pct'])])
text+=table(['dtype','Kernel','DRAM %','真实DRAM GiB/s','Achieved Occupancy %','DRAM read MiB','L2 hit %'],rows)
text += ['NCU Attention使用`--replay-mode application --cache-control none --profile-from-start off`：应用在profile range前显式驱逐，每pass重建相同输入，保留partition→merge真实缓存依赖。热缓存对照不驱逐。未分段只有4/8个CTA（B1/B2），Occupancy低；分段后展开序列并行度，但不能将partition指标称作整个Attention指标，也不能简单平均两个kernel的DRAM%/Occupancy。',
'',
'FP16/BF16大形状的不驱逐结果也保存在CSV，但超L2不等于无缓存命中，且动态频率可能使不驱逐版本略慢；不把这种小幅反转解释成清缓存可加速。简历采用cold结果。',
'','### GEMV与RMSNorm：同精度、同缓存、同Baseline','',
'GEMV为8192×4096/B1，FP32权重128 MiB，FP16/BF16权重64 MiB，均超L2；仍对每次调用显式驱逐。cuBLAS同存储精度，FP32用Sgemv、FP16/BF16用GemmEx，PEDANTIC FP32累加，禁用低精度中间归约和TF32。审计形状的全部方法逐元素对齐CPU FP64。',
'']
rows=[]
for dt in ['f32','f16','bf16']:
    b=op(dt,'sgemv','cold','baseline');s=op(dt,'sgemv','cold','optimized');lib=op(dt,'sgemv','cold','cublasSgemv' if dt=='f32' else 'cublasGemmEx');r=nmetric(f'gemv-{dt}-m1')
    rows.append([dt,f(b['p50_us']),f(s['p50_us'])+'/'+f(s['p95_us']),f(lib['p50_us']),f(number(lib['p50_us'])/number(s['p50_us']),3)+'×',f(s['effective_GiB_s']),f(r['dram_GiB_s']),f(r['dram_pct']),f(r['occupancy_pct'])])
text+=table(['dtype','naive P50 µs','optimized P50/P95 µs','cuBLAS P50 µs','cuBLAS/optimized','有效GiB/s','NCU DRAM GiB/s','DRAM %','Occupancy %'],rows)
text += ['**新冷缓存结果不支持“GEMV优于cuBLAS”。** 本实现可有较高DRAM利用率，但flush后的常规event延迟仍比cuBLAS慢。NCU replay/频率与正常计时不同，不能拿profiler中的单kernel时长替代100样本P50，也不能把硬件利用率直接变成端到端加速。历史230.06 GiB/s、96.59%为旧FP32冷NCU实测，保留归档，新表不覆盖旧证据。',
'',
'RMSNorm为128×4096；X+Y+gamma逻辑字节FP32为4.016 MiB、FP16/BF16为2.008 MiB，小于L2。历史8.192→6.144 µs是FP32热缓存，不能标冷。两个版本都是256 threads/row与warp/block归约；optimized的差别是向量读写、X驻留寄存器，baseline在输出阶段重读X。',
'']
rows=[]
for dt in ['f32','f16','bf16']:
    for cache in ['hot','cold']:
        b,s=(op(dt,'rmsnorm',cache,i) for i in ['baseline','optimized'])
        rows.append([dt,cache,f(b['p50_us'])+'/'+f(b['p95_us']),f(s['p50_us'])+'/'+f(s['p95_us']),f(number(b['p50_us'])/number(s['p50_us']))+'×'])
text+=table(['dtype','cache','标量P50/P95 µs','向量/寄存器P50/P95 µs','加速比'],rows)
text += ['RMS有效带宽分子统一为唯一X/Y/gamma逻辑字节，baseline重读X不膨胀分子；实际DRAM计数见NCU CSV。近微秒event有量化影响，报告当前样本点估计，不宣称跨机器固定收益。',
'','### CUDA Graph：更大模型、完整步与架构权衡','',
'原4层d512/FFN1536/H8/KV2/vocab512/B1/seq129随机权重模型的FP16历史计算链路250.343→221.740 µs（11.43%），未显式驱逐且排除metadata与采样。它适合展示小算子/小Batch下Launch开销，不能外推成生产模型吞吐提升。',
'',
'新测试包括同小模型及16层d1024/FFN2816/H16/KV4/vocab8192、容量2048、B1/seq129；均是随机权重、固定输入重复单步，非训练模型生成TPS。大模型FP32 checkpoint755,634,204 bytes，seed719；FP16权重流量远超L2。每个样本先驱逐并同步，再开始wall/event计时。**每个scope中eager/Graph按成对交错、交替先后顺序采样，预热双方各20次**，降低长批次顺序偏差。较早顺序计时数据仅归档，不作为正式收益。',
'',
'`compute`=Host提交到计算完成；`full_cpu_greedy`=staging拷贝、token/length/table H2D、计算、logits/error D2H、CPU unpack+argmax；`full_gpu_greedy`=staging、3次H2D、计算、GPU argmax及4-byte Token/error回传。两个full Graph均实际捕获传输。均不含CPU页分配/请求调度、tokenizer、文本输出。',
'']
rows=[]
for layers in [4,16]:
    for scope in ['compute','full_cpu_greedy','full_gpu_greedy']:
        e,g=(dec(layers,scope,m) for m in ['eager','graph'])
        rows.append([layers,scope,f(e['p50_wall_us'])+'/'+f(e['p95_wall_us']),f(g['p50_wall_us'])+'/'+f(g['p95_wall_us']),f(pct(e,g,'p50_wall_us'))+'%',f(e['p50_device_us'])+'→'+f(g['p50_device_us'])])
text+=table(['FP16层数','scope','eager wall P50/P95 µs','Graph wall P50/P95 µs','P50降低','device P50 µs'],rows)
e,g=(dec(16,'full_gpu_greedy',m) for m in ['eager','graph'])
text += [f'16层完整GPU greedy步的同scope Graph收益为{f(pct(e,g,"p50_wall_us"))}%，绝对节省{f(number(e["p50_wall_us"])-number(g["p50_wall_us"]))} µs。只能比较同scope中的eager/Graph；GPU greedy减少回读与CPU处理的收益不能全部归因于Graph。生产引擎仍使用CPU sampler，本轮传输/argmax集成为benchmark原型。',
'',
'**明确数值限制：16层BF16动态前向未通过原有CPU FP64容差。** 在pos143得到-0.0849609375，对照-0.11328125，abs=0.0283203125；允许界约0.025664，故判失败。该步eager与Graph所有8192个logits逐位一致，排除了Graph更新造成差异；更符合深层BF16舍入累积的表现，但不把该推断当已证明根因。未放宽容差、未换seed、未发布失败阶段的性能。BF16 Attention超L2、原1/3层链路和新4层传输Graph仍独立通过验收。16层Graph简历收益仅引用FP16。' if bf16_failed else '16层BF16不在本次正式性能矩阵中；其余dtype/形状只引用对应验收证据。',
'',
'已知BF16数值边界复现（当前seed下应报FAIL，结果不进入正式矩阵）：`build-cuda/bench_decode --audit-decode build-cuda/fixtures/audit16.bin bf16 5`。完整失败日志与原采集命令位于`rejected-decode-bf16/`。' if bf16_failed else '',
'',
'架构权衡：设备地址、shape与容量固定，seq_len/Block Table是设备数据，更新内容不需重建Graph。旧版把H2D放外部属于接口与pinned staging生命周期选择，并非无法捕获memcpy；固定pinned地址可把上传/回传捕获进Graph。CPU修改staging前必须确认上一轮已消费。GPU greedy可只回传Token ID，但CPU页管理、抢占与temperature/top-p语义仍需单独设计；单请求下一步依赖当前Token，不能声称所有传输可异步隐藏。跨请求双缓冲/多stream应在真正调度器中另测。',
'','### NSys：同精度FP16新证据','',
f'NSys trace使用FP16固定输入重复计算步，未显式驱逐，仅用于Launch归因；每模式5步：eager={nsys["eager"]["kernel_launch_api_calls"]}次Kernel API、Graph={nsys["graph"]["graph_launch_api_calls"]}次Graph API；两边各{nsys["graph"]["gpu_kernels"]}个GPU Kernel。每步Host提交**51→1**，GPU Kernel仍51个，另有1个memset（计算Graph共52节点）。',
'',
f'插桩GPU活动窗口间隙比例{f(nsys["eager"]["gpu_gap_percent"])}%→{f(nsys["graph"]["gpu_gap_percent"])}%；包含event/同步与profiler影响，仅用于归因，不用来替换正式延迟。旧NSys默认FP32，不能将旧间隙比率当作FP16同精度数据。',
'',f'![FP16 NSys真实时间线](../{rel}/nsys/decode-timeline.png)',
'','### 复现命令与证据身份','','```sh',
'cmake -S . -B build-cuda -DNANO_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89',
'cmake --build build-cuda -j4',
'python3 benchmarks/audit/make_model.py build-cuda/fixtures/audit16.bin',
'python3 benchmarks/audit/run_audit.py --out benchmarks/audit/new-run',
'python3 benchmarks/audit/profile_audit.py build-cuda --out benchmarks/audit/new-run/ncu',
'python3 benchmarks/audit/run_decode.py build-cuda benchmarks/audit/new-run',
'python3 benchmarks/audit/validate_audit.py build-cuda benchmarks/audit/new-run',
'bash benchmarks/round2/profile_graph.sh build-cuda/bench_decode build-cuda/fixtures/perf.bin benchmarks/audit/new-run/nsys f16',
'python3 benchmarks/round2/analyze_nsys.py benchmarks/audit/new-run/nsys/decode-graph.sqlite --plot',
'python3 tools/record_cuda_environment.py build-cuda benchmarks/audit/new-run',
'python3 benchmarks/audit/summarize_audit.py benchmarks/audit/new-run',
'```','',
'`new-run`必须为新目录；各测量与已完成验证按具体状态发布，partial中的未完成项不得计为PASS。`commands.json`、`decode-commands.json`、`validation-commands.json`、`ncu/commands.json`和日志包含本次精确绝对命令；各telemetry保存时序证据。原始NCU/NSys二进制按仓库规则保留本地，公开CSV、哈希和真实数据图。',
'',
'`initial-measurement-state.json`保留Attention/GEMV/RMS与NCU采集时的既有source/binary SHA256；后续只修改Decode计时/诊断后重编译，新的Decode与sanitizer使用最终environment/telemetry中的二进制身份；NSys使用`sequential-stage-environment.json`对应的二进制（计算捕获路径未改动）。不得用后续二进制哈希冒充此前采集身份。environment.json为16:53归档快照，不因后续汇总脚本修改而回写；后续CPU检查与文档身份另存publication-state.json。资源保护逻辑在卡顿后才补充，不能声称此前采集已经有主存阈值保护。',
'','### 桌面卡顿与资源约束','',
'两次16层整步Racecheck没有写出完成摘要，后续检测到系统重新启动。最后可用遥测显示GPU占用100%、约749MiB/8188MiB显存，不能解释为显存耗尽。16:49之后检查可用主存约10GiB、Swap为0；但这些是重启后的快照，不能反证卡顿时主存未曾紧张。之前没有主存峰值记录；现有内核日志未查到OOM或NVIDIA Xid，卡顿根因尚未证实。',
'',
'完整性能循环包含大量预热、flush、eager/Graph重复，放进Racecheck给桌面GPU带来不必要的持续满载；这是验证流程安排不当。'+('新增`--validate-decode MODEL DT`独立正确性入口：不执行预热、计时或flush，保留原模型、计算/CPU greedy/GPU greedy三种范围、两条传输Graph、5个动态位置及全部物理KV校验。原5样本入口执行164次完整前向，新入口执行16次；没有过滤Kernel、缩小模型、放宽容差或发布插桩耗时为性能。旧性能循环的Racecheck仍未完成，当前关闭的是相同完整计算链路的正确性验收。' if large_race_pass else '额外Racecheck仍未完成，不重试旧性能循环。'),
'',
'guard记录MemAvailable/Swap/内存PSI、目标进程RSS/Swap与cgroup实际限制；默认至少预留3GiB主存、1GiB显存，单子进程超过180秒即终止自身进程组，sanitizer以nice+10运行。'+('本次首次正确性重试因主机总Swap超过原512MiB阈值被守卫主动终止（仍有约10GiB可用内存，目标VmSwap=0），完整记录保留在`racecheck-resolution-guard-aborted/`。随后把验证放入独立systemd用户scope，实际施加MemoryMax=6GiB、MemorySwapMax=0；主机总Swap上限改为1GiB，Racecheck只用1个分析线程。成功日志、命令、资源时序和新验证二进制身份保存在`racecheck-resolution/`；既有测量binary/environment与全部性能CSV保持不变。' if resolution else '资源保护不能保证抢占单个长GPU Kernel，故不能代替避免高开销重复。'),
*(['',f'正确性验证实测：Racecheck总wall {f(resolution["validation_wall_seconds"]["racecheck"])}秒（不属于性能指标），成功阶段观测到的cgroup内存峰值{f(resolution["observed_cgroup_memory_peak_MiB"])} MiB、job Swap=0、最小主机MemAvailable {f(resolution["min_MemAvailable_MiB"])} MiB；cgroup未发生OOM kill。Memcheck另行完成且零错误。所有实测限制均从本次telemetry核对，不能追溯声称旧中断阶段也受这些限制。', '', '本次闭环精确命令（其余15项不重跑）：', '', '```sh', 'cmake -S . -B build-cuda-validation -DNANO_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89', 'cmake --build build-cuda-validation --target bench_decode -j2', f'python3 benchmarks/audit/finish_validation.py build-cuda-validation {rel} --model build-cuda/fixtures/audit16.bin', '```', '', '`finish_validation.py`仅接受partial状态并拒绝覆盖既有resolution目录；当前正式目录已complete，重复验证应选择新目录。单独复测Racecheck可使用下列命令；日志文件名应为新路径：', '', '```sh', 'NV_BENCH_MAX_SWAP_MIB=1024 python3 benchmarks/audit/guard_command.py benchmarks/audit/check-telemetry.jsonl -- systemd-run --user --scope --quiet -p MemoryMax=6G -p MemorySwapMax=0 nice -n 10 compute-sanitizer --tool racecheck --racecheck-num-workers 1 --error-exitcode 99 --log-file benchmarks/audit/check-racecheck.log build-cuda-validation/bench_decode --validate-decode build-cuda/fixtures/audit16.bin f16', '```'] if resolution else []),
'','## 2. 简历改写版（中文，4条）','']
f16=att('f16','cold','split_online_vec4');naive=att('f16','cold','serial_online');vec=att('f16','cold','unsplit_vec4');r=nmetric('attention-f16-cold-m1');rmsb=op('f32','rmsnorm','cold','baseline');rmso=op('f32','rmsnorm','cold','optimized');gemv=op('f32','sgemv','cold','optimized');lib=op('f32','sgemv','cold','cublasSgemv');gm=nmetric('gemv-f32-m1')
resume=[f'针对长序列Decode的串行与并行度瓶颈，实现64-token分段Online Softmax及max/sumexp归约合并；RTX4060 Laptop上FP16 B2/H16/KV4/D128、S32768（KV约128MiB），冷L2 P50 {f(naive["p50_us"])}→{f(f16["p50_us"])} µs，较自有未分段基线{f(number(naive["p50_us"])/number(f16["p50_us"]))}×，向量未分段消融{f(number(vec["p50_us"])/number(f16["p50_us"]))}×；分段Kernel NCU DRAM {f(r["dram_pct"])}%、Occupancy {f(r["occupancy_pct"])}%。',
f'针对RMSNorm重复访存，实现向量读写与输入寄存器驻留；FP32 128×4096、每调用前驱逐128MiB，对比同256-thread行归约标量基线，P50 {f(rmsb["p50_us"])}→{f(rmso["p50_us"])} µs（{f(number(rmsb["p50_us"])/number(rmso["p50_us"]))}×）；GEMV FP32 8192×4096/B1冷NCU达{f(gm["dram_GiB_s"])} GiB/s、DRAM {f(gm["dram_pct"])}%，另测冷缓存P50 {f(gemv["p50_us"])} µs，cuBLAS {f(lib["p50_us"])} µs，明确性能边界。',
f'针对B1 Decode的Host提交开销，利用固定地址与动态长度/页表复用CUDA Graph，并在benchmark内捕获元数据H2D、GPU greedy与Token ID回传；16层d1024随机模型FP16、冷L2完整步P50 {f(e["p50_wall_us"])}→{f(g["p50_wall_us"])} µs（降低{f(pct(e,g,"p50_wall_us"))}%），小模型FP16 NSys证实每步51次Kernel提交→1次Graph提交。',
'针对分页与低精度正确性，覆盖FP32/FP16/BF16、GQA/MQA、乱序页、尾部及动态Graph更新；独立CPU FP64逐输出/逐物理KV槽验收容差内零失败，算子及小模型Memcheck/Racecheck零错误，16层FP16校验1677万KV槽并通过'+('独立正确性入口的Memcheck/Racecheck，16/16项验收完成。' if large_race_pass else 'Memcheck。')+('另披露16层BF16深层数值容差限制。' if bf16_failed else '')]
text += ['- '+x for x in resume]+['','## 3. README补强清单','',
'- [x] 以当前cold/hot表替换混合口径绝对延迟，写明128MiB驱逐位于计时外；硬件精确L2字节、dtype与实际长度。',
'- [x] Attention三路Baseline、FP16/BF16超L2、P50/P95、有效GiB/s与partition/merge真实DRAM/Occupancy分列。',
'- [x] GEMV同精度cuBLAS对照，不声称跑赢；RMS两个版本准确命名，热/冷区分。',
'- [x] 小/16层FP16模型的compute与完整CPU/GPU greedy范围、交错采样、wall/device时间及随机权重限制。',
'- [x] 同精度FP16 NSys原始报告、SQLite、官方统计及51→1真实时间线；说明GPU Kernel数量不变。',
'- [x] 固定seed、完整命令、source/binary/model哈希、进程守卫、sanitizer与CPU FP64误差日志。',
'- [x] 保留受干扰/失败/顺序计时记录并排除，明确16层BF16失败，不更改容差或隐去反例。',
('- [x] 16层FP16独立正确性入口完整Racecheck零错误/零警告；16/16项矩阵闭环，保留旧性能循环中断与资源守卫终止记录。' if large_race_pass else '- [ ] 额外16层整步Racecheck未完成；不作为通过证据。'),
'']
report='\n'.join(text);(out/'REPORT.md').write_text(report)
if a.publish:
    (root/'docs/PERFORMANCE_AUDIT_20261126.md').write_text(report)
    (root/'docs/RESUME_CUDA_AUDITED.md').write_text('# CUDA项目简历表述（2026-11-26实测）\n\n'+ '\n\n'.join('- '+x for x in resume)+'\n\n完整口径与复现：[评测审计](PERFORMANCE_AUDIT_20261126.md)。\n')
    main=root/'README.md';readme=main.read_text()
    start=readme.index('## 历史实测与当前审计结论') if '## 历史实测与当前审计结论' in readme else readme.index('## 冷L2实测与验收')
    end=readme.index('## 构建与一键复现',start)
    section=['## 冷L2实测与验收','',
    'RTX4060 Laptop，24 SM/CC8.9，L2 **33,554,432 bytes（32 MiB）**；Driver615.71.09、CUDA13.4、cuBLAS13.8。未锁频，进程/频率/温度快照随结果保存；测量期间无其他CUDA计算任务。',
    '', '每次算子调用前同stream驱逐128MiB，flush不计时。以下来自本轮100样本真实实测；P50/P95与完整Baseline见[审计报告](docs/PERFORMANCE_AUDIT_20261126.md)。', '']
    ar=[]
    for dt in ['f32','f16','bf16']:
        baseline,optimized=(att(dt,'cold',i) for i in ['serial_online','split_online_vec4'])
        ar.append([dt,'B1/H16/KV4/D128/S4096' if dt=='f32' else 'B2/H16/KV4/D128/S32768',f(baseline['p50_us']),f(optimized['p50_us']),f(number(baseline['p50_us'])/number(optimized['p50_us']))+'×'])
    section+=table(['Attention精度','形状','未分段P50 µs','分段P50 µs','加速比'],ar)
    section += [f'FP32原形状热缓存P50 {f(att("f32","hot","split_online_vec4")["p50_us"])} µs，驱逐后{f(att("f32","cold","split_online_vec4")["p50_us"])} µs；NCU热DRAM读取0、冷约16MiB，确认L2影响。低精度B2实际长度[32768,32765]，唯一KV约127.994MiB。基线是自有每head一warp未分段实现；新增向量未分段消融与分段核分别对照，不能称为成熟库比较。', '']
    section+=table(['冷缓存对照','Baseline P50 µs','优化 P50 µs','结论'],[
        ['RMSNorm FP32 128×4096',f(rmsb['p50_us']),f(rmso['p50_us']),f(number(rmsb['p50_us'])/number(rmso['p50_us']))+'×；标量重读X→向量/寄存器驻留'],
        ['GEMV FP32 8192×4096/B1，W128MiB',f(lib['p50_us'])+'（cuBLAS）',f(gemv['p50_us']),'本实现仍慢于cuBLAS，不声称超越'],
        ['16层d1024随机模型FP16完整GPU greedy步',f(e['p50_wall_us']),f(g['p50_wall_us']),f(pct(e,g,'p50_wall_us'))+'%降低；含metadata H2D/Token回传']])
    section+=['完整步固定B1/seq129，不含CPU页管理/调度或文本输出，非训练模型生成TPS。Graph对照双方预热20次，100组成对交错、交替先后采样。', '',
    f'独立NCU：FP16 Attention partition DRAM {f(r["dram_pct"])}%、{f(r["dram_GiB_s"])} GiB/s、Occupancy {f(r["occupancy_pct"])}%；FP32 GEMV DRAM {f(gm["dram_pct"])}%、{f(gm["dram_GiB_s"])} GiB/s。有效带宽与真实DRAM带宽分开，partition不能代表整条Attention。', '',
    f'8/8 CTest，{len(validation)}/16项Memcheck/Racecheck矩阵零错误；'+('16层FP16使用完整模型的独立正确性入口完成Racecheck，保留5步动态Graph和1677万KV槽校验。' if large_race_pass else '16层FP16普通数值与Memcheck通过，额外整步Racecheck未完成。')+'CPU FP64逐输出/逐物理KV槽在已验收范围容差内零失败。'+('**16层BF16动态前向超出既定容差，已保留失败并排除性能结论；没有放宽容差。** ' if bf16_failed else '16层BF16未纳入当前正式性能矩阵。')+'数值零失败与浮点零误差分开表述。', '',
    'FP16 NSys真实SQLite：每模式5步，255次Kernel提交→5次Graph提交，即每步51→1，GPU仍执行相同51个Kernel。trace是重复计算步，只用于Launch归因。', '',
    f'![FP16 NSys实际时间线]({rel}/nsys/decode-timeline.png)', '',
    '精确构建/重测/NCU/NSys/sanitizer命令、缓存说明及原始证据见[评测入口](benchmarks/audit/README.md)，[4条简历改写](docs/RESUME_CUDA_AUDITED.md)。', '']
    readme=readme[:start]+'\n'.join(section)+'\n'+readme[end:]
    readme=readme.replace('新增评测代码已编译，独占 GPU 的性能与 sanitizer 验收待跑；受并发 VLLM 干扰的结果已拒收。','已完成独占GPU冷缓存重测与15项sanitizer验证；额外16层整步Racecheck未完成，受干扰/失败记录保留并排除。')
    if large_race_pass:readme=readme.replace('已完成独占GPU冷缓存重测与15项sanitizer验证；额外16层整步Racecheck未完成，受干扰/失败记录保留并排除。','已完成独占GPU冷缓存重测与16/16项sanitizer矩阵；16层FP16完整链路采用独立正确性入口完成Racecheck，旧中断/失败记录保留并排除。')
    main.write_text(readme)
    historical=root/'docs/CUDA_VALIDATION.md';body=historical.read_text()
    body=body.replace('73.73µs为热缓存；新增代码尚未重新完成GPU验收。','73.73µs为历史热缓存；新增正式性能已重测，15项sanitizer通过，额外16层Racecheck未完成；BF16限制见最新报告。')
    historical.write_text(body)
    entry=root/'benchmarks/audit/README.md';body=entry.read_text()
    body=body.replace('状态：GPU释放后已完成冷/热Attention、低精度超L2与NCU采集；正在完成交错Decode与sanitizer最终验收。','状态：冷/热Attention、低精度超L2、NCU、交错FP16 Decode、8/8 CTest及15份Memcheck/Racecheck已验收。额外16层整步Racecheck未完成，不自动重试。')
    entry.write_text(body)

# Hash all evidence, including local native reports, without including this manifest.
evidence=[f for f in out.rglob('*') if f.is_file() and f.name!='evidence-sha256.json' and 'tmp' not in f.parts]
(out/'evidence-sha256.json').write_text(json.dumps({str(f.relative_to(root)):sha(f) for f in sorted(evidence)},indent=2)+'\n')
print(f'PASS measured tables: 18 attention, 30 operators, 28 NCU, 12 decoder rows; {len(validation)} sanitizer logs, large_full_racecheck_passed={large_race_pass}; {out}/REPORT.md')
