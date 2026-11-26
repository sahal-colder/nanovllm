#!/usr/bin/env python3
"""Verify complete measured grids and generate an evidence-backed report."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re

def read(p):return list(csv.DictReader(p.open()))
p=argparse.ArgumentParser(description=__doc__);p.add_argument('results',type=Path);p.add_argument('--require-profilers',action='store_true');p.add_argument('--publish',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=a.results.resolve()
attention=read(out/'attention.csv');decoder=read(out/'decode.csv')
expected={(dt,str(b),'16',str(kv),'128',str(seq),mode) for dt in ['f32','f16','bf16'] for b in [1,4] for kv in [16,4,1] for seq in [128,1024,1037,4096] for mode in ['serial_online','split_online_vec4']}
actual=set()
for r in attention:
    key=tuple(r[k] for k in ['dtype','batch','heads','kv_heads','head_dim','seq_len','implementation']);assert key not in actual;actual.add(key)
    batch,heads,kv,d,seq=(int(r[k]) for k in ['batch','heads','kv_heads','head_dim','seq_len'])
    size=4 if r['dtype']=='f32' else 2
    logical_bytes=size*(2*sum(seq-3*b for b in range(batch))*kv*d+2*batch*heads*d)
    assert int(r['ideal_bytes'])==logical_bytes
    p50,p95,bw=(float(r[k]) for k in ['p50_us','p95_us','effective_GiB_s'])
    assert 0<p50<=p95 and int(r['samples'])==300
    assert math.isclose(bw,logical_bytes/(p50*1.e-6)/2**30,rel_tol=3e-6)
    assert all(math.isfinite(float(r[k])) for k in ['max_abs_err','max_rel_err'])
assert actual==expected and len(attention)==144
assert {(r['dtype'],r['mode']) for r in decoder}=={(d,m) for d in ['f32','f16','bf16'] for m in ['eager','graph']}
decode=[]
for dt in ['f32','f16','bf16']:
    eager=next(r for r in decoder if r['dtype']==dt and r['mode']=='eager');graph=next(r for r in decoder if r['dtype']==dt and r['mode']=='graph')
    for r in [eager,graph]:
        assert int(r['samples'])==100 and int(r['graph_nodes'])==52
        assert 0<float(r['p50_wall_us'])<=float(r['p95_wall_us'])
    decode.append(dict(dtype=dt,eager=eager,graph=graph,speedup=float(eager['p50_wall_us'])/float(graph['p50_wall_us']),reduction_percent=100*(1-float(graph['p50_wall_us'])/float(eager['p50_wall_us']))))
assert '100% tests passed, 0 tests failed out of 8' in (out/'ctest.log').read_text()
for file in ['memcheck-final.log','gemv-rms-memcheck.log']:assert 'ERROR SUMMARY: 0 errors' in (out/file).read_text()
for file in ['racecheck-final.log','gemv-rms-racecheck.log']:assert '0 errors, 0 warnings' in (out/file).read_text()
ncu=[]
for directory in ['operators-ncu','attention-ncu']:
    folder=out/directory
    if not (folder/'status.csv').exists():continue
    statuses=read(folder/'status.csv');assert all(s['status']=='captured' for s in statuses)
    for status in statuses:
        report=folder/status['report'];assert report.is_file()
        raw=list(csv.reader((folder/status['csv']).open()));header,units=raw[:2]
        for values in raw[2:]:
            def metric(key,unit=None):
                idx=header.index(key)
                if unit:assert units[idx]==unit
                return float(values[idx].replace(',','')),units[idx]
            bandwidth,unit=metric('dram__bytes.sum.per_second');factor={'byte/s':1,'Kbyte/s':1e3,'Mbyte/s':1e6,'Gbyte/s':1e9,'Tbyte/s':1e12}[unit]
            name=values[header.index('Kernel Name')]
            kind='attention_partition' if 'partition_kernel' in name else 'attention_merge' if 'merge_kernel' in name else 'sgemv' if 'gemv' in name else 'rmsnorm'
            ncu.append(dict(operator=kind,dtype=status['dtype'],dram_pct=metric('dram__throughput.avg.pct_of_peak_sustained_elapsed','%')[0],
                            dram_GiB_s=bandwidth*factor/2**30,achieved_occupancy_pct=metric('sm__warps_active.avg.pct_of_peak_sustained_active','%')[0],
                            kernel=name,raw_csv=os.path.relpath(folder/status['csv'],root),report=os.path.relpath(report,root)))
if a.require_profilers:assert len(ncu)==12
if ncu:
    with (out/'ncu-metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ncu[0]));w.writeheader();w.writerows(ncu)
timeline=json.loads((out/'nsys/timeline-summary.json').read_text()) if (out/'nsys/timeline-summary.json').exists() else None
if a.require_profilers:assert timeline and timeline['host_launches_per_step']=={'eager':51,'graph':1}
pretrained=json.loads((out/'pretrained/validation.json').read_text()) if (out/'pretrained/validation.json').exists() else None
sample=[r for r in attention if r['dtype']=='f32' and r['batch']=='1' and r['kv_heads']=='4' and r['seq_len']=='4096']
summary=dict(attention_rows=144,attention_cases=72,attention_example=sample,decode=decode,ncu=ncu,nsys=timeline,pretrained=pretrained,
             validation_log=(out/'correctness-final.log').read_text())
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
text=['# CUDA 第二轮阶段验收', '',
      '实现仓库为 nano-vLLM-C（`~/nanovllm`），按用户后续指示复用其 C11 模型/调度/采样框架。',
      'MiniInfer-C 保留第一轮成果；本轮未导入其存储层。KV-Fabric 仅更新简历与面试文档。', '',
      '## 工程与正确性', '',
      '- C11 Host，CUDA 设备函数通过 C ABI 暴露；无新增 class/RAII/vtable/无意义 ctx。',
      '- 完整链路：Embedding → RMSNorm → QKV SGEMV → 邻接对 RoPE / 分页 KV 写入 →',
      '  Paged Attention → O SGEMV / 残差 → RMSNorm → Gate/Up SGEMV → SwiGLU →',
      '  Down SGEMV / 残差 → Final RMSNorm → Classifier SGEMV / CPU 采样。',
      '- 8/8 CTest 通过（两项为 fixture 生成）；Memcheck/Racecheck 对旧算子及新增链路均通过。',
      '- Attention 90 项随机/边界用例：主矩阵 seq 128/1024/1037/4096、GQA 比 1/4/8、',
      '  batch 1/3、三种 dtype；另含 D=1/31/65/127/128/256、空行、非对齐、极端 logits、',
      '  NaN 投毒未满页尾、workspace/输出哨兵和非法设备元数据。',
      '- CPU 参考用 FP64 两遍 Softmax，与 GPU Online Softmax 独立实现；每个输出元素检查。',
      '- 单层 batch=1、三层 batch=2，三种 dtype；分别 21 / 83 步，逐 logits 与全部物理 KV',
      '  槽对照独立 CPU FP64 前向，并验证同一 Graph 跨页、跨 64-token 分段读取新长度/页表。',
      '- Graph 节点逐项检查只含 Kernel/Memset，没有 Host、内存分配或 Host 数据复制节点。',
      '- 原有调度器在 eager/Graph、普通/紧张页池下均匹配 CPU greedy Token ID，覆盖抢占重算。']
if pretrained:text+=['- 真实 TinyStories 260K checkpoint：4 个 prompt × 32 = 128 个生成 Token，CPU/eager/Graph',
                     '  FP32 greedy ID 完全一致；模型和 tokenizer SHA256 经固定来源校验。']
text+=['', '## Paged Attention 代表性实测', '',
       'RTX 4060 Laptop；FP32，B=1，H=16，KV=4，D=128，seq=4096。',
       '随机输入和乱序页，warmup 10 次；60 次 Graph replay × 5 次独立 event 调用，300 样本。',
       '对照是本项目串行 Warp 参考，不是 FlashAttention/cuDNN 等成熟库。', '',
       '| 实现 | P50/P95 µs | 有效 GiB/s | max abs / max rel |', '| --- | ---: | ---: | ---: |']
for r in sample:text.append(f'| {r["implementation"]} | {float(r["p50_us"]):.2f}/{float(r["p95_us"]):.2f} | {float(r["effective_GiB_s"]):.2f} | {float(r["max_abs_err"]):.3g} / {float(r["max_rel_err"]):.3g} |')
text+=['', '完整 144 行覆盖 B=1/4、H=16、KV=16/4/1、D=128、seq=128/1024/1037/4096、三种 dtype。',
       '有效字节数采用唯一 KV + Q/O：`dtype_bytes*(2*sum(lengths)*KV*D+2*B*H*D)`。',
       'GQA 复读、分段 scratch 与元数据不膨胀分子；这是算法有效带宽，不是 DRAM 计数。',
       '主矩阵 cache_policy=' + attention[0].get('cache_policy','hot (historical, no explicit flush)') + '；冷缓存每次调用前驱逐，flush 不计入延迟。', '',
       '## 完整 Decode 与 Graph', '',
       '随机权重 4 层、hidden=512、FFN=1536、H=8、KV=2、vocab=512、seq=129、B=1；',
       '包含 embedding、全部层和最终 logits。输入/长度固定的重复单步，初始 KV 为随机数据。',
       'warmup 20 次/mode、100 个样本；Host 单调时钟覆盖提交至完成等待，另记录 CUDA event。',
       '以下计算区间不含 Host 元数据 H2D、logits D2H、采样、模型加载与 Graph 实例化。', '',
       '| dtype | eager P50/P95 µs | Graph P50/P95 µs | P50 降低 |', '| --- | ---: | ---: | ---: |']
for r in decode:
    e,g=r['eager'],r['graph'];text.append(f'| {r["dtype"]} | {float(e["p50_wall_us"]):.2f}/{float(e["p95_wall_us"]):.2f} | {float(g["p50_wall_us"]):.2f}/{float(g["p95_wall_us"]):.2f} | {r["reduction_percent"]:.2f}% |')
if timeline:
    e,g=timeline['eager'],timeline['graph']
    text += ['', f'Nsight Systems 每模式 5 步：eager {e["kernel_launch_api_calls"]} 次 Kernel Launch，',
             f'Graph {g["graph_launch_api_calls"]} 次 Graph Launch；两者均执行 {e["gpu_kernels"]} 个 GPU Kernel，',
             f'即每步 Host 提交 51 → 1，GPU Kernel 数未减少。GPU 活动窗口内间隙比例',
             f'{e["gpu_gap_percent"]:.2f}% → {g["gpu_gap_percent"]:.2f}%，并未消除所有设备调度间隙。',
             '此统计含 profiler/event 影响，只用于归因；正式延迟使用未插桩的 100 样本 CSV。',
             '原始 `.nsys-rep`、SQLite、官方统计及真实数据绘制的 Timeline 均归档。']
text += ['', '## NCU 硬件对照', '',
         'SGEMV：8192×4096/B1；RMSNorm：128×4096；Attention：B1/H16/KV4/D128/S4096。',
         '每 Kernel 一次采样，cache flush、未锁时钟；利用率由硬件计数读取，非标称带宽反推。', '',
         '| Kernel | dtype | DRAM % | 实测 GiB/s | Achieved Occupancy % |', '| --- | --- | ---: | ---: | ---: |']
for r in ncu:text.append(f'| {r["operator"]} | {r["dtype"]} | {r["dram_pct"]:.2f} | {r["dram_GiB_s"]:.2f} | {r["achieved_occupancy_pct"]:.2f} |')
if not ncu:text+=['', '本次未执行 NCU 采集；未提供硬件计数器验收结论。']
text += ['', 'Attention 的分段和合并分别列指标。合并仅 16 个 CTA，低 Occupancy 是当前小头数形状的',
         '限制，不能将分段指标称作整条 Attention 的利用率。当前实现仍有 SFU、归约与调度开销。', '',
         '## 误差与适用边界', '',
         '- Attention：FP32 atol/rtol=2e-5/3e-4，FP16=1e-3/2e-3，BF16=8e-3/1.6e-2。',
         '- 完整前向：FP32=1e-4/3e-4，FP16=3e-3/8e-3，BF16=2e-2/5e-2。每个元素满足',
         '  `abs_err <= atol + rtol*abs(ref)`；同时保留真实 max_abs / max_rel。近零参考值的相对误差',
         '  可能很大，不能脱离绝对误差解读；FP16/BF16 不承诺 CPU bitwise 或 Token ID 一致。',
         '- SGEMV 适用低 batch Decode；高 batch 推荐可复用权重的 GEMM。原调度器每请求串行',
         '  GPU forward，并非 GPU continuous-batching GEMM；独立 Kernel 测试覆盖真正 B>1。',
         '- 只支持 legacy llama2.c FP32 checkpoint（加载后可转 FP16/BF16）和邻接对 RoPE。',
         '  未实现 Qwen/safetensors/GGUF 加载、量化、Tensor Parallel、CUDA Prefill 优化或生产服务。',
         '- Graph 的 batch、模型形状、页池容量与设备地址固定；token/seq_len/Block Table 内容动态。',
         '  分配、页引用调度和 pinned 元数据上传在捕获外，单 stream 依赖及上传事件保护 staging。',
         '- 本轮未对接 KV-Fabric 远程/分层存储；两项目分别负责计算执行与存储传输。', '',
         '## 复现与证据', '',
         '构建与一键验证命令见仓库 README；全部本轮结果位于 `benchmarks/operators/round2-results/`。',
         '原始 benchmark CSV、NCU `.ncu-repz`/CSV、Nsys `.nsys-rep`/SQLite、sanitizer 日志和',
         'checkpoint 哈希保留；environment.json 提供源文件与计时二进制 SHA256。', '']
report=out/'REPORT.md';report.write_text('\n'.join(text))
if a.publish:(root/'docs/CUDA_VALIDATION.md').write_text('\n'.join(text))
print('Verified 144 attention rows, 6 decoder rows,',len(ncu),'hardware metric rows; report:',report)
