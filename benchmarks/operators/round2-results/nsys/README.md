# Nsys 公开归档说明

公开分支保留官方统计 CSV、`timeline-summary.json` / CSV，以及从实际 SQLite 数据绘制的
`decode-timeline.png`。这张图不是 Nsight UI 截屏。

原始 `decode-graph.nsys-rep` 与 `decode-graph.sqlite` 仍保留在验收机器本地，
自动审批因其可能包含进程环境或认证材料而拒绝公开上传；两者加入 `.gitignore`。原始 NCU 报告亦保留本地，见[公开归档范围](../../PROFILING_ARTIFACTS.md)。
性能统计及其计时口径没有更改，未生成替代的原始报告。

在自己的机器重新采集：

```sh
bash tools/validate_cuda.sh build-cuda benchmarks/operators/runs/profile-run --profile
python3 benchmarks/round2/analyze_nsys.py benchmarks/operators/runs/profile-run/nsys/decode-graph.sqlite --plot
```

运行脚本会生成新原始报告；NCU 采集需相应性能计数器权限。
