# 公开归档范围

本分支公开 C11/CUDA 实现、测试、基准工具、实测 CSV/JSON、验证日志及 Timeline PNG。
所有性能数值保持原始测量结果；第一轮历史数据与本轮采集仍分别注明出处。

自动审批拒绝将原始 NCU/Nsys 二进制报告与 SQLite 公开上传，原因是其中可能包含环境、
路径或认证相关元数据，而本次授权为代码上传。原始 `.ncu-rep` / `.ncu-repz` /
`.nsys-rep` / `.sqlite` 保留在验收机器本地并加入 `.gitignore`；公开提交不包含它们。
归档中的路径、状态和环境哈希记录的是验收时本地状态，不代表原始文件均在公开副本中。

同样，本地三个简历生成脚本不发布；`environment.json` 中其哈希保留作历史记录，
推理实现、算子、测试和基准工具的源码均已公开。模型与构建产物不纳入版本控制。

使用以下命令在新结果目录生成自己的完整原始报告，再运行分析/汇总工具：

```sh
bash tools/validate_cuda.sh build-cuda benchmarks/operators/runs/profile-run --profile
```

不要对缺少原始文件的公开结果副本运行 `summarize.py --require-profilers`；
该严格验收选项要求本地原始报告存在。复现需 CUDA GPU，NCU 还需性能计数器权限。
