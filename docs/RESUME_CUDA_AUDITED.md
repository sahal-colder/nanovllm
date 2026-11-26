# CUDA项目简历表述（2026-11-26实测）

- 针对长序列Decode的串行与并行度瓶颈，实现64-token分段Online Softmax及max/sumexp归约合并；RTX4060 Laptop上FP16 B2/H16/KV4/D128、S32768（KV约128MiB），冷L2 P50 23314.43→1168.38 µs，较自有未分段基线19.95×，向量未分段消融21.31×；分段Kernel NCU DRAM 63.69%、Occupancy 79.54%。

- 针对RMSNorm重复访存，实现向量读写与输入寄存器驻留；FP32 128×4096、每调用前驱逐128MiB，对比同256-thread行归约标量基线，P50 25.60→18.43 µs（1.39×）；GEMV FP32 8192×4096/B1冷NCU达230.37 GiB/s、DRAM 96.72%，另测冷缓存P50 620.54 µs，cuBLAS 558.08 µs，明确性能边界。

- 针对B1 Decode的Host提交开销，利用固定地址与动态长度/页表复用CUDA Graph，并在benchmark内捕获元数据H2D、GPU greedy与Token ID回传；16层d1024随机模型FP16、冷L2完整步P50 2735.13→2622.72 µs（降低4.11%），小模型FP16 NSys证实每步51次Kernel提交→1次Graph提交。

- 针对分页与低精度正确性，覆盖FP32/FP16/BF16、GQA/MQA、乱序页、尾部及动态Graph更新；独立CPU FP64逐输出/逐物理KV槽验收容差内零失败，算子及小模型Memcheck/Racecheck零错误，16层FP16校验1677万KV槽并通过Memcheck。另披露16层BF16深层数值容差限制。

完整口径与复现：[评测审计](PERFORMANCE_AUDIT_20261126.md)。
