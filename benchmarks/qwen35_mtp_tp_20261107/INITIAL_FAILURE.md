# 初始 TP4 批量验证失败记录

`real_tp4/result.json` 是修复前试跑，C1 serial 4/4，batched 2/4；两条首次差异为归档输入2的输出位置21（332 vs318）、输入76的位置11（12847 vs3545）。资源回收成功，但数值验收失败。不得将此记录当通过结果或合格性能结果。

初版只固定了每行 GEMM，行并行输出仍将整个窗口交给一次 BF16 NCCL all-reduce。后续修复在 decode_equivalent 上也按单 token 形状归约，普通未启用 MTP 的多卡 decode Graph 不受此改动影响。最终验收仅引用修复后的记录。`real_tp2`、`final_tp2` 也属于该修复之前的结果，保留作历史。

初始回归中的 fake runner 缺少 world_size，已使预测同步助手兼容单卡测试替身，原失败日志保留。最终 finite-logits 诊断在 benchmark 模块导入时安装到各 rank，检查开销包含计时。

后续隔离实验 `nccl_shape_tp4.json`：相同rank BF16局部分量，整体[2,5120]与逐行[1,5120]归约的10240个元素中3299个不同、max_abs=0.0625。该实验完全不运行GEMM，直接证实本环境NCCL BF16归约对输入张量形状敏感；最终固定形状后TP2/4/8的48次MTP输出全部通过同配置比较。
