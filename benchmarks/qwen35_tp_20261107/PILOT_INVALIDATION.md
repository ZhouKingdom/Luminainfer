# 试跑不作为性能验收

`pilot_tp2`、`pilot_graph_tp2` 的C128在完整有限值检查之前，曾生成大量token 0。根因是TP2递归状态池的slot偏移超过有符号32位元素索引范围；原kernel在slot乘stride后才转换指针，发生地址溢出。

已将读/写slot均先转int64再计算偏移，新增约9GB跨度的in-place/out-of-place回归。原始JSON与日志保留；上述C128吞吐不能当有效结果。所有`pilot_*`仅为调试阶段记录，正式结果使用之后带每步logits有限值检查的新目录。

原生Graph初次单测另发现假runner的target_hidden未发布，已补相同的hidden记录调用。原2 failed / 140 passed日志保留，修复后重跑，未放宽比较。
