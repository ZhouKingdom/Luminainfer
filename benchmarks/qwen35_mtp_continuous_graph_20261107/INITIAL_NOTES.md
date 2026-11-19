# 初始记录与比较边界

1. `continuous_unit.log` 中两页容量的初始失败暴露真实调度死锁：部分prefill持页，唯一decoder跨页时先抢占自己，重排后的等待队首无法分配两页、后方partial又持有剩余页。已在scheduler的decode分配失败分支先回收partial prefill，`continuous_fixed.log`及正式162项CUDA回归通过。最初抽取普通step执行器时的缩进错误已在运行前修正，未作为验收结果。
2. `pilot_tp2` 发生在Graph缓冲分配跨rank准备检查补充之前，且Graph模块存在延迟导入；只用于开发试跑，最终结果不用它。`final_tp2`使用最终执行代码，6阶段完整通过。
3. `final_tp4`初始记录：continuous_eager 8/8，cohort_eager 对照连续ordinary仅6/8；请求索引5输出位置11为114733 vs96760，索引7位置7为81347 vs45543。连续/固定cohort的prefill打包和调度形状不同，不能把不同调度模式直接作为相同数值历史。该记录保留为跨调度失败，不能算通过。
4. 后续新增 `cohort_ordinary`：普通generate按相同4请求cohort执行，cohort_eager对照它逐token通过。连续Graph继续严格与连续ordinary及continuous_eager比较，没有降低阈值。记录 `same_as_continuous_ordinary` 保留跨调度差异。
5. `synthetic_tp2/tp4`为早期Graph/动态slots/提交恢复检查；最终分布式验收用 `synthetic_final_tp2/tp4/tp8`，增加单rank Graph元数据分配失败，所有rank回滚、工作进程存活并能继续重放。

正式结果使用 final_tp2、verified_tp4、verified_tp8。各自快照和benchmark源码随结果保存，阶段数量6/7/7的区别来自新增cohort普通控制，不影响连续Graph两轮对照。

6. 首批synthetic在故障注入后留下异常引用环，JSON已通过但进程在通信组/Graph清理时未退出。仅删除Graph属性不足，显式gc回收后可退出；进一步将正式ModelRunner.exit加入gc与销毁NCCL前的跨rank同步。早期进程均按父命令/PID核实后终止，记录在test_cleanup*.json。最终用synthetic_exit_tp*直接调用正式exit，父进程mp.spawn(join=True)返回后才写processes_exited=True；未再仅凭JSON判断退出。正式性能运行时早期小模型留有空闲CUDA上下文，属于共享非隔离测量；不把结果当严格强扩展效率。计时后生产代码仅此退出增强，前向/接受/调度AST与正式快照完全一致。
