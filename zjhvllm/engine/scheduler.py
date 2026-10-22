# =============================================================================
# 文件功能：调度器（Scheduler）
# 核心职责：
#   1. 管理等待（WAITING）和运行（RUNNING）两个队列，负责序列的生命周期流转。
#   2. 执行两阶段调度：先调度 Prefill（预填充），再调度 Decode（逐字生成）。
#   3. 结合 BlockManager（显存管理器）处理显存分配、释放以及抢占（Preemption）。
#   4. 实现分块预填充（Chunked Prefill）和抢占式调度，保证显存安全。
# =============================================================================

from collections import deque
from time import perf_counter

from zjhvllm.config import Config
from zjhvllm.engine.sequence import Sequence, SequenceStatus
from zjhvllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        """初始化调度器，从配置中读取关键参数并创建块管理器"""
        self.max_num_seqs = config.max_num_seqs          # 单次 Batch 允许的最大序列数量
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 单次 Step 允许处理的最大 Token 总数
        self.eos = config.eos                            # 结束符（End-of-Sequence）的 Token ID
        self.block_size = config.kvcache_block_size      # 逻辑块大小（每个块含多少个 Token）
        # 初始化显存块管理器，负责物理块的分配、释放和缓存共享
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        
        # 等待队列（WAITING）：存放尚未调度的序列（包括新请求和被抢占的序列）
        self.waiting: deque[Sequence] = deque()
        # 运行队列（RUNNING）：存放当前正在参与推理的序列
        self.running: deque[Sequence] = deque()

        # 交替调度旗标：当 running 与 waiting 同时非空时，
        # 控制 Decode 与 Prefill 轮流执行（False = 本轮优先 Decode，初始为 Decode 优先）
        self._prefill_turn = False

    def is_finished(self):
        """检查是否所有序列都已处理完毕（用于退出主循环）"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """向调度器提交一个新序列，追加到等待队列末尾"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度函数，决定本轮需要送入模型执行的序列列表。
        :return: (scheduled_seqs, is_prefill)
                 scheduled_seqs: 本轮调度的序列列表
                 is_prefill: True 表示本轮执行 Prefill 阶段，False 表示执行 Decode 阶段

        调度顺序（与 vLLM 一致）：Decode 优先，Prefill 次之。
          1. 先调度 RUNNING 队列（Decode），保证已投入的序列持续产出 Token，
             避免新到的 Prefill 请求无限期推迟已有请求的输出（饿死）。
          2. 再调度 WAITING 队列（Prefill，支持 Chunked Prefill）。
        当 running 与 waiting 同时非空时，二者交替执行（每轮切换一次），
        保证 Decode 优先的同时，WAITING 中的新请求也不会被饿死。
        """
        seqs: list[Sequence] = []

        # ---- 交替调度：若 running 与 waiting 并存且本轮轮到 Prefill，先做 Prefill ----
        if self.waiting and self.running and self._prefill_turn:
            seqs = self._schedule_prefills()
            if seqs:
                self._prefill_turn = False
                return seqs, True
            # Prefill 无序列可调度（显存不足），回退到 Decode
            self._prefill_turn = True

        # ---- 阶段一：Decode（优先处理 running 队列）----
        if self.running:
            seqs = self._schedule_decodes()
            if seqs:
                self._prefill_turn = True
                return seqs, False

        # ---- 阶段二：Prefill（处理 waiting 队列，支持 Chunked Prefill）----
        if self.waiting:
            seqs = self._schedule_prefills()
            if seqs:
                self._prefill_turn = False
                return seqs, True

        # 两队列均无可调度的序列：要么全部完成（正常，主循环据此退出），
        # 要么是 KV Cache 显存不足导致无法分配（异常，明确报错而非 assert 崩溃）。
        if not self.waiting and not self.running:
            return [], True
        raise RuntimeError(
            "Scheduler: 没有可调度的序列（KV Cache 显存不足，无法容纳任何请求），"
            "请增大 num_kvcache_blocks 或减小请求规模"
        )

    def _schedule_prefills(self) -> list[Sequence]:
        """
        Prefill（预填充）调度：处理 waiting 队列，支持 Chunked Prefill（分块预填充）。
        关键点：
          - 每个序列本轮实际处理的 Token 数 = min(待处理数, 剩余 Token 预算)。
          - 允许"任意多个"序列被截断（分块预填充），以最大化 Token 预算利用率，
            而不是只允许第一个序列独占预算（旧实现会浪费预算并放大延迟）。
          - 被截断的序列仍留在 waiting 队列头部，下一轮优先续算剩余部分。
        :return: 本轮调度的序列列表（可能为空）
        """
        scheduled_seqs: list[Sequence] = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]                 # 窥视队首，但不弹出

            # 计算当前 Batch 剩余的 Token 配额
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break                             # 配额已耗尽，无法再塞入更多 Token

            # ---------- 计算本次可以处理多少 Token ----------
            if not seq.block_table:
                # 情况 A：全新序列。先检查显存，返回可复用的缓存块数量
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break                         # 显存不足，停止调度新序列
                # 需要计算的 Token 数量 = 总长度 - 已经缓存的 Token 数
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
                # 正式分配显存块（此时序列尚无块表）
                self.block_manager.allocate(seq, num_cached_blocks)
            else:
                # 情况 B：序列已有块表（Chunked Prefill 续算），计算剩余未缓存 Token 数
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # ---------- Chunked Prefill ----------
            # 本轮实际调度 Token 数 = min(待处理数, 剩余配额)。
            # 配额不足时序列会被"截断"，留在 waiting 头部，下一轮继续处理。
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            # 若本轮已将该序列的全部 Token 算完，预填充完成，转入 RUNNING
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()            # 弹出已完成的序列
                self.running.append(seq)          # 追加到运行队列末尾
            # 若未完成（分块截断），序列保留在 waiting 队列头部，
            # 下一轮 schedule 会继续处理它（case B 分支），直到全部算完。

            scheduled_seqs.append(seq)

        return scheduled_seqs

    def _schedule_decodes(self) -> list[Sequence]:
        """
        Decode（自回归生成）调度：处理 running 队列，每次为每个序列追加 1 个新 Token。
        若空闲块不足，则触发抢占（Preemption），优先抢占运行队列末尾的序列。
        :return: 本轮调度的序列列表（可能为空）
        """
        scheduled_seqs: list[Sequence] = []

        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 从运行队列左侧取出一个序列（FIFO 顺序）
            seq = self.running.popleft()

            # ---------- 尝试为该序列追加一个新 Token（显存检查） ----------
            # 循环检查，如果不能追加（即显存块不够），则触发抢占（Preemption）
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 策略：优先抢占运行队列末尾的序列（Last-In-First-Out 抢占）
                    # 因为末尾的序列最晚被调度，抢占它影响最小（类似 LRU 思想）
                    victim = self.running.pop()
                    self.preempt(victim)
                else:
                    # 如果运行队列只有当前序列了，那就只能抢占自己
                    self.preempt(seq)
                    break  # 抢占自己后，这个序列就不参与本轮调度了，跳出内层循环

            # Python 的 while...else 语法：
            # 如果 while 循环没有被 break 打断（即成功 append），则执行 else 分支
            else:
                # 成功通过显存检查，配置本轮调度信息
                seq.num_scheduled_tokens = 1          # Decode 阶段每次都只生成 1 个新 Token
                seq.is_prefill = False                # 标记为非预填充阶段
                self.block_manager.may_append(seq)    # 如果跨了块边界，分配新块
                scheduled_seqs.append(seq)

        # 将调度到的序列重新放回运行队列的左侧（保持原有相对顺序，且置顶）
        # 这里 reversed 是为了保持 scheduled_seqs 的顺序：原本 popleft 取出 A,B，
        # scheduled_seqs 是 [A, B]，extendleft(reversed) -> 先放 B，再放 A，队列变为 [A, B]。
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))

        return scheduled_seqs

    def preempt(self, seq: Sequence):
        """
        抢占（Preemption）：将一个正在运行的序列踢出 GPU 显存，放回等待队列。
        通常在显存不足时，为了给新序列或当前序列腾出空间而调用。
        """
        seq.status = SequenceStatus.WAITING    # 状态降级为等待
        seq.is_prefill = True                  # 重置为预填充标记（重新进来时需要重算 Prefix）
        self.block_manager.deallocate(seq)     # 释放其占用的所有物理块（递减引用计数）
        self.waiting.appendleft(seq)           # 放回等待队列头部（优先被重新调度）

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        模型前向推理完成后的后处理钩子。
        负责更新序列状态、更新哈希缓存、追加生成的 Token 以及检查结束条件。
        :param seqs: 本轮参与推理的序列列表（顺序与模型输出一一对应）
        :param token_ids: 模型输出的 Token ID 列表（对应每个序列）
        :param is_prefill: 本轮是否为 Prefill 阶段
        """
        stream = []  # 本轮真正追加了 token 的 (seq_id, token_id)，供流式输出
        for seq, token_id in zip(seqs, token_ids):
            # ---------- 1. 更新全局哈希索引（Prefix Caching） ----------
            # 将本次计算出的新 KV Cache 块的哈希指纹注册到全局字典中
            self.block_manager.hash_blocks(seq)

            # ---------- 2. 同步缓存计数 ----------
            # 标记本轮调度的 token 已经成功缓存在显存中
            seq.num_cached_tokens += seq.num_scheduled_tokens
            # 重置本轮调度指令
            seq.num_scheduled_tokens = 0

            # ---------- 3. 判断是否需要生成新 Token ----------
            # 如果处于 Prefill 阶段，且本轮并没有把整个 Prompt 算完（分块截断），
            # 则本轮是不会产生新的 Decode Token 的，直接跳过 append_token。
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue

            # ---------- 4. 追加模型生成的 Token ----------
            seq.append_token(token_id)
            stream.append((seq.seq_id, token_id))
            # 记录每个 completion token 的生成时刻（首个用于 TTFT，相邻差用于 ITL）
            now = perf_counter()
            seq.token_times.append(now)
            if seq.first_token_time is None:
                seq.first_token_time = now
            # 若使用固定种子，则 seed 逐 token 递增，保证可复现且每个位置独立
            if seq.seed is not None:
                seq.last_ran_seed = (seq.last_ran_seed + 1) if seq.last_ran_seed is not None else seq.seed

            # ---------- 5. 检查是否满足结束条件 ----------
            # 条件 A：生成了 EOS 结束符（且未被忽略）
            # 条件 B：生成的 Completion Token 数量达到用户设定的最大值
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED          # 标记为已完成
                seq.finished_time = perf_counter()            # 记录完成时刻
                self.block_manager.deallocate(seq)            # 立即释放显存
                self.running.remove(seq)                      # 从运行队列中移除
        return stream