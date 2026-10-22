"""多进程运行 ExpertParallelMoE，rank 0 打印输出（JSON）。

test_ep_moe.py 用它做多卡 EP 验证：所有 rank 输入相同的 x，每卡独立跑 EP forward，
rank 0 打印自己的输出，父进程与单卡 reference 比对。
用法：python _ep_runner.py <world_size> <norm_topk_prob: 0/1>
"""

import json
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from zjhvllm.layers.ep_moe import ExpertParallelMoE

HIDDEN, INTER, N_EXPERTS, N_TOPK, N_TOKENS = 32, 16, 8, 2, 64


def _free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def worker(rank, world_size, port, norm_topk_prob):
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    torch.manual_seed(0)
    moe = ExpertParallelMoE(HIDDEN, INTER, N_EXPERTS, N_TOPK,
                            norm_topk_prob=bool(norm_topk_prob)).cuda()
    x = torch.randn(N_TOKENS, HIDDEN, device="cuda")  # 所有 rank 相同的输入

    with torch.inference_mode():
        out = moe(x)

    if rank == 0:
        print("__RESULT__" + json.dumps(out.cpu().tolist()), flush=True)
    dist.destroy_process_group()


def main():
    world_size = int(sys.argv[1])
    norm_topk_prob = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    port = _free_port()
    mp.spawn(worker, args=(world_size, port, norm_topk_prob), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
