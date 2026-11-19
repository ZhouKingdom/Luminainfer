"""Isolate BF16 all-reduce shape sensitivity with identical partial sums."""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,world,port,output):
    torch.cuda.set_device(rank);torch.set_num_threads(1)
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=world,
        timeout=timedelta(minutes=2))
    try:
        torch.manual_seed(719+rank)
        rows=[]
        for count in (2,4,8,16):
            source=torch.randn(count,5120,device='cuda',dtype=torch.bfloat16)
            whole=source.clone();dist.all_reduce(whole)
            separate=source.clone()
            for index in range(count):dist.all_reduce(separate[index:index+1])
            delta=(whole.float()-separate.float()).abs()
            rows.append({'rows':count,'elements':source.numel(),'different':int((delta!=0).sum()),'max_abs':delta.max().item()})
        if rank==0:Path(output).write_text(json.dumps({'tp':world,'dtype':'bfloat16','same_input_partials':True,'comparisons':rows},indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tp',type=int,default=4)
    p.add_argument('--port',type=int,default=2465);p.add_argument('--output',required=True);args=p.parse_args()
    mp.spawn(worker,args=(args.tp,args.port,args.output),nprocs=args.tp,join=True)
