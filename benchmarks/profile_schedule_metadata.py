"""Single-model A/B for host metadata; suitable for nsys CUDA/NVTX tracing."""
import json
from pathlib import Path
import torch
from zjhvllm import LLM,SamplingParams


def main():
    model='/home/data/jhzhou/huggingface/Qwen3.5-27B'
    llm=LLM(model,enforce_eager=True,max_num_seqs=16,max_num_batched_tokens=1024,max_model_len=512)
    prompts=[llm.tokenizer.encode(f'Case {i}: Explain inference batching and scheduling. '*30)[:128] for i in range(8)]
    params=SamplingParams(greedy=True,max_tokens=1,ignore_eos=True)
    original=llm.model_runner.run_model
    rows=[]
    def capture(*args,**kw):
        logits=original(*args,**kw)
        rows.append(logits.detach().float().cpu())
        return logits
    llm.model_runner.run_model=capture
    expected=None
    report={}
    try:
        llm.generate(prompts,params,use_tqdm=False)
        rows.clear()
        torch.cuda.profiler.start()
        for enabled in (False,True):
            llm.model_runner.config.cache_schedule_metadata=enabled
            name='host_metadata' if enabled else 'device_scalars'
            rows.clear()
            with torch.cuda.nvtx.range(name):
                output=llm.generate(prompts,params,use_tqdm=False)
                torch.cuda.synchronize()
            if expected is None:
                expected=rows[-1].clone()
            torch.testing.assert_close(rows[-1],expected,atol=0,rtol=0)
            report[name]={'logits_max_abs':(rows[-1]-expected).abs().max().item(),
                          'token_ids':[row['token_ids'] for row in output]}
        torch.cuda.profiler.stop()
        (Path(__file__).resolve().parent/'scheduling_metadata_profile.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        llm.model_runner.run_model=original
        llm.exit()


if __name__=='__main__':main()
