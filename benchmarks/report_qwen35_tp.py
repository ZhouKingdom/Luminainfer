"""Report validated TP runs without mixing E2E and steady decode rates."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args();root=args.directory
    runs={};rows=[]
    for tp in (1,2,4,8):
        directory=root/f'final_tp{tp}'
        x=json.loads((directory/'result.json').read_text())
        assert all(x[k] for k in ('complete','all_resources_reclaimed','lengths_passed',
                                  'same_config_repeat_exact','all_logits_finite'))
        assert all(v['exact']==v['paired'] for v in x['eager_parity'])
        for name,expected in x['source_sha256'].items():
            assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==expected
        runs[tp]=x
        for c in x['args']['concurrency']:
            phases=[p for p in x['phases'] if p['concurrency']==c]
            seconds=sum(p['seconds'] for p in phases)
            steady_seconds=sum(p['steady_seconds'] for p in phases)
            rows.append({'tp':tp,'concurrency':c,'requests':sum(len(p['outputs']) for p in phases),
                'e2e_output_tok_s':sum(p['output_tokens'] for p in phases)/seconds,
                'steady_output_tok_s':sum(p['steady_tokens'] for p in phases)/steady_seconds if steady_seconds else None,
                'max_decode_batch':max(p['max_decode_batch'] for p in phases),
                'state_slots':x['ranks'][0]['state_slots'],
                'parameter_gib_per_rank':x['ranks'][0]['parameter_bytes']/1024**3,
                'graph_replays':sum(p['graph_replays'] for p in phases)})
    comparisons=[]
    for tp in (2,4,8):
        for p in runs[tp]['phases']:
            base=next(q for q in runs[1]['phases'] if q['repeat']==p['repeat'] and q['concurrency']==p['concurrency'])
            comparisons.append({'tp':tp,'repeat':p['repeat'],'concurrency':p['concurrency'],
                'exact':sum(a==b for a,b in zip(p['outputs'],base['outputs'])),'paired':len(p['outputs'])})
    summary={'validated':True,'related_cuda_tests':144,'rows':rows,'graph_eager_parity':{
        str(tp):x['eager_parity'] for tp,x in runs.items()},'all_logits_finite':True,
        'all_resources_reclaimed':True,'repeat_outputs_exact':True,
        'cross_tp_greedy_comparison':comparisons,
        'cross_tp_note':'C128 TP1 has only 49 allocated slots; this mixes TP rounding and prefill scheduling, not a forced-same-history numerical test',
        'source_snapshots_verified':True,'contract':runs[1]['contract']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Qwen3.5-27B TP 容量验收','',
        '8×A100 80GB内不同卡数的真实张量并行，非多副本。固定128输入/128输出、greedy、ignore EOS，原生引擎、非HTTP。',
        '每配置两轮，反转并发顺序；加载/JIT/Graph捕获不计时，每步有限值检查计时。两轮输出完全一致、Graph/eager严格对照全通过、资源完整回收。','',
        '| TP | 入队请求 | 实际最大decode batch | 端到端output tok/s | 完整batch窗口output tok/s | 每rank参数GiB |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        steady=f"{r['steady_output_tok_s']:.2f}" if r['steady_output_tok_s'] is not None else '未达到'
        lines.append(f"| {r['tp']} | {r['concurrency']} | {r['max_decode_batch']} | {r['e2e_output_tok_s']:.2f} | {steady} | {r['parameter_gib_per_rank']:.2f} |")
    lines += ['', '端到端包含prefill/排队；完整batch窗口只取所有请求均已prefill且都未结束的decode步骤，不能当E2E吞吐。TP1当前状态槽配置为49，128入队并非128同时活跃。',
        '本轮瓶颈还包括逐序列GDN prefill、原生Graph外的采样与调度、NCCL和复制LM head，不能从稳态容量推断服务SLA。',
        '144项相关CUDA测试通过；分布式小模型TP2/4/8相对FP32参考max_abs≤3.58e-7。跨TP数的BF16输出不保证逐位一致，原始比较在summary.json保留，未作新的HF严格logits或任务准确率验收。',
        '最初试跑的32位状态地址溢出已修复；pilot数据不作为有效吞吐证据，见PILOT_INVALIDATION.md。',
        '正式每目录含完整输出、各rank局部参数/状态/KV形状及源码SHA256快照。MTP投机仍限定TP1。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
