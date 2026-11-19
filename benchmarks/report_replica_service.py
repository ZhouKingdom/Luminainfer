"""Build auditable tables from saved raw request records."""
import argparse
import csv
import json
from pathlib import Path

from benchmarks.service_metrics import summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    report = json.loads((directory/'summary.json').read_text())
    cfg = report['config']
    baseline = next(p for p in report['phases'] if p['mode']=='closed' and p['replicas']==1 and p['concurrency_per_replica']==8)
    table = []
    for phase in report['phases']:
        rows = [json.loads(line) for line in (directory/(phase['name']+'.jsonl')).read_text().splitlines()]
        assert len(rows)==len({r['id'] for r in rows})
        start = max(r['finished_at'] for r in rows)-phase['wall_seconds']
        checked = summarize(rows,start,phase['replicas'])
        for key in ('output_tokens_per_second','requests_per_second'):
            assert abs(checked[key]-phase[key])<1e-6, (phase['name'], key)
        speedup = phase['output_tokens_per_second']/baseline['output_tokens_per_second']
        comparable = phase['mode']=='closed' and phase['concurrency_per_replica']==8
        row = {'phase':phase['name'],'replicas':phase['replicas'],'total_concurrency':phase['replicas']*phase['concurrency_per_replica'] if phase['mode']=='closed' else 'open-loop',
               'requests':phase['requests'],'output_tok_s':phase['output_tokens_per_second'],
               'requests_s':phase['requests_per_second'], 'ttft_p50_ms':phase['ttft_seconds']['p50']*1000,
               'ttft_p99_ms':phase['ttft_seconds']['p99']*1000,'tpot_p99_ms':phase['tpot_seconds']['p99']*1000,
               'itl_p99_ms':phase['itl_seconds']['p99']*1000,'slo_attainment':phase['slo']['attainment'],
               'slo_goodput_requests_s':phase['slo']['goodput_requests_per_second'],
               'scaling_vs_1gpu':speedup if comparable else '', 'scaling_efficiency':speedup/phase['replicas'] if comparable else ''}
        table.append(row)
        with (directory/(phase['name']+'_latencies.csv')).open('w') as f:
            writer = csv.DictWriter(f,fieldnames=['id','gpu','input_tokens','output_tokens','ttft_seconds','tpot_seconds','e2e_seconds','loadgen_lag_seconds'])
            writer.writeheader()
            for r in rows:
                writer.writerow({**{k:r[k] for k in ['id','gpu','input_tokens','output_tokens']},
                    'ttft_seconds':r['token_times'][0]-r['scheduled_at'],
                    'tpot_seconds':(r['token_times'][-1]-r['token_times'][0])/(len(r['token_times'])-1),
                    'e2e_seconds':r['finished_at']-r['scheduled_at'],
                    'loadgen_lag_seconds':r['submitted_at']-r['scheduled_at']})
    with (directory/'results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    seen = {}
    matches = mismatches = 0
    for phase in report['phases']:
        if phase['mode']!='closed' or phase['concurrency_per_replica']!=8:
            continue
        for line in (directory/(phase['name']+'.jsonl')).read_text().splitlines():
            r = json.loads(line)
            if r['id'] in seen:
                if seen[r['id']]==r['output_sha256']:
                    matches += 1
                else:
                    mismatches += 1
            else:
                seen[r['id']] = r['output_sha256']
    (directory/'output_consistency.json').write_text(json.dumps({'matching_repeated_requests':matches,'mismatching_repeated_requests':mismatches,'scope':'closed-loop, concurrency=8 per replica, identical request IDs across replica counts'},indent=2)+'\n')
    lines=['# Qwen3.5-27B 多副本容量与尾延迟实测','',
           '日期：2026-11-01。A100-SXM4-80GB；每 GPU 独立单卡文本模型，eager、BF16 权重与 FP32 recurrent state。该实验不新增模型并行。','',
           f"固定输入 {cfg['input_tokens']} tokens、输出 {cfg['output_tokens']} tokens；每副本 {cfg['requests_per_replica']} 条请求；greedy、ignore_eos；token budget={cfg['token_budget']}，max_num_seqs={cfg['max_seqs']}。",
           '引擎原生接口计时，未包含 HTTP/SSE、外部网络、tokenization、模型加载和预热。prompt 来自固定句式的自然语言合成文本，不是真实生产 trace。','',
           '## 已完成场景','',
           '| 场景 | GPU 副本 | 总并发 | 请求数 | output tok/s | TTFT p99 | TPOT p99 | ITL p99 | SLO 达标率 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['phase']} | {row['replicas']} | {row['total_concurrency']} | {row['requests']} | {row['output_tok_s']:.1f} | {row['ttft_p99_ms']:.1f}ms | {row['tpot_p99_ms']:.1f}ms | {row['itl_p99_ms']:.1f}ms | {row['slo_attainment']*100:.1f}% |")
    lines+=['','## 扩展性口径','',
            'closed-loop 每副本保持 8 个在途请求，完成后补充新请求；每副本请求数、输入/输出长度相同，这是副本容量的弱扩展测试。按共同起点到最后请求完成的总 makespan 计算聚合吞吐，不相加各副本局部吞吐。它不是单请求 8 卡加速，也不是 TP/EP scaling。','']
    for row in table:
        if row['scaling_vs_1gpu']!='':
            lines.append(f"- {row['replicas']} GPU：{row['scaling_vs_1gpu']:.3f}× 单卡容量，效率 {row['scaling_efficiency']*100:.1f}%。")
    lines+=['','## 延迟与 SLO 定义','',
            f'固定每卡并发 8 的扩展实验中，重复请求输出 hash：{matches} 次相同、{mismatches} 次不同；不代表 HuggingFace 全模型精度验收。',
            '聚合 open-loop 到达序列为 Poisson，按请求序号轮询分发到副本；不把各副本的到达序列当成独立 Poisson。',
            '- TTFT：首 token 时刻减计划到达/提交时刻。open-loop 使用预先生成的 Poisson 计划到达时间，包含引擎忙碌造成的实际提交滞后；该滞后另行记录，不能从延迟中扣掉。',
            '- TPOT：每请求 (最后 token 时刻−首 token 时刻)/(输出 token 数−1)。ITL 为逐 token 间隔，p99 对整个 phase 的间隔池取分位数，间隔并非相互独立样本。',
            '- SLO 是本实验预设门槛 TTFT≤2s 且 TPOT≤100ms，不是已验证的生产 SLA。SLO goodput=同时满足两项门槛的请求数/总 makespan。',
            '- 每副本/小规模档位少于 1000 条请求，p99 标为 exploratory。8 副本有 1024 条请求，仍只有单次实验，没有重复试验置信区间。',
            '- open-loop 采用固定总到达速率，吞吐可能受 arrival rate 限制；不能当作饱和吞吐峰值。',
            '- 各场景分别报告指标；不能把高并发峰值吞吐与低并发低尾延迟拼成同一个场景。','',
            '## 基线限制','',
            '现有 verl 环境为 vLLM 0.11.0，其模型注册表未支持 Qwen3.5，因此本轮没有同模型 vLLM 对照；不使用 Qwen3 的数据代替它，也不宣称相对 vLLM 提升。','',
            '## 原始证据','',
            '- summary.json：运行参数、GPU/状态槽/缓存容量与各场景汇总。',
            '- manifest.json：确定性请求清单；*_r*_*.jsonl：每请求计划/实际提交、完成时刻、全部 token 时间戳与输出 hash。',
            '- results.csv 与 *_latencies.csv：可复核的聚合/逐请求结果。',
            '- source_snapshot/sha256.json 与 environment.json：本次实现快照与依赖版本。','',
            '复现命令见 summary.json 的 config；使用 benchmarks.bench_replica_service。生成本报告：python -m benchmarks.report_replica_service '+str(directory)+'.']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',directory/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
