"""Recompute paired, swapped-GPU trials without averaging p99 percentiles."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
from benchmarks.service_metrics import quantiles,summarize


def report(directory):
    source=json.loads((directory/'summary.json').read_text())
    manifest={r['id']:r for r in json.loads((directory/'manifest.json').read_text())}
    grouped=defaultdict(list)
    decisions=Counter(); all_cases=[]
    records={}
    for phase in source['phases']:
        rows=[json.loads(line) for line in (directory/(phase['name']+'.jsonl')).read_text().splitlines()]
        assert len({r['id'] for r in rows})==len(rows)==phase['requests']
        assert all(r['output_tokens']==manifest[r['id']]['max_tokens'] for r in rows)
        assert all(r['input_tokens']==len(manifest[r['id']]['prompt_token_ids']) for r in rows)
        assert all(a<=b for r in rows for a,b in zip(r['token_times'],r['token_times'][1:]))
        start=max(r['finished_at'] for r in rows)-phase['wall_seconds']
        check=summarize(rows,start,phase['replicas'])
        assert abs(check['output_tokens_per_second']-phase['output_tokens_per_second'])<1e-7
        assert check['slo']['passed_requests']==phase['slo']['passed_requests']
        grouped[(phase['mode'],phase['policy'])].append((phase,rows))
        records[(phase['mode'],phase['policy'],phase['trial'])]={r['id']:r for r in rows}
        steps=[json.loads(line) for line in (directory/(phase['name']+'_steps.jsonl')).read_text().splitlines()]
        if phase['policy']=='adaptive':
            decisions.update(step.get('decision',{}).get('reason','ordinary_step') for step in steps)
    def strata(rows,field):
        result={}
        for value in sorted({r[field] for r in rows}):
            items=[r for r in rows if r[field]==value]
            tpots=[(r['token_times'][-1]-r['token_times'][0])/(r['output_tokens']-1) for r in items]
            ttfts=[r['token_times'][0]-r['scheduled_at'] for r in items]
            result[str(value)]={'requests':len(items),'ttft':quantiles(ttfts),'tpot':quantiles(tpots),
                'slo_attainment':sum(a<=2. and b<=.1 for a,b in zip(ttfts,tpots))/len(items)}
        return result
    combined={}
    for (mode,policy),runs in grouped.items():
        rows=[r for _,items in runs for r in items]
        seconds=sum(p['wall_seconds'] for p,_ in runs)
        passed=sum(p['slo']['passed_requests'] for p,_ in runs)
        combined[f'{mode}/{policy}']={'requests':len(rows),'trials':len(runs),
            'output_tokens_per_second':sum(r['output_tokens'] for r in rows)/seconds,
            'goodput':passed/seconds,'slo_attainment':passed/len(rows),
            'ttft':quantiles([r['token_times'][0]-r['scheduled_at'] for r in rows]),
            'tpot':quantiles([(r['token_times'][-1]-r['token_times'][0])/(r['output_tokens']-1) for r in rows]),
            'itl':quantiles([b-a for r in rows for a,b in zip(r['token_times'],r['token_times'][1:])]),
            'by_input_length':strata(rows,'input_tokens'),'by_output_length':strata(rows,'output_tokens')}

    trial_deltas=[]
    for mode,policy,trial in sorted(records):
        if policy!='alternating' or (mode,'adaptive',trial) not in records:continue
        a=records[(mode,policy,trial)];b=records[(mode,'adaptive',trial)]
        assert a.keys()==b.keys()
        # Actual planned arrivals must match after removing the common phase start.
        for rid in a:
            if mode!='closed':assert abs(a[rid]['scheduled_at']-b[rid]['scheduled_at'])<1e-8
            x,y=a[rid]['token_ids'],b[rid]['token_ids']
            if x!=y:
                d=next(i for i,(u,v) in enumerate(zip(x,y)) if u!=v)
                all_cases.append({'mode':mode,'trial':trial,'request':rid,'input_tokens':a[rid]['input_tokens'],
                    'first_divergence':d,'baseline_token':x[d],'candidate_token':y[d],
                    'prefix':manifest[rid]['prompt_token_ids']+x[:d]})
        phases={p['policy']:p for p in source['phases'] if p['mode']==mode and p['trial']==trial}
        p,q=phases['alternating'],phases['adaptive']
        trial_deltas.append({'mode':mode,'trial':trial,
            'baseline_gpus':[v['gpu'] for v in p['peaks']],'candidate_gpus':[v['gpu'] for v in q['peaks']],
            'goodput_delta_pct':(q['slo']['goodput_requests_per_second']/p['slo']['goodput_requests_per_second']-1)*100,
            'throughput_delta_pct':(q['output_tokens_per_second']/p['output_tokens_per_second']-1)*100,
            'matching_outputs':sum(a[r]['output_sha256']==b[r]['output_sha256'] for r in a),'requests':len(a)})
    result={'combined':combined,'trial_deltas':trial_deltas,'decisions':dict(decisions),'divergent_requests':len(all_cases)}
    (directory/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (directory/'divergence_cases.json').write_text(json.dumps(all_cases,indent=2)+'\n')
    lines=['# 自适应调度：混合长度与突发流量对照','',
        '8×A100，两个四副本策略组。seed=29；输入 64/128/512/1024、输出 32/64/128 token；每场景每策略两轮各 256 请求，第二轮交换 GPU0–3/GPU4–7。计划到达序列、请求和轮询路由完全相同。',
        '闭环每卡并发16；Poisson 聚合2 req/s；突发每2s同时到达8请求（聚合4 req/s）。引擎原生接口，无 HTTP/网络。', '',
        '| 场景 | 策略 | tok/s | goodput req/s | SLO达标率 | TTFT p99 ms | TPOT p99 ms | ITL p99 ms |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for key,p in sorted(combined.items()):
        mode,policy=key.split('/')
        lines.append(f"| {mode} | {policy} | {p['output_tokens_per_second']:.1f} | {p['goodput']:.3f} | {p['slo_attainment']:.2%} | {p['ttft']['p99']*1000:.1f} | {p['tpot']['p99']*1000:.1f} | {p['itl']['p99']*1000:.1f} |")
    lines+=['','## 每轮收益与硬件交叉','',
        '| 场景 | 轮次 | baseline GPU | adaptive GPU | goodput变化 | 吞吐变化 | 输出hash一致 |',
        '|---|---:|---|---|---:|---:|---:|']
    for row in trial_deltas:
        lines.append(f"| {row['mode']} | {row['trial']} | {row['baseline_gpus']} | {row['candidate_gpus']} | {row['goodput_delta_pct']:+.1f}% | {row['throughput_delta_pct']:+.1f}% | {row['matching_outputs']}/{row['requests']} |")
    lines+=['','## 口径与限制','',
        '- 预设 SLO：每请求 TTFT≤2s 且平均TPOT≤100ms。没有按结果调整阈值。',
        '- 汇总吞吐/goodput为总token/总达标请求数除以两轮各自makespan之和；p99从合并的逐请求/逐token数据重新计算，不平均各轮p99。',
        '- 每策略每场景合计512请求，p99仍为探索性数据；只有两轮交叉，不提供统计显著性结论或生产SLA。重复同一manifest，不是独立语料样本。',
        '- CPU 元数据缓存对两种策略都开启。adaptive只改变执行顺序和prefill分块，不修改模型算术；batch/chunk形状变化仍会改变BF16输出，不能宣称跨策略逐位一致。',
        '- 执行预算是不可抢占前向的耗时预测，不是硬实时保证。负载超过目标SLO可承受能力时，decode余量与TTFT保护可能冲突。',
        '- 默认保持alternating；adaptive显式启用。需要结合所有场景、吞吐代价和HF审计判断，不能只挑goodput提升项。',
        '', '原始请求、逐步调度决策、manifest、源码快照与SHA256均保存在本目录。完整分层指标见summary.json；策略分歧见divergence_cases.json。']
    lines+=['','## 策略作用与采用结论','',
        f'决策计数：{dict(decisions)}。本轮真正按credit选择128-token配额仅1次；其余主要是延后prefill直到decode有余量或TTFT保护触发。收益主要来自时序变化，不能把它包装成预算预测全面奏效。',
        '饱和闭环goodput两轮分别+37.6%/+41.8%，汇总约+39.7%，但吞吐约−4.7%，SLO达标率仅11.3%→16.6%，TTFT/ITL p99没有改善。突发goodput两轮均下降，汇总约−3.7%，TTFT p99约0.85s→1.81s。Poisson收益不足1%，不能据此宣称显著改善。',
        '采用结论：默认alternating与已验证的CPU元数据优化保持不变；adaptive为未通过完整验收的实验选项，不能当作已验收生产策略或写成全面性能提升。']
    audit_path=directory/'hf_divergence_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        lines+=['','## 数值审计（未通过验收）','',
            f'共有{len(all_cases)}处跨策略输出分歧。按输入长度分层抽查{len(audit["cases"])}处；相同历史的HF单条完整前向参考中，candidate regret比baseline高出1以上的有{audit["candidate_worse_by_gt1"]}处，反方向有{audit["baseline_worse_by_gt1"]}处。这些分歧不能全部用近邻tie解释；既有baseline也存在大偏差，不据此断言调度是唯一根因。',
            '审计只覆盖抽样首次分歧，不能替代整模型raw logits验收。具体case、prefix与候选分数见hf_divergence_audit.json和divergence_cases.json。']
    self_path=directory/'hf_self_reference_audit.json'
    if self_path.exists():
        own=json.loads(self_path.read_text())
        lines += [f'HF自身batch1完整前向、batch2完整前向、batch1缓存增量前向的参考top1在{own["reference_top1_disagreements"]}/{len(own["cases"])}个抽样历史上不一致；所有参考均认为candidate相对baseline regret高出1以上的仍有{own["candidate_worse_gt1_all_references"]}处。参考路径敏感性不能用于豁免剩余误差，详见hf_self_reference_audit.json。']
    lines+=['','## 回归','',
        '最终回归44 passed、1 failed。保留的失败为Qwen3-0.6B test_batch_equals_individual：首个偏差已定位到第一次decode首层QKV GEMM的batch形状变化，输入相同而输出有0.001953125差异，最终在同分候选处分叉；新增同batch形状的请求重排隔离测试通过。原断言未放宽。详细日志/JUnit在../adaptive_regression.log与../adaptive_regression.xml。']
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();report(args.directory)
