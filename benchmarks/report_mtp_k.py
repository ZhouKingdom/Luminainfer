"""Summarize recurrent MTP-K, including depth survival and real execution cost."""
import hashlib
import json
from pathlib import Path

ROOT=Path('benchmarks/qwen35_mtp_k_20261107')


def read_case(name):
    folder=ROOT/name;x=json.loads((folder/'result.json').read_text())
    assert x['complete'] and x['processes_exited'] and x['args']['mtp_steps']==[1,2,4]
    for path,digest in x['source_sha256'].items():
        frozen=folder/'source'/path
        assert hashlib.sha256(frozen.read_bytes()).hexdigest()==digest
        if path.startswith('zjhvllm/'):assert frozen.read_bytes()==Path(path).read_bytes(),path
    strict=x['args']['backend']=='decode_equivalent'
    reference=x['warmups'][0]['outputs']
    for phase in x['warmups']+x['phases']:
        assert phase['reclaimed'] and len(phase['outputs'])==8
        if strict:assert phase['outputs']==reference and phase['exact']==8
        for group in phase['groups']:
            counts=group['ranks'][0]['counts']
            for rank in group['ranks']:
                for key in ('verification_forwards','verification_rows','draft_forwards','draft_batch_calls','draft_extend_pairs'):
                    assert rank['counts'].get(key,0)==counts.get(key,0)
    modes={}
    for phase in x['phases']:modes.setdefault(phase['mode'],[]).append(phase)
    assert set(modes)=={'native_graph','no_draft_fused_graph','mtp_k1_graph','mtp_k2_graph','mtp_k4_graph'}
    rates={}
    for mode,phases in modes.items():
        assert sorted(p['repeat'] for p in phases)==[0,1]
        assert phases[0]['outputs']==phases[1]['outputs']
        rounds=[r for p in phases for g in p['groups'] for r in g['rounds']]
        counts={}
        for p in phases:
            for group in p['groups']:
                for key,value in group['ranks'][0]['counts'].items():counts[key]=counts.get(key,0)+value
        tokens=sum(p['decode_output_tokens'] for p in phases)
        if rounds:assert sum(r['state_commit_tokens'] for r in rounds)==tokens
        depth=[]
        if mode.startswith('mtp_k'):
            k=int(mode.split('_')[1][1:])
            for d in range(1,k+1):
                eligible=sum(r['proposed']>=d for r in rounds)
                reached=sum(r['proposed']>=d and r['accepted']>=d-1 for r in rounds)
                accepted=sum(r['accepted']>=d for r in rounds)
                depth.append({'depth':d,'proposed_rounds':eligible,'reached_rounds':reached,'accepted_rounds':accepted,
                    'joint_accept_pct':100*accepted/eligible if eligible else None,
                    'conditional_accept_pct':100*accepted/reached if reached else None})
        rates[mode]={'decode_tok_s':tokens/sum(p['decode_seconds'] for p in phases),
            'e2e_tok_s':sum(p['output_tokens'] for p in phases)/sum(p['e2e_seconds'] for p in phases),
            'decode_seconds':sum(p['decode_seconds'] for p in phases),
            'counts_rank0':counts,'exact':phases[0]['exact'],
            'same_as_k1':sum(a==b for a,b in zip(phases[0]['outputs'],modes['mtp_k1_graph'][0]['outputs'])),
            'proposed':sum(p['proposed'] for p in phases),'accepted':sum(p['accepted'] for p in phases),
            'verification_request_rounds':len(rounds),'mean_emitted_per_request_round':tokens/len(rounds) if rounds else None,
            'target_forwards':sum(p['target_forwards'] for p in phases),
            'target_rows_per_output':sum(p['target_rows'] for p in phases)/tokens,
            'acceptance_by_depth':depth,'repeat_decode_tok_s':[p['decode_tok_s'] for p in phases]}
    for k in (1,2,4):
        rate=rates[f'mtp_k{k}_graph']
        rate['vs_k1_pct']=(rate['decode_tok_s']/rates['mtp_k1_graph']['decode_tok_s']-1)*100
        rate['vs_native_pct']=(rate['decode_tok_s']/rates['native_graph']['decode_tok_s']-1)*100
    return {'name':name,'args':x['args'],'rates':rates,
        'measured_generations':sum(len(p['outputs']) for p in x['phases']),
        'measured_output_tokens':sum(p['output_tokens'] for p in x['phases'])}


def main():
    rows=[read_case('default_c1'),read_case('batched_c4')]
    for tp in (2,4,8):
        x=json.loads((ROOT/f'synthetic_tp{tp}.json').read_text())
        assert x['passed'] and x['processes_exited'] and len(x['ranks'])==tp
        assert all(r['mtp_steps']==4 and r['recursive_rollback_faults']==['prepare','prediction'] for r in x['ranks'])
    summary={'cases':rows,'measured_generations':sum(r['measured_generations'] for r in rows),
             'measured_output_tokens':sum(r['measured_output_tokens'] for r in rows)}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 递归MTP-k实现与实测','',
        '已按SGLang topk=1递归草稿路径实现K步草稿：后续候选使用前一步草稿hidden和预测token，而非读取尚不存在的目标hidden。checkpoint仍只有一层MTP，K是递归次数，不是加载K个独立训练的MTP层。参考提交和SHA256见SGLang递归草稿参考.json。', '',
        '## 实现与状态边界','',
        '- 新增 `mtp_num_steps=1–8`，默认1。K>1自动使用批量递归；已生成的预测留在GPU，每批结束仅一次转为CPU token列表，没有每步.item()或主机准备投票。',
        '- 第一轮扩展只使用已提交的真实目标hidden/token对。后续递归KV始终视为临时数据，返回前截回第一轮真实扩展的边界；即使目标全接受，下轮仍用真实目标hidden重建，避免近似hidden污染长期KV。',
        '- 任一步失败恢复所有请求的草稿长度；目标验证失败由原事务恢复S/conv、KV页和草稿边界。多卡支持准备故障及最终预测广播完成后的可恢复错误；不承诺通信序列内部任意错误可恢复。',
        '- Graph和融合卷积/递推扩展到最多9行输入（pending+8草稿），保存每步S/conv，按请求独立接受、提交。更宽的手工验证窗口保留原eager路径。',
        '- 构造时按配置最大K预留scratch，再计算KV预算；预算不足在Config中提前报错。布局缓存默认16、可显式扩至64；本轮真实测试使用64，scratch预算1024MiB/配置请求。', '',
        '## 成对吞吐','',
        '真实27B、BF16、TP2，同一常驻引擎、同prefill、同输入和输出预算；所有模式完整预热后测两轮，第二轮反序。加载、JIT和Graph捕获不计时，全rank有限值检查计时。单位为prefill后的output tok/s。', '',
        '| 配置 | 普通Graph | K=1 | K=2 | K=4 | K=2相对K=1 | K=2相对普通 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case in rows:
        r=case['rates'];label=f"{case['args']['backend']} / C{case['args']['capacity']} / {case['args']['tokens']}输出"
        lines.append(f"| {label} | {r['native_graph']['decode_tok_s']:.3f} | {r['mtp_k1_graph']['decode_tok_s']:.3f} | {r['mtp_k2_graph']['decode_tok_s']:.3f} | {r['mtp_k4_graph']['decode_tok_s']:.3f} | {r['mtp_k2_graph']['vs_k1_pct']:+.2f}% | {r['mtp_k2_graph']['vs_native_pct']:+.2f}% |")
    lines+=['', '默认C1为严格逐行数值控制，全部K输出8/8一致。batched C4使用正常批量GEMM/norm，但其验证窗口相对普通解码存在已知BF16输出差异；K1/K2/K4各自相对普通均4/8完整序列一致，K2相对K1为8/8，K4相对K1仅4/8。原始输出全部保留，不能将实验batched结果作为默认无损交付。', '',
        '普通Graph仍是MTP常驻引擎中匹配后端的控制：默认路径刻意固定C1算术，batched路径使用批量算术且attention仍有逐行控制。上一轮关闭MTP的实际普通C4引擎约123 tok/s，是另一个性能基线；此处未把低速逐行控制当作最佳普通批量推理。两种配置的并发及输出长度也不同，不用它们的差异单独归因某个因素。', '',
        '| 配置 / K | E2E output tok/s | 每请求轮平均提交token | 目标计算行/解码输出token | 目标遍历次数 | 草稿前向次数 | 草稿函数累计s |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for case in rows:
        for k in (1,2,4):
            r=case['rates'][f'mtp_k{k}_graph'];c=r['counts_rank0']
            lines.append(f"| {case['args']['backend']} C{case['args']['capacity']} / {k} | {r['e2e_tok_s']:.3f} | {r['mean_emitted_per_request_round']:.3f} | {r['target_rows_per_output']:.3f} | {r['target_forwards']} | {c['draft_forwards']} | {c['draft_seconds']:.3f} |")
    lines+=['', '计数与累计耗时合并两轮；草稿函数耗时为rank0同步边界wall time，含函数内通信/检查，不含函数外全部IPC/事务，不是纯GPU kernel profile。平均提交量直接统计实际提交，包含尾部预算截断的轮次。', '',
        '## 草稿深度与接受','',
        '| 配置 / K | 深度 | 前缀联合接受率 | 已接受前缀条件下的本步接受率 |',
        '|---|---:|---:|---:|']
    for case in rows:
        for k in (2,4):
            for d in case['rates'][f'mtp_k{k}_graph']['acceptance_by_depth']:
                lines.append(f"| {case['args']['backend']} C{case['args']['capacity']} / {k} | {d['depth']} | {d['joint_accept_pct']:.2f}% | {d['conditional_accept_pct']:.2f}% |")
    lines+=['', '联合接受率按具有该深度草稿的请求轮计算；条件接受率分母进一步限定此前缀已接受。不同深度的预算筛选不同，不能直接相加代替全体平均提交量。一般无EOS/预算截断时，期望产出为1+各深度前缀联合接受概率之和。', '',
        '结论：K确实增加每次验证的提交量；本轮batched C4的K2优于K1，但K4增加了更多草稿前向和未提交验证行，收益回落。固定更大K并不自动超过普通解码，应按实际平均接受长度与每轮成本选择K，并继续减少递归草稿/验证开销。没有用MTP-1接受率直接外推更深草稿。', '',
        '## 验收与复现','',
        f"正式共{summary['measured_generations']}次生成、{summary['measured_output_tokens']}个输出token；各模式两轮完整输出一致，所有组资源回收、工作进程正常退出。默认路径跨K也严格一致；实验batched差异保留。163项相关CUDA/CPU回归通过，包含K2/4/8宽窗口全接受、部分拒绝和跨页；TP2/4/8的递归MTP4与宽窗口Graph/故障回滚均通过。真实模型只测TP2的K1/2/4，不宣称K8真实质量/吞吐已验收。", '',
        '```bash',
        'CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. /home/data/jhzhou/conda_envs/lumina/bin/python benchmarks/bench_mtp_net_gain.py --tp 2 --capacity 4 --tokens 64 --repeats 2 --backend batched --mtp-steps 1 2 4 --port 2550 --output benchmarks/MTP多步复测_新运行',
        '```', '',
        '默认严格数值C1对照去掉--backend batched，并改--capacity 1 --tokens 32；先检查GPU占用、独立端口和新输出目录。两种模式都使用融合验证和批量草稿，分别比较对应普通Graph。', '',
        '原始结果、逐轮接受长度、计数、冻结源码与SHA256见default_c1/和batched_c4/的result.json；final_regression.log为回归证据，synthetic_tp*.json记录各rank的故障覆盖与退出。当前默认仍K1，未自动启用动态K，也未捕获递归草稿Graph。']
    (ROOT/'MTP多步实现与测试报告.md').write_text('\n'.join(lines)+'\n')
    for case in rows:
        print(case['name'],[(k,case['rates'][f'mtp_k{k}_graph']['decode_tok_s']) for k in (1,2,4)])


if __name__=='__main__':main()
