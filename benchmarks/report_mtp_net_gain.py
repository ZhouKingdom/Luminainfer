"""Summarize the matched Graph experiment without attributing Graph to MTP."""
import hashlib
import json
from pathlib import Path

root=Path('benchmarks/qwen35_mtp_net_gain_20261107')
summary={'complete':True,'scope':'Qwen3.5-27B default decode_equivalent, TP2/4/8, fixed C4 cohorts, eight prompts, 64 output tokens, two reverse-order repeats, all-rank finite checks',
    'measured_generations':0,'measured_output_tokens':0,'tp':[]}
for tp in (2,4,8):
    folder=root/f'final_tp{tp}'
    x=json.loads((folder/'result.json').read_text())
    assert x['complete'] and x['processes_exited']
    assert x['args']['capacity']==4 and x['args']['tokens']==64 and x['args']['repeats']==2
    for name,sha in x['source_sha256'].items():
        frozen=folder/'source'/name
        assert hashlib.sha256(frozen.read_bytes()).hexdigest()==sha
        if name.startswith('zjhvllm/'):assert Path(name).read_bytes()==frozen.read_bytes(),name
    assert set(x['native_buckets'])=={1,2,3,4}
    reference=x['warmups'][0]['outputs']
    assert all(w['exact']==8 and w['outputs']==reference for w in x['warmups'])
    modes={}
    for p in x['phases']:
        assert p['exact']==8 and p['outputs']==reference and p['reclaimed']
        assert p['decode_output_tokens']==504 and p['output_tokens']==512
        summary['measured_generations']+=8;summary['measured_output_tokens']+=512
        for g in p['groups']:
            rank0=g['ranks'][0]
            for rank in g['ranks']:
                assert rank['graphs']==rank0['graphs']
                for key in ('native_target_forwards','native_target_rows','verification_forwards','verification_rows','draft_forwards','draft_extend_pairs'):
                    assert rank['counts'].get(key,0)==rank0['counts'].get(key,0)
        modes.setdefault(p['mode'],[]).append(p)
    rates={}
    for mode,phases in modes.items():
        assert sorted(p['repeat'] for p in phases)==[0,1]
        rates[mode]={'decode_tok_s':sum(p['decode_output_tokens'] for p in phases)/sum(p['decode_seconds'] for p in phases),
            'e2e_tok_s':sum(p['output_tokens'] for p in phases)/sum(p['e2e_seconds'] for p in phases),
            'target_forwards':sum(p['target_forwards'] for p in phases),'target_rows':sum(p['target_rows'] for p in phases),
            'draft_forwards':sum(p['draft_forwards'] for p in phases),'proposed':sum(p['proposed'] for p in phases),
            'accepted':sum(p['accepted'] for p in phases),'repeat_decode_tok_s':[p['decode_tok_s'] for p in phases],
            'repeat_e2e_tok_s':[p['e2e_tok_s'] for p in phases]}
    no=rates['no_draft_graph'];native=rates['native_graph'];mtp=rates['mtp_graph']
    assert no['proposed']==no['accepted']==no['draft_forwards']==0
    assert no['target_forwards']==native['target_forwards']==252
    assert no['target_rows']==native['target_rows']==1008
    summary['tp'].append({'tp':tp,'rates':rates,
        'net_decode_vs_matched_control_pct':(mtp['decode_tok_s']/no['decode_tok_s']-1)*100,
        'net_e2e_vs_matched_control_pct':(mtp['e2e_tok_s']/no['e2e_tok_s']-1)*100,
        'net_decode_vs_native_graph_pct':(mtp['decode_tok_s']/native['decode_tok_s']-1)*100,
        'net_e2e_vs_native_graph_pct':(mtp['e2e_tok_s']/native['e2e_tok_s']-1)*100,
        'acceptance_pct':mtp['accepted']/mtp['proposed']*100,
        'target_forward_reduction_pct':(1-mtp['target_forwards']/no['target_forwards'])*100,
        'target_row_increase_pct':(mtp['target_rows']/no['target_rows']-1)*100})
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 默认多卡 MTP 投机净收益测试','',
    '本轮直接比较开启/关闭真实MTP草稿的净收益，双方都开启Graph。另加入直接走普通decode Graph的控制，避免把验证事务开销当作无草稿解码的必要成本。生产引擎计算代码未修改，所有插桩在基准脚本内。',
    '', 'Qwen3.5-27B，BF16，默认decode_equivalent，TP2/4/8，8条输入、每条64输出、greedy/ignore EOS，固定C4 cohort。输入长度36/25/106/128/64/96/255/256。每模式完整预热后测两轮，第二轮反转模式顺序。',
    '', '| TP | 普通decode Graph | 无草稿验证Graph | MTP验证Graph | MTP相对无草稿 | MTP相对普通Graph |','|---:|---:|---:|---:|---:|---:|']
for row in summary['tp']:
    r=row['rates'];lines.append(f"| {row['tp']} | {r['native_graph']['decode_tok_s']:.3f} | {r['no_draft_graph']['decode_tok_s']:.3f} | {r['mtp_graph']['decode_tok_s']:.3f} | {row['net_decode_vs_matched_control_pct']:+.2f}% | {row['net_decode_vs_native_graph_pct']:+.2f}% |")
lines+=['', '上表单位output tok/s，仅统计相同prefill完成后的解码窗口；每模式各1008个计时输出token（两轮、每轮504），不把prefill混入分母。先做完全相同的普通eager prefill，再切换解码模式；每请求首个输出不计入decode分子。',
    '', '| TP | 普通Graph E2E | 无草稿Graph E2E | MTP Graph E2E | MTP相对无草稿E2E |','|---:|---:|---:|---:|---:|']
for row in summary['tp']:
    r=row['rates'];lines.append(f"| {row['tp']} | {r['native_graph']['e2e_tok_s']:.3f} | {r['no_draft_graph']['e2e_tok_s']:.3f} | {r['mtp_graph']['e2e_tok_s']:.3f} | {row['net_e2e_vs_matched_control_pct']:+.2f}% |")
lines+=['', 'E2E是各cohort的prefill与decode wall time之和，每模式1024个输出token；排除加载、Graph捕获/预热、基准控制切换和测量数据收集。有效计算期间全rank有限值检查计时。MTP首次草稿扩展成本、真实草稿层/LM head、IPC、快照、接受与提交均计入。',
    '', '| TP | 草稿接受率 | 目标遍历次数下降 | 目标计算行数增加 | 两轮MTP目标遍历 | 两轮MTP草稿前向 |','|---:|---:|---:|---:|---:|---:|']
for row in summary['tp']:
    r=row['rates']['mtp_graph'];lines.append(f"| {row['tp']} | {row['acceptance_pct']:.2f}% | {row['target_forward_reduction_pct']:.2f}% | {row['target_row_increase_pct']:+.2f}% | {r['target_forwards']} | {r['draft_forwards']} |")
lines+=['', '无草稿两轮固定为252次目标遍历、1008行目标计算。一次两行窗口虽只遍历模型一次，但默认数值路径内部仍逐token投影/递归/归约；遍历次数下降不等于GPU目标计算同比下降。投机还增加草稿层及共享LM head计算，拒绝时有未提交的目标行。因此接受率高不保证净提速。',
    '', f"正式测量共{summary['measured_generations']}次生成、{summary['measured_output_tokens']}个输出token，所有模式/重复的完整输出均与同引擎普通Graph参考逐token一致、每组资源完整回收、各rank计数一致、全部工作进程正常退出。另有完整预热记录，不混入正式吞吐。",
    '', '普通Graph控制保持与MTP目标相同的逐token数值算术，并在基准中额外捕获B3，防止缺桶回退eager使无草稿控制吃亏；它仍是hidden-state Graph加Graph外LM head/采样。无草稿验证控制则与MTP一样捕获整个目标与LM head，保留相同验证快照/提交机制，唯一关闭草稿。两者分开报告，不冒充同一种普通服务路径。',
    '', '结论只覆盖当前默认数值后端、此C4固定cohort小集和对应硬件，不代表所有输入、连续在线负载或高并发容量，也不能否定其他实现/后端下的投机算法。TP2/4在互不重叠GPU组分组测量，TP8随后单独测量；共享CPU、非隔离环境，不用来宣称强扩展效率。',
    '', '原始输出、逐轮接受、各rank目标/草稿次数与时间、源码SHA256和预热记录见final_tp*/result.json。控制Graph动态槽/页与B1/2/3/4相对eager的数值检查4项通过，连同原Graph/连续补位测试11项通过；基准导入不污染生产测试。']
(root/'投机净收益报告.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='tp'}))
for row in summary['tp']:print(row['tp'],row['net_decode_vs_matched_control_pct'],row['net_e2e_vs_matched_control_pct'])
