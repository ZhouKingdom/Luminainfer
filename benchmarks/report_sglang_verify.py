"""Validate the SGLang-inspired experiment and keep numerical baselines explicit."""
import hashlib
import json
from pathlib import Path

ROOT = Path('benchmarks/qwen35_sglang_verify_20261107')


def validate_source(folder, result, current=False):
    for name, digest in result['source_sha256'].items():
        frozen = folder/'source'/name
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == digest, name
        if current and name.startswith('zjhvllm/'):
            assert frozen.read_bytes() == Path(name).read_bytes(), name


def read_mtp(name, current=False):
    folder = ROOT/name
    x = json.loads((folder/'result.json').read_text())
    assert x['complete'] and x['processes_exited']
    validate_source(folder, x, current)
    modes = {}
    for phase in x['warmups']+x['phases']:
        assert phase['reclaimed'] and len(phase['outputs']) == 8
        if x['args']['backend'] == 'decode_equivalent':
            assert phase['outputs'] == x['warmups'][0]['outputs'] and phase['exact'] == 8
        for group in phase['groups']:
            first = group['ranks'][0]
            for rank in group['ranks']:
                for key in ('verification_rows','verification_forwards','draft_forwards','draft_batch_calls'):
                    assert rank['counts'].get(key,0) == first['counts'].get(key,0)
        if phase['mode'].startswith('oracle'):
            assert phase['proposed'] == phase['accepted'] > 0 and phase['draft_forwards'] == 0
    for phase in x['phases']:
        modes.setdefault(phase['mode'],[]).append(phase)
    r = {}
    for mode, phases in modes.items():
        assert sorted(p['repeat'] for p in phases) == [0,1]
        assert all(p['outputs']==phases[0]['outputs'] for p in phases)
        r[mode] = {'decode_tok_s':sum(p['decode_output_tokens'] for p in phases)/sum(p['decode_seconds'] for p in phases),
                   'e2e_tok_s':sum(p['output_tokens'] for p in phases)/sum(p['e2e_seconds'] for p in phases),
                   'exact':phases[0]['exact'],'proposed':sum(p['proposed'] for p in phases),
                   'accepted':sum(p['accepted'] for p in phases),
                   'repeat_decode_tok_s':[p['decode_tok_s'] for p in phases]}
    # Kernel fusion must preserve its own unfused backend even when batched
    # arithmetic differs from the ordinary single-step batch shapes.
    for fused, original in [('mtp_batched_fused_graph','mtp_batched_graph'),('no_draft_fused_graph','no_draft_graph')]:
        assert [p['outputs'] for p in modes[fused]] == [p['outputs'] for p in modes[original]]
        assert r[fused]['accepted'] == r[original]['accepted'] and r[fused]['proposed'] == r[original]['proposed']
    return {'name':name,'args':x['args'],'rates':r,
            'mtp_fusion_gain_pct':(r['mtp_batched_fused_graph']['decode_tok_s']/r['mtp_batched_graph']['decode_tok_s']-1)*100,
            'mtp_vs_same_arithmetic_native_pct':(r['mtp_batched_fused_graph']['decode_tok_s']/r['native_graph']['decode_tok_s']-1)*100,
            'generations':sum(len(p['outputs']) for p in x['phases']),
            'output_tokens':sum(p['output_tokens'] for p in x['phases'])}


def main():
    primary = [read_mtp('full_tp2_c1',True),read_mtp('full_tp2_c4',True)]
    recurrence_only = [read_mtp('final_tp2_c1'),read_mtp('final_tp2_c4')]
    batched = read_mtp('batched_tp2_c4',True)
    ordinary_folder = ROOT/'ordinary_tp2_c4'
    ordinary = json.loads((ordinary_folder/'result.json').read_text())
    assert ordinary['complete'] and ordinary['processes_exited']
    validate_source(ordinary_folder,ordinary,True)
    assert len(ordinary['phases'])==2
    assert all(p['outputs']==ordinary['warmups'][0]['outputs'] and p['reclaimed'] for p in ordinary['phases'])
    ordinary_rates = {key:sum(p[tokens] for p in ordinary['phases'])/sum(p[seconds] for p in ordinary['phases'])
                      for key,tokens,seconds in [('decode_tok_s','decode_output_tokens','decode_seconds'),('e2e_tok_s','output_tokens','e2e_seconds')]}
    ordinary_rates['same_as_rowwise_reference']=ordinary['phases'][0]['same_as_rowwise_reference']
    for tp in (2,4,8):
        validation=json.loads((ROOT/f'full_synthetic_tp{tp}.json').read_text())
        assert validation['passed'] and validation['processes_exited']
        assert len(validation['ranks'])==tp
        assert all(rank['passed'] and rank['verification_fused_gdn'] and rank['packed_draft']
                   and rank['packed_draft_rollback_faults']==['prepare','prediction'] for rank in validation['ranks'])
    micro=json.loads((ROOT/'kernel_microbench.json').read_text())
    assert all(r['exact'] for r in micro['rows'])
    reference=json.loads((ROOT/'参考源码记录.json').read_text())
    summary={'primary':primary,'recurrence_only':recurrence_only,'batched_experimental':batched,
             'ordinary_actual':ordinary_rates,'microbench':micro,'reference':reference,
             'main_measured_generations':sum(r['generations'] for r in primary)+batched['generations']+16,
             'main_measured_output_tokens':sum(r['output_tokens'] for r in primary)+batched['output_tokens']+sum(p['output_tokens'] for p in ordinary['phases'])}
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# SGLang参考实现与验证报告','',
           f"参考本地SGLang提交 `{reference['commit']}`，选定文件无本地修改；SHA256见参考源码记录.json。没有运行SGLang服务性能对比。",'',
           '## 参考了什么','',
           '- SGLang的TritonGDNKernel.target_verify：禁止直接更新持久S，窗口递推在一个kernel中完成，缓存各步中间状态，再按接受边界提交。',
           '- GDN后端的causal_conv1d_update验证分支：批量处理短窗口、保存中间conv窗口。',
           '- Qwen3.5模型还合并qkv/z及b/a投影，草稿worker有独立Graph。Lumina本轮只实现短窗口卷积与递推融合，保留默认GEMM/norm及NCCL算术，并未完整复刻这些优化。','',
           '源文件：`/home/jhzhou/sft_prj/sglang/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py`、`kernels/ops/attention/linear/fused_sigmoid_gating_recurrent.py`、`srt/layers/attention/linear/gdn_backend.py`、`srt/models/qwen3_5.py`。后三条相对于同一python/sglang目录。','',
           '## Lumina改动','',
           '新增可选 `verification_fused_gdn=True`，用于1/2行验证窗口。自实现Triton卷积窗口与递推kernel，读取请求私有初始状态，递推时不重读上一时刻S，所有中间S/conv写入scratch，接受后沿用原提交与故障回滚。保持BF16卷积输出舍入边界、FP32状态及原递推算术顺序；FP32/其他卷积宽度保留原卷积准备路径，更宽验证窗口回退原实现。默认关闭。',
           '', 'Graph持有独立cu_seqlens缓冲，重放更新动态元数据；融合与原路径分别缓存，避免切换开关后误用旧Graph。','',
           '## 严格逐行算术对照','',
           '真实27B、TP2、BF16、默认decode_equivalent；每配置8条输入，C1每条32输出，C4每条64输出。完整预热、两轮反序、相同prefill，同引擎内开关融合。单位为prefill后output tok/s。',
           '', '| 并发 | 同算术普通Graph | 原批量草稿MTP | 融合后MTP | MTP路径提升 | 原免费草稿 | 融合免费草稿 |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for row in primary:
        r=row['rates'];lines.append(f"| {row['args']['capacity']} | {r['native_graph']['decode_tok_s']:.3f} | {r['mtp_batched_graph']['decode_tok_s']:.3f} | {r['mtp_batched_fused_graph']['decode_tok_s']:.3f} | {row['mtp_fusion_gain_pct']:+.2f}% | {r['oracle_graph']['decode_tok_s']:.3f} | {r['oracle_fused_graph']['decode_tok_s']:.3f} |")
    lines+=['', '本节全部模式输出逐token一致。免费草稿来自离线参考、全部接受、没有草稿模型成本，只用于诊断，不可当作真实MTP成果。这里的普通Graph刻意固定C1算术，与关闭MTP的普通批量引擎并非同一路径。', '',
            '## 实际普通引擎与批量算术实验','',
            f"关闭MTP、保持实际普通批量Graph，同一C4输入/64输出、预热后两轮：解码 **{ordinary_rates['decode_tok_s']:.3f} tok/s**，E2E **{ordinary_rates['e2e_tok_s']:.3f} tok/s**。完整输出相对逐行参考为{ordinary_rates['same_as_rowwise_reference']}/8一致，跨算术路径差异保留。这个结果说明战胜逐行参考不能等价于战胜实际普通推理。",'',
            '| batched实验模式（C4） | 解码 tok/s | E2E tok/s | 相对本后端普通Graph的完整序列一致数 |',
            '|---|---:|---:|---:|']
    for name,label in [('native_graph','普通Graph'),('no_draft_graph','原无草稿验证'),('mtp_batched_graph','原批量草稿MTP'),('no_draft_fused_graph','融合无草稿验证'),('mtp_batched_fused_graph','融合MTP')]:
        r=batched['rates'][name];lines.append(f"| {label} | {r['decode_tok_s']:.3f} | {r['e2e_tok_s']:.3f} | {r['exact']}/8 |")
    lines+=['', f"融合相对原批量草稿MTP：{batched['mtp_fusion_gain_pct']:+.2f}%；相对本后端普通Graph：{batched['mtp_vs_same_arithmetic_native_pct']:+.2f}%。融合前后MTP完整输出与接受计数相同，但batched后端自身相对普通解码已有输出差异，不能作为默认无损加速交付。",'',
            '## 正确性、局部计时与证据','',
            '- 完整融合相关CUDA回归142项通过，额外1项Graph开关切换测试通过；覆盖BF16卷积、每步FP32 S、中间conv、混合长度、动态槽位/页表、Graph及旧数值路径。TP2/4/8分布式接受/回滚测试全部通过并正常退出。',
            f"- 最新完整融合/批量后端/实际普通控制正式共{summary['main_measured_generations']}次生成、{summary['main_measured_output_tokens']}个输出token。严格逐行后端输出全部匹配；实验batched差异逐项保留。递推首版试验独立保存在final_tp2_c*，不混入这些统计。",
            '- 递推kernel隔离CUDA Graph计时：'+ '；'.join(f"C{r['capacity']} {r['cuda_graph_milliseconds']['decode_loop']*1000:.2f}→{r['cuda_graph_milliseconds']['window_kernel']*1000:.2f}微秒（{r['speedup']:.2f}×）" for r in micro['rows'])+'。仅准备好的q/k/v/g/beta递推，不含投影、conv、通信或调度，不能当端到端收益。',
            '- 两轮速率用累计token/累计时间合并；同引擎各模式顺序反转，加载/JIT/Graph捕获不计时，全rank有限值检查计时。各实验使用不重叠GPU组，CPU共享，非隔离环境；实际普通控制是单独引擎，避免拿微小差异作强结论。',
            '- 原始结果、逐轮输出与各rank计数：full_tp2_c*/result.json、batched_tp2_c4/result.json、ordinary_tp2_c4/result.json。每组有冻结源码与SHA256。', '',
            '## 后续重点','',
            '继续参考SGLang的合并投影、批量attention和草稿Graph，减少默认逐行路径的成本；批量算术路径需单独完成输出差异与质量验收。仍以实际普通批量Graph为性能目标，以相同后端的普通解码为正确性控制。没有因为局部内核快了就宣布投机已经有效，也未宣称HF严格logits验收通过。']
    lines += ['', '## 复现命令', '', '使用现有lumina解释器，从Lumina根目录运行；先检查GPU空闲，输出目录必须是新的，端口不得与其他引擎冲突。', '',
              '```bash',
              'CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. /home/data/jhzhou/conda_envs/lumina/bin/python benchmarks/bench_mtp_net_gain.py --tp 2 --tokens 64 --repeats 2 --oracle --batch-draft --compare-fused-gdn --port 2542 --output benchmarks/融合复测_新运行',
              '```', '',
              '测试batched后端时去掉--oracle并增加--backend batched；C1增加--capacity 1 --tokens 32。实际普通引擎控制用benchmarks/bench_qwen35_ordinary_control.py的--workload指向对应result.json，并指定新的--output及独立--port。']
    (ROOT/'SGLang参考实现与验证报告.md').write_text('\n'.join(lines)+'\n')
    for row in primary+[batched]:
        print(row['name'],'fusion',row['mtp_fusion_gain_pct'],'vs native',row['mtp_vs_same_arithmetic_native_pct'])
    print('actual ordinary',ordinary_rates)


if __name__=='__main__':main()
