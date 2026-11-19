"""Recompute paired scheduling metrics from raw per-request traces."""
import json
from pathlib import Path
from benchmarks.service_metrics import summarize


def load(directory):
    report=json.loads((directory/'summary.json').read_text())
    phases={}
    for p in report['phases']:
        rows=[json.loads(line) for line in (directory/(p['name']+'.jsonl')).read_text().splitlines()]
        assert len(rows)==p['requests']==len({r['id'] for r in rows})
        start=max(r['finished_at'] for r in rows)-p['wall_seconds']
        check=summarize(rows,start,p['replicas'])
        assert abs(check['output_tokens_per_second']-p['output_tokens_per_second'])<1e-6
        phases[p['name']]=p
    return report,phases


def main():
    root=Path(__file__).resolve().parent
    old,base=load(root/'scheduling_ab_baseline')
    new,candidate=load(root/'scheduling_ab_bounded')
    assert old['manifest_sha256']==new['manifest_sha256']
    assert old['config']['seed']==new['config']['seed']==17
    metadata={}
    if (root/'scheduling_ab_metadata_only'/'summary.json').exists():
        _,metadata=load(root/'scheduling_ab_metadata_only')
    comparison={}
    lines=['# 调度优化对照实验','',
           'Qwen3.5-27B，A100 80GB。GPU0–3 为原交替策略（GPU scalar metadata），GPU4–7 为 512-token 活跃 prefill 预算＋CPU 元数据缓存。每种策略 4 个独立单卡副本；相同 seed=17、请求 manifest、128-token 输入 / 64-token 输出。','',
           '闭环每卡并发 16；open-loop 为聚合 Poisson 4 req/s 轮询分发。各策略每场景 1024 请求；引擎原生接口，不含 HTTP/网络。','',
           '| 场景 | 方案 | output tok/s | TTFT p99 | TPOT p99 | ITL p99 | SLO 达标率 | SLO goodput req/s |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for phase,a in base.items():
        if phase not in candidate:continue
        b=candidate[phase]
        comparison[phase]={
            'throughput_change_pct':(b['output_tokens_per_second']/a['output_tokens_per_second']-1)*100,
            'ttft_p99_change_pct':(b['ttft_seconds']['p99']/a['ttft_seconds']['p99']-1)*100,
            'tpot_p99_change_pct':(b['tpot_seconds']['p99']/a['tpot_seconds']['p99']-1)*100,
            'itl_p99_change_pct':(b['itl_seconds']['p99']/a['itl_seconds']['p99']-1)*100,
            'slo_goodput_change_pct':(b['slo']['goodput_requests_per_second']/a['slo']['goodput_requests_per_second']-1)*100,
            'baseline':a,'candidate':b}
        variants=[('原策略',a),('bounded＋CPU元数据',b)]
        if phase in metadata:
            m=metadata[phase]
            comparison[phase]['metadata_only']=m
            comparison[phase]['metadata_only_throughput_change_pct']=(m['output_tokens_per_second']/a['output_tokens_per_second']-1)*100
            comparison[phase]['metadata_only_goodput_change_pct']=(m['slo']['goodput_requests_per_second']/a['slo']['goodput_requests_per_second']-1)*100
            variants.append(('仅CPU元数据（默认采用）',m))
        for name,p in variants:
            lines.append(f"| {phase} | {name} | {p['output_tokens_per_second']:.1f} | {p['ttft_seconds']['p99']*1000:.1f}ms | {p['tpot_seconds']['p99']*1000:.1f}ms | {p['itl_seconds']['p99']*1000:.1f}ms | {p['slo']['attainment']*100:.2f}% | {p['slo']['goodput_requests_per_second']:.3f} |")
    lines+=['','## 默认行为与使用方式','',
            '默认继续使用 alternating 交替调度；CPU 元数据缓存默认开启，未修改矩阵算术或 decode 算子。可用 cache_schedule_metadata=False 恢复旧读取方式。',
            'bounded 仅为显式选项：LLM(..., scheduling_policy="bounded", active_prefill_tokens=512)。适合需要限制单次停顿、且接受平均 TPOT/goodput 折中的实验，不默认启用。',
            '单独的 metadata-only 闭环复测沿用同一 GPU0–3 和 manifest，时间上晚于主 A/B；不能视作额外独立重复试验。',
            '', '## 必须保留的限制','',
            '- 预设 SLO 为每请求 TTFT≤2s 且 TPOT≤100ms；不按结果修改门槛。ITL 为逐 token 间隔，TPOT 为每请求首 token 到末 token 的平均间隔，两者不能混用。',
            '- 512-token 预算降低单次 prefill 阻塞，但增加 prefill 轮数，可能使平均 TPOT/SLO goodput 退化。必须同时看 TTFT、吞吐及 goodput，不能仅凭 ITL p99 下降宣称全面改善。',
            '- 主 A/B 同时改变预算和 CPU 元数据；总收益不能全部归因于单一因素。CPU 元数据的数值一致性有独立单测，整模型位级一致性与 nsys profile 另存。',
            '- 仅单次实验，固定长度合成文本，硬件分组未交叉轮换；没有生产 SLA 或跨引擎结论。open-loop 从计划到达时刻计 TTFT，包含实际提交滞后。',
            '- 改变批次组成会触发现有 bf16 模型的批次敏感性，不能声称不同调度下所有输出 token 都精确一致。','',
            '## 未采用的混合执行实验','',
            '先前试验将 prefill/decode 放进同一 varlen 前向，32 请求 pilot 显示 ITL 降低，但 HF 首次分歧审计出现明显 logit gap，未完成数值验收。因此混合执行未进入生产代码，其源代码与测试归档于 scheduling_mixed_experiment/source/，失败结果和审计在 scheduling_pilot_*。',
            '当前生产代码保留原来的分离执行、decode 算子和 IEEE FP32 精度；bounded 是显式调度选项，原 alternating 策略继续可用。','',
            '## 原始证据','',
            '- scheduling_ab_baseline/、scheduling_ab_bounded/、scheduling_ab_metadata_only/：汇总、manifest 和逐请求时间戳/token IDs。',
            '- scheduling_ab_source/：本次两个主实验使用的源代码快照及 SHA256。',
            '- scheduling_pilot_hf_audit.json：混合执行首次分歧的 HF 审计，包含明显超出近邻 tie 的案例。',
            '- scheduling_metadata_profile.json 与 profiles/scheduling_metadata*：安全 CPU 元数据优化的数值/profile 证据。']
    profile_path=root/'scheduling_metadata_nsys.json'
    if profile_path.exists():
        trace=json.loads(profile_path.read_text())
        old_sync=trace['device_scalars']['api_calls']['cudaStreamSynchronize_v3020']['count']
        new_sync=trace['host_metadata']['api_calls']['cudaStreamSynchronize_v3020']['count']
        old_copies=trace['device_scalars']['api_calls']['cudaMemcpyAsync_v3020']['count']
        new_copies=trace['host_metadata']['api_calls']['cudaMemcpyAsync_v3020']['count']
        consistency=json.loads((root/'scheduling_metadata_output_consistency.json').read_text())
        lines += ['', '## 已采用的安全优化证据','',
                  f'默认采用 CPU 元数据缓存，保留原 alternating 调度。25 项回归测试通过；单独 27B 整模型 profile 的完整 logits 最大差异为 0；闭环 {consistency["matching_requests"]}/{consistency["total_requests"]} 请求的输出 hash 与原版本一致。',
                  f'同一 8 请求 prefill 的 nsys 证据：cudaStreamSynchronize 调用 {old_sync}→{new_sync}；cudaMemcpyAsync {old_copies}→{new_copies}。减少的是重复 host/device 握手，不是修改 FP32 算术。',
                  '报告中的 100% SLO attainment 仅对应本轮有限样本，不能当作线上 SLA；100ms TPOT 门槛附近的小变化可能导致 goodput 较大的变化。']
    lines += ['', '## 扩展回归中的已知未通过项','',
              '调度专项 25 项通过。扩展 Qwen3 logits/生成回归合计 36 passed、1 failed：test_batch_equals_individual。关闭 cache_schedule_metadata 后同一用例同样失败，三个请求的批处理/单条输出及分歧位置与开启时完全相同。该问题在旧路径复现，根因尚未分析；没有放宽或删除断言，不能宣称完整测试全绿。详见 scheduling_verification.json。']
    (root/'scheduling_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (root/'SCHEDULING_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Updated',root/'SCHEDULING_REPORT.md')


if __name__=='__main__':main()
