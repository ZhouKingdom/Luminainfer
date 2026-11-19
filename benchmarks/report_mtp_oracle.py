"""Validate and summarize offline-oracle diagnostics without claiming speedup."""
import argparse
import hashlib
import json
from pathlib import Path


def summarize(folder):
    result = json.loads((folder / 'result.json').read_text())
    assert result['complete'] and result['processes_exited'], folder
    assert result['args']['oracle'] and result['args']['batch_draft'] and result['args']['backend'] == 'decode_equivalent'
    for name, expected in result['source_sha256'].items():
        assert hashlib.sha256((folder / 'source' / name).read_bytes()).hexdigest() == expected, name
    reference = result['warmups'][0]['outputs']
    for phase in result['warmups'] + result['phases']:
        assert phase['outputs'] == reference and phase['exact'] == len(reference)
        assert phase['reclaimed']
        for group in phase['groups']:
            rank0 = group['ranks'][0]
            for rank in group['ranks']:
                for key in ('native_target_forwards', 'native_target_rows',
                            'verification_forwards', 'verification_rows', 'draft_forwards'):
                    assert rank['counts'].get(key, 0) == rank0['counts'].get(key, 0)
        if phase['mode'] == 'oracle_graph':
            assert phase['proposed'] == phase['accepted'] > 0
            assert phase['draft_forwards'] == 0
    modes = {}
    for phase in result['phases']:
        modes.setdefault(phase['mode'], []).append(phase)
    assert set(modes) == {'native_graph', 'no_draft_graph', 'mtp_graph', 'oracle_graph', 'mtp_batched_graph'}
    rates = {}
    for mode, phases in modes.items():
        assert sorted(p['repeat'] for p in phases) == list(range(result['args']['repeats']))
        decode_seconds = sum(p['decode_seconds'] for p in phases)
        counts = {}
        for phase in phases:
            for group in phase['groups']:
                for name, value in group['ranks'][0]['counts'].items():
                    counts[name] = counts.get(name, 0) + value
        rates[mode] = {
            'decode_tok_s': sum(p['decode_output_tokens'] for p in phases) / decode_seconds,
            'e2e_tok_s': sum(p['output_tokens'] for p in phases) / sum(p['e2e_seconds'] for p in phases),
            'decode_seconds': decode_seconds, 'counts_rank0': counts,
            'proposed': sum(p['proposed'] for p in phases),
            'accepted': sum(p['accepted'] for p in phases),
            'target_rows': sum(p['target_rows'] for p in phases),
            'target_forwards': sum(p['target_forwards'] for p in phases),
            'repeat_decode_tok_s': [p['decode_tok_s'] for p in phases],
        }
    native = rates['native_graph']['decode_tok_s']
    assert rates['oracle_graph']['target_rows'] == rates['native_graph']['target_rows']
    return {'folder': str(folder), 'args': result['args'], 'rates': rates,
            'oracle_vs_native_pct': (rates['oracle_graph']['decode_tok_s'] / native - 1) * 100,
            'batch_draft_vs_serial_pct': (rates['mtp_batched_graph']['decode_tok_s'] / rates['mtp_graph']['decode_tok_s'] - 1) * 100,
            'batch_draft_vs_native_pct': (rates['mtp_batched_graph']['decode_tok_s'] / native - 1) * 100,
            'mtp_vs_native_pct': (rates['mtp_graph']['decode_tok_s'] / native - 1) * 100,
            'measured_generations': sum(len(p['outputs']) for p in result['phases']),
            'measured_output_tokens': sum(p['output_tokens'] for p in result['phases'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('benchmarks/qwen35_mtp_oracle_20261107'))
    parser.add_argument('--folders', nargs='+', default=['final_tp2_c4', 'final_tp2_c1'])
    args = parser.parse_args()
    for tp, name in [(2, 'synthetic_batch_fixed_tp2.json'), (4, 'synthetic_batch_tp4.json'), (8, 'synthetic_batch_tp8.json')]:
        validation = json.loads((args.root / name).read_text())
        assert validation['tp'] == tp and validation['passed'] and validation['processes_exited']
        assert len(validation['ranks']) == tp
        assert all(rank['passed'] and rank['packed_draft']
                   and rank['packed_draft_rollback_faults'] == ['prepare', 'prediction']
                   for rank in validation['ranks'])
    rows = [summarize(args.root / folder) for folder in args.folders]
    (args.root / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    lines = ['# MTP-1 免费正确草稿诊断', '',
             '这是测当前验证执行路径收益空间的乐观诊断，免费草稿由普通Graph完整输出预先提供，不能部署，也不能作为真实MTP性能成果。目标验证、Graph输入更新、S/conv历史写入、状态提交、页管理及调度仍实际执行。', '',
             '默认decode_equivalent，真实Qwen3.5-27B、BF16、TP2。C4每条64输出；C1每条32输出，各8条输入；每模式完整预热后两轮反序测量。所有模式共享相同prefill。当前源码含词表并行等后续修改，因此同时重测真实MTP和普通控制，不直接拿历史结果拼接。', '',
             '| 并发 | 普通Graph | 无草稿验证Graph | 逐请求MTP Graph | 批量草稿MTP Graph | 免费正确草稿Graph | 免费草稿相对普通 |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        r = row['rates']
        lines.append(f"| {row['args']['capacity']} | {r['native_graph']['decode_tok_s']:.3f} | {r['no_draft_graph']['decode_tok_s']:.3f} | {r['mtp_graph']['decode_tok_s']:.3f} | {r['mtp_batched_graph']['decode_tok_s']:.3f} | {r['oracle_graph']['decode_tok_s']:.3f} | {row['oracle_vs_native_pct']:+.2f}% |")
    lines += ['', '批量草稿相对逐请求草稿/普通Graph的变化：' + '；'.join(f"C{r['args']['capacity']}：{r['batch_draft_vs_serial_pct']:+.2f}% / {r['batch_draft_vs_native_pct']:+.2f}%" for r in rows) + '。', '', '单位为prefill后解码output tok/s；加载、捕获、预热不计时，全rank有限值检查计时。相同输出token数除以两轮累计耗时，不平均各轮速率。不同GPU组并行测量，CPU共享，非隔离环境。', '',
              '| 并发 | 普通目标行数 | 真实MTP目标行数 | 免费草稿目标行数 | 免费草稿接受/提出 |',
              '|---:|---:|---:|---:|---:|']
    for row in rows:
        r = row['rates']; oracle = r['oracle_graph']
        lines.append(f"| {row['args']['capacity']} | {r['native_graph']['target_rows']} | {r['mtp_graph']['target_rows']} | {oracle['target_rows']} | {oracle['accepted']}/{oracle['proposed']} |")
    lines += ['', '| 并发 | 普通Graph E2E | 逐请求MTP E2E | 批量草稿MTP E2E | 免费草稿 E2E |',
              '|---:|---:|---:|---:|---:|']
    for row in rows:
        r = row['rates']
        lines.append(f"| {row['args']['capacity']} | {r['native_graph']['e2e_tok_s']:.3f} | {r['mtp_graph']['e2e_tok_s']:.3f} | {r['mtp_batched_graph']['e2e_tok_s']:.3f} | {r['oracle_graph']['e2e_tok_s']:.3f} |")
    lines += ['', 'E2E按各cohort的相同prefill及decode时间之和计算，单位output tok/s；排除加载、捕获、预热及基准控制切换。不同并发配置的输出长度也不同，不用于单独归因并发扩展效率。']
    lines += ['', '| 并发 | 逐请求草稿前向次数 | 批量草稿前向次数 | 逐请求草稿耗时 s | 批量草稿耗时 s | 批量路径验证耗时 / 解码总耗时 s |',
              '|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        r = row['rates']; serial = r['mtp_graph']['counts_rank0']; batch = r['mtp_batched_graph']['counts_rank0']
        lines.append(f"| {row['args']['capacity']} | {serial['draft_forwards']} | {batch['draft_forwards']} | {serial['draft_seconds']:.3f} | {batch['draft_seconds']:.3f} | {batch['verification_seconds']:.3f} / {r['mtp_batched_graph']['decode_seconds']:.3f} |")
    lines += ['', '上表合并两轮，耗时为rank0包装器同步边界的wall time，包含对应函数内通信/有限值检查，不是纯GPU kernel profile，也不包含函数外全部IPC/事务开销。批量路径改变草稿GEMM形状，草稿可能变化；最终目标输出仍严格一致。']
    lines += ['', '免费草稿的参考前缀每次严格检查，所有提出token必须被接受；草稿模型前向和草稿IPC为零。仍保留Python查表/前缀检查开销，因此是当前路径下的乐观工程诊断，不是任意实现的数学性能上界。', '',
              f"正式共{sum(r['measured_generations'] for r in rows)}次生成、{sum(r['measured_output_tokens'] for r in rows)}个输出token；所有完整输出与各自普通Graph参考逐token一致，所有组资源回收、全部工作进程正常退出。原始计数、逐轮耗时、输出及源码快照见各目录result.json和source/。", '',
              '若免费草稿仍不超过普通Graph，说明在该配置下仅消除真实草稿成本尚无已证明的收益空间，应优先改进验证执行路径；它不能否定其他数值后端、多步窗口或其他负载。若出现小幅收益，仍需给真实草稿预留预算并复测，不能直接声称MTP已提速。', '',
              '本轮新增可选mtp_batch_draft=True：合并草稿投影、embedding/LM head和预测通信；attention仍按请求访问私有KV，草稿暂未捕获Graph。目标验证算术未改变。默认保持关闭。另修复事务分类依赖被包装函数名导致的多余S/conv快照，正式所有模式均基于修正后的同一源码。可选batched数值后端已有输出分歧，因此本诊断拒绝将固定普通输出作为该后端的完美草稿。']
    lines += ['', '旧插桩额外快照与一次无效故障注入的说明见 [测量修正说明](测量修正说明.md)。相关CUDA回归118项通过；多卡故障验证结果见synthetic_batch_fixed_tp2.json及synthetic_batch_tp4/8.json，须同时检查passed与processes_exited。']
    (args.root / '理想草稿诊断报告.md').write_text('\n'.join(lines) + '\n')
    for row in rows:
        print(f"C{row['args']['capacity']}: oracle/native={row['oracle_vs_native_pct']:+.2f}% mtp/native={row['mtp_vs_native_pct']:+.2f}%")


if __name__ == '__main__':
    main()
