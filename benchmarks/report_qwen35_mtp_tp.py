"""Summarize only post-fix distributed MTP checkpoints and raw output parity."""
import hashlib
import ast
import json
from pathlib import Path

root=Path('benchmarks/qwen35_mtp_tp_20261107')
rows=[];requests=outputs=0
text_only_differences=set()
class StripDocstrings(ast.NodeTransformer):
    def generic_visit(self,node):
        super().generic_visit(node)
        if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                node.body.pop(0)
        return node

def executable_ast(path):
    return ast.dump(StripDocstrings().visit(ast.parse(path.read_text())),include_attributes=False)
for tp in (2,4,8):
    run=root/f'fixed_tp{tp}'
    result=json.loads((run/'result.json').read_text())
    assert result['complete'] and result['finite_checks_rank0']>0
    assert all(r['mtp']['loaded_groups']==15 for r in result['ranks'])
    assert all(hashlib.sha256((run/'source'/name).read_bytes()).hexdigest()==sha for name,sha in result['sha256'].items())
    for name,sha in result['sha256'].items():
        if name.startswith('zjhvllm/'):
            current=Path(name)
            if hashlib.sha256(current.read_bytes()).hexdigest()!=sha:
                assert executable_ast(current)==executable_ast(run/'source'/name),name
                text_only_differences.add(name)
    synthetic=json.loads((root/f'synthetic_fixed_tp{tp}.json').read_text());assert synthetic['passed']
    for phase in result['phases']:
        assert phase['exact']==phase['paired'] and phase['reclaimed']
        requests+=phase['paired'];outputs+=sum(map(len,phase['output_ids']))
        rows.append({'tp':tp,**{k:phase[k] for k in ('concurrency','mode','exact','paired','output_tok_s','accepted','proposed')}})
summary={'passed':True,'generations':requests,'output_tokens':outputs,'rows':rows,
    'comparison':'same engine/TP/numerical backend, per concurrency; finite checks on all ranks; output IDs and errors retained',
    'performance_limit':'small correctness workload, long prompts and 32-token outputs; E2E includes prefill, finite checks and coordination; no isolated speedup claim',
    'regression':'155 passed, 0 skipped','runtime_matches_current_ast':True,'text_only_differences':sorted(text_only_differences)}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 多卡 MTP-1 验收','',
    '真实 Qwen3.5-27B，8×A100-80GB，TP2/4/8，默认 decode_equivalent、BF16、eager、greedy、ignore_eos。',
    '归档输入 ID 2/39/76/127，长度36/106/990/2076，各输出32 token。每个 TP 测 C1 ordinary/serial/batched、C2与C4 ordinary/batched。',
    f'共 {requests} 次正式生成、{outputs} 个输出 token；其中48次MTP生成全部与相同配置普通输出逐 token 一致，36次普通生成作为参考，页与状态槽全部回收。各 rank 的15组MTP权重完整加载、每次LM head输出都检查有限。',
    '', '| TP | 并发 | 模式 | 一致请求 | E2E output tok/s | 草稿接受/提议 |','|---:|---:|---|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['tp']} | {row['concurrency']} | {row['mode']} | {row['exact']}/{row['paired']} | {row['output_tok_s']:.3f} | {row['accepted']}/{row['proposed']} |")
lines.extend(['','这是正确性测试，包含长prompt分块prefill和有限值检查，不代表C128容量或纯decode吞吐。TP2/4使用互不重叠GPU组并行运行，TP8随后单独执行，不是隔离环境强扩展评测。不据此宣称投机加速。',
    '', '155项相关CUDA回归通过，无skip；分布式小模型在TP2/4/8验证混合接受[1,0]、精确状态提交、单rank准备/草稿/提交错误后的全rank回滚及继续运行。',
    '', '初版TP4批量数值验收失败，原记录完整保留，见 [INITIAL_FAILURE.md](INITIAL_FAILURE.md)。正式记录仅使用fixed_tp*目录，其源码快照和SHA256随result.json保存；每个配置原始输出、每轮接受统计、rank局部cache形状均可复查。',
    '', '隔离NCCL实验：TP4相同BF16局部分量，[2,5120]整体归约与两次[1,5120]逐行归约，10240元素中3299个不同、最大差0.0625；不运行GEMM，原始数据见nccl_shape_tp4.json。',
    '', '源码冻结后只更新了模型模块docstring和配置注释，已逐文件剥离docstring后比较AST，执行代码与三组正式快照完全一致。',
    '', '尚未实现：多卡MTP验证Graph、投机连续补位、随机拒绝采样。仅本轮C1/C2/C4、默认后端完成真实模型验收；容量配置仍允许1–8，不宣称C8已独立验收。'])
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'generations':requests,'output_tokens':outputs,'passed':True}))
