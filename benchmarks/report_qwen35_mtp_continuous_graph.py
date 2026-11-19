"""Validate raw TP Graph/refill evidence and report comparable output rates."""
import hashlib
import ast
import json
from pathlib import Path

root=Path('benchmarks/qwen35_mtp_continuous_graph_20261107')
runs={2:'final_tp2',4:'verified_tp4',8:'verified_tp8'}
summary={'passed':True,'regression':'162 passed, 0 skipped','generations':0,'output_tokens':0,'tp':[]}
rows=[]
late_source_changes=set()
class Normalize(ast.NodeTransformer):
    def generic_visit(self,node):
        super().generic_visit(node)
        if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
        if isinstance(node,ast.ClassDef) and node.name=='ModelRunner':
            for child in node.body:
                if isinstance(child,ast.FunctionDef) and child.name=='exit':child.body=[ast.Pass()]
        if isinstance(node,ast.Module):
            node.body=[child for child in node.body if not (isinstance(child,ast.Import) and len(child.names)==1 and child.names[0].name=='gc')]
        return node

def compute_ast(file):return ast.dump(Normalize().visit(ast.parse(file.read_text())),include_attributes=False)
for tp,name in runs.items():
    run=root/name;x=json.loads((run/'result.json').read_text());assert x['complete'] and x['finite_checks_rank0']>0
    for rel,sha in x['source_sha256'].items():
        file=run/'source'/rel;assert hashlib.sha256(file.read_bytes()).hexdigest()==sha
        if rel.startswith('zjhvllm/') and Path(rel).read_bytes()!=file.read_bytes():
            assert rel in ('zjhvllm/engine/model_runner.py','zjhvllm/config.py','zjhvllm/models/qwen3_5.py')
            assert compute_ast(Path(rel))==compute_ast(file),rel
            late_source_changes.add(rel)
    synthetic=json.loads((root/f'synthetic_exit_tp{tp}.json').read_text())
    assert synthetic['passed'] and synthetic['processes_exited']
    assert all('graph_allocate' in rank['rollback_faults'] for rank in synthetic['ranks'])
    phases={p['mode']:p for p in x['phases']}
    for phase in x['phases']:
        assert phase['exact']==phase['paired']==8 and phase['reclaimed']
        summary['generations']+=phase['paired'];summary['output_tokens']+=sum(map(len,phase['output_ids']))
        if phase['mode'].startswith('continuous') or phase['mode']=='graph_warmup':
            assert phase['refill_before_initial_cohort_finished']
        rows.append({'tp':tp,**{k:phase[k] for k in ('mode','exact','reference','output_tok_s','graph_rounds','refill_before_initial_cohort_finished') if k in phase}})
    measured=[phases['continuous_graph'],phases['continuous_graph_repeat']]
    graph_rate=sum(sum(map(len,p['output_ids'])) for p in measured)/sum(p['seconds'] for p in measured)
    for p in measured:
        assert p['graph_rounds']>0 and p['output_ids']==phases['continuous_eager']['output_ids']==phases['ordinary']['output_ids']
        ranks=p['ranks'];assert len(ranks)==tp
        assert all(r['verification_graphs']==ranks[0]['verification_graphs'] for r in ranks)
        assert all(len(r['verification_graphs'])<=16 for r in ranks)
    summary['tp'].append({'tp':tp,'ordinary_eager_tok_s':phases['ordinary']['output_tok_s'],
        'continuous_mtp_eager_tok_s':phases['continuous_eager']['output_tok_s'],
        'cohort_mtp_eager_tok_s':phases['cohort_eager']['output_tok_s'],
        'graph_cold_tok_s':phases['graph_warmup']['output_tok_s'],
        'continuous_graph_pooled_tok_s':graph_rate,
        'graph_vs_continuous_eager_pct':(graph_rate/phases['continuous_eager']['output_tok_s']-1)*100,
        'graph_vs_ordinary_eager_pct':(graph_rate/phases['ordinary']['output_tok_s']-1)*100,
        'captured_layouts':len(measured[-1]['ranks'][0]['verification_graphs']),
        'graph_request_rounds_per_measurement':[p['graph_rounds'] for p in measured]})
summary['rows']=rows
summary['compute_ast_matches_frozen_runtime']=True
summary['late_source_changes']=sorted(late_source_changes)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# 多卡验证 Graph 与连续补位验收','',
    '162项相关CUDA回归通过，无skip。TP2/4/8分布式小模型均验证Graph动态请求、混合接受[1,0]、单rank准备/提交/草稿/Graph缓冲分配错误后的回滚和继续运行。',
    '', '真实Qwen3.5-27B、8×A100-80GB、TP2/4/8、默认decode_equivalent、BF16、eager普通decode、显式验证Graph。8条输入长度36/25/106/128/64/96/255/256，输出预算8/48/16/32/12/24/9/17，总输出166 token/阶段，最大并发4。',
    f"共{summary['generations']}次正式生成、{summary['output_tokens']}个输出token，所有请求回收页和状态槽。96次连续MTP（eager、Graph冷启动、Graph预热后两轮）全部与相同TP普通解码逐token一致。另24次固定cohort使用对应普通控制。所有rank每次有效logits输出检查有限。",
    '', '| TP | 普通eager E2E | 连续MTP eager E2E | Graph冷启动E2E | 连续MTP Graph预热后E2E | Graph布局数 |','|---:|---:|---:|---:|---:|---:|']
for t in summary['tp']:
    lines.append(f"| {t['tp']} | {t['ordinary_eager_tok_s']:.3f} | {t['continuous_mtp_eager_tok_s']:.3f} | {t['graph_cold_tok_s']:.3f} | {t['continuous_graph_pooled_tok_s']:.3f} | {t['captured_layouts']} |")
lines+=['', '单位output tok/s。Graph预热后合并两轮wall time计算；捕获成本单列在冷启动阶段，不计入预热后数字。E2E含prefill和全rank有限值检查。TP2/4分组运行、TP8随后测量，早期小模型验证在退出时留有空闲CUDA上下文（后已修复并清理）；共享主机非隔离环境，这是小集功能与延迟负载，不是C128容量或投机算法独立收益评测。',
    '', '每个连续阶段均记录第5个请求在最初4个请求全部完成前开始生成；固定cohort对应控制为False。分布式各rank的Graph布局/重放计数完全一致，包含混合1/2行尾部窗口；精确动态slots/pages/positions和页255–257由CUDA单测覆盖。',
    '', 'TP4初始固定cohort与连续普通基线有2条跨调度输出差异，保留final_tp4失败记录。新增同cohort普通解码控制后严格比较通过；未放宽连续Graph/eager逐token验收。见INITIAL_NOTES.md。',
    '', '每组源码快照、SHA256、完整输出、逐轮接受、调度轨迹和资源/rank元数据随result.json保存；报告逐文件核对冻结源码SHA256；当前计算路径的AST与正式快照一致。计时后只加强了exit里的gc回收/通信同步，并改了两处说明文字；exit在E2E计时外，已用实际生产exit重跑TP2/4/8故障恢复测试并确认全部子进程退出。',
    '', '使用和边界见 [使用指南](../../使用指南.md)。仍greedy、MTP-1、显式1–8请求；FP8/更宽窗口/超过16种布局回退eager。本轮真实容量4，未宣称C8/C128验收或HTTP服务自动MTP。']
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
