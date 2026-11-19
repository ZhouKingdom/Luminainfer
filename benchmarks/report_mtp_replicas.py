"""Report all measured paths, keeping numerical rejection separate from speed."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory
    status = json.loads((directory / "status.json").read_text())
    report = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if status["state"] != "complete" or report["requests"] != status["planned_generations"]:
        raise SystemExit("refusing to report an incomplete experiment as complete")
    for name, expected in manifest["source_hashes"].items():
        if hashlib.sha256((directory / "source" / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source snapshot checksum mismatch: {name}")
    modes = report["modes"]
    validation = {"measurement_complete": True, "requests": report["requests"],
                  "all_resources_reclaimed": all(m["resources_reclaimed"] for m in modes.values()),
                  "baseline_cross_repeat_passed": not report["baseline_cross_repeat_mismatches"],
                  "modes": {name: {"exact": m["exact_requests"], "paired": m["paired_requests"],
                                     "numerical_passed": m["exact_requests"] == m["requests"] == m["paired_requests"]}
                            for name, m in modes.items()}}
    validation["numerical_validation_passed"] = all(m["numerical_passed"] for m in validation["modes"].values())
    (directory / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    lines = ["# Qwen3.5-27B MTP-1 八卡扩大对照", "",
             f"已完成 {report['requests']} 次生成：128 个不同输入 × 两轮 × 三模式。8×A100 80GB，独立副本、TP=1、每卡活跃并发1、总活跃并发8，原生引擎 closed-loop。",
             "", "## 逐 token 验收与指标", "",
             "| 模式 | 严格一致请求 | output tok/s | TTFT p99 (ms) | 平均 TPOT p99 (ms) | E2E p99 (s) | 草稿接受率 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    batch_label = ("MTP 批量验证（数值通过）" if validation["modes"]["batched_experimental"]["numerical_passed"]
                   else "MTP 批量实验（未验收）")
    labels = {"baseline": "普通 decode", "serial": "MTP 串行参考", "batched_experimental": batch_label}
    for mode, label in labels.items():
        item = modes[mode]
        acceptance = f"{item['acceptance_rate']:.2%}" if item["acceptance_rate"] is not None else "—"
        lines.append(f"| {label} | {item['exact_requests']}/{item['paired_requests']} | "
                     f"{item['output_tokens_per_second']:.2f} | {1000*item['ttft_seconds']['p99']:.2f} | "
                     f"{1000*item['tpot_seconds']['p99']:.2f} | {item['e2e_seconds']['p99']:.3f} | {acceptance} |")
    lines += ["", "同 `(repeat, request_id)` 配对普通 decode；基线自身一列是自比较，跨轮比较单独列在下文。",
              "串行参考沿用普通 decode 内核、没有投机目标前向节省。批量路径只要存在输出失配，整组吞吐和延迟差值都不验收为有效加速。",
              "", "## 批量实验分歧", "", "| 输入 | 配对请求 | 输出分歧 |", "|---|---:|---:|"]
    for category, values in modes["batched_experimental"]["by_category"].items():
        lines.append(f"| {category} | {values['requests']} | {values['mismatches']} |")
    rows = [json.loads(line) for phase in report["phases"] for path in directory.glob(phase["name"] + ".gpu*.jsonl")
            for line in path.read_text().splitlines()]
    cross = {}
    for mode in labels:
        by_repeat = {(r["repeat"], r["id"]): r for r in rows if r["mode"] == mode}
        ids = {r["id"] for r in rows if r["mode"] == mode}
        cross[mode] = {"output_differences": sum(by_repeat[(0, i)]["token_ids"] != by_repeat[(1, i)]["token_ids"] for i in ids),
                       "different_gpu": sum(by_repeat[(0, i)]["gpu"] != by_repeat[(1, i)]["gpu"] for i in ids)}
    lines += ["", "## 复现与资源", "",
              "第二轮轮换物理 GPU、反转 baseline/batched 测试顺序：", ""]
    for mode, value in cross.items():
        lines.append(f"- {labels[mode]}：{value['different_gpu']}/128 条换卡；两轮输出分歧 {value['output_differences']}/128。")
    reclaimed = sum(r["resources_reclaimed"] for r in rows)
    tokens = sum(r["output_tokens"] for r in rows)
    lines += [f"- 输出总计 {tokens:,} token，KV 页/状态槽完整回收 {reclaimed}/{len(rows)} 次。", "",
              "## 方法边界", "",
              "输入为本地 QA 问题34条、AIME-2025题干30条、函数续写32条、技术文档长上下文32条；输入20–2076 token，输出上限64/128，正常遵守EOS。无参考答案、无oracle草稿。",
              "吞吐使用统一屏障到最后请求完成的模式 wall time，合并两轮；基线也驻留 MTP 权重。",
              "每模式256请求，p99为经验分位数，不是生产SLA。平均TPOT按每请求首末输出跨度计算；批次交付间隔不等于逐token ITL。",
              "未测试HTTP/网络、高并发C128、TP=8或1→8副本扩展效率。本测试不能代替HF logits或任务准确率验收。", "",
              "## 原始证据", "",
              "[统计](summary.json)、[数值验收](validation.json)、[固定输入](workload.json)、[清单与源码SHA256](manifest.json)、[进程记录](processes.json)、[最终状态](status.json)、[源码快照](source/)。",
              "每张卡每个阶段的 JSONL 包含完整 token、时间、MTP接受及重放计数。输出分歧没有过滤、阈值没有放宽。", ""]
    if manifest.get("verification_arithmetic") == "tokenwise_decode_shapes":
        lines += ["批量窗口沿用逐 token 的投影、norm、递归和 attention 算子。目标模型调用次数减少，"
                  "不代表矩阵计算或 kernel 次数相同比例减少；性能以本报告实测值为准。", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
    checksums = {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "checksums.json"}
    (directory / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(json.dumps({"requests": len(rows), "output_tokens": tokens, "reclaimed": reclaimed,
                      "cross_repeat": cross}, indent=2))


if __name__ == "__main__":
    main()
