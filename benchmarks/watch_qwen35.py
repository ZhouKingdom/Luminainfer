"""Persistent, finite Qwen3.5 experiment queue on genuinely idle shared GPUs.

init freezes sources and inputs; run survives chat exit when launched detached.
STOP stops current owned process group; PAUSE lets it finish before waiting.
GPU locks coordinate only other copies of this runner, not external schedulers.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from zoneinfo import ZoneInfo

import qwen35_baseline as bench


ROOT = Path(__file__).resolve().parents[1]
PYTHONS = {"lumina": "/home/data/jhzhou/conda_envs/lumina/bin/python",
           "vllm": "/home/data/jhzhou/conda_envs/vllm/bin/python"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def matrix():
    cases = []
    def add(tp, capacity, inp, out, kind="graph", requests=None, replica=0):
        cases.append({"id": f"{kind}_TP{tp}_C{capacity}_I{inp}_O{out}_N{requests or capacity}_R{replica}",
                      "tp": tp, "capacity": capacity, "input": inp, "output": out,
                      "requests": requests or capacity, "kind": kind, "replica": replica,
                      "repeats": 1 if kind in ("smoke", "profile") else 3})
    for tp in (2, 1, 4, 8):
        add(tp, 1, 32, 8, "smoke")
    for tp in (2, 4, 1, 8):
        for inp, out in ((128, 128), (2048, 128), (128, 512)):
            for capacity in (1, 8, 32, 128):
                add(tp, capacity, inp, out)
    add(2, 32, 128, 128, replica=1)
    for inp, out in ((128, 128), (2048, 128), (128, 512)):
        for capacity in (1, 32):
            add(2, capacity, inp, out, "eager")
    for tp in (2, 4):
        add(tp, 32, 128, 32, "tail", requests=1024)
    for inp in (128, 2048):
        for capacity in (1, 32):
            add(2, capacity, inp, 32, "profile")
    return cases


def package_versions(python):
    code = "import importlib.metadata as m,json; print(json.dumps({n:m.version(n) for n in ['torch','transformers','triton']}))"
    result = json.loads(subprocess.check_output([python, "-c", code], text=True, timeout=30))
    if python == PYTHONS["vllm"]:
        result["vllm"] = subprocess.check_output([python, "-c", "import importlib.metadata as m;print(m.version('vllm'))"], text=True, timeout=30).strip()
    return result


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def init(args):
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = directory/"source"
    for folder in ("zjhvllm", "benchmarks"):
        if folder == "zjhvllm":
            shutil.copytree(ROOT/folder, source/folder, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            (source/folder).mkdir()
            for name in ("qwen35_baseline.py", "qwen35_profile_summary.py", "watch_qwen35.py"):
                shutil.copyfile(ROOT/folder/name, source/folder/name)
    bench.prepare(argparse.Namespace(model=args.model, requests=1024, input_tokens=2048,
                  output_tokens=512, seed=0, texts=None, output=directory/"input_pool.json"))
    pool = json.loads((directory/"input_pool.json").read_text())
    cases = matrix()
    (directory/"workloads").mkdir()
    for i, case in enumerate(cases):
        workload = pool | {"requests": [{**r, "input_ids": r["input_ids"][:case["input"]], "max_tokens": case["output"]}
                                       for r in pool["requests"][:case["requests"]]]}
        path = directory/"workloads"/(case["id"]+".json")
        bench.write_json(path, workload)
        case["workload"] = str(path)
        case["workload_sha256"] = bench.digest(path)
        case["order"] = ["lumina", "vllm"] if (i+case["replica"]) % 2 == 0 else ["vllm", "lumina"]
        case["status"] = "pending"
        case["attempts"] = []
    # Ensure the repeated performance anchor runs in the opposite engine order.
    anchor = [c for c in cases if c["kind"] == "graph" and c["tp"] == 2 and c["capacity"] == 32
              and c["input"] == 128 and c["output"] == 128]
    anchor[1]["order"] = list(reversed(anchor[0]["order"]))
    plan = {"schema": "qwen35-idle-queue-v1", "created": now(), "model": str(Path(args.model).resolve()),
            "pythons": PYTHONS, "cases": cases, "source": str(source),
            "expected_packages": {engine: package_versions(python) for engine, python in PYTHONS.items()},
            "gpu_lock_root": str(ROOT/"benchmarks/.qwen35_gpu_locks"),
            "source_sha256": {str(p.relative_to(source)): bench.digest(p) for p in source.rglob("*.py")},
            "poll_seconds": 30, "idle_samples": 4, "max_used_mib": 256, "max_util_percent": 5,
            "max_job_seconds": 2700, "min_free_disk_gib": 8,
            "note": "65 paired settings, 130 engine runs; finite queue, waits indefinitely for idle GPUs; no performance claims until executed"}
    bench.write_json(directory/"plan.json", plan)
    update(directory, plan, "created")
    print(json.dumps({"directory": str(directory), "cases": len(cases), "engine_runs": 2*len(cases)}))


def gpu_snapshot():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=15)
    gpus = []
    for row in text.splitlines():
        index, uuid, used, util = [s.strip() for s in row.split(",")]
        gpus.append({"index": int(index), "uuid": uuid, "used_mib": int(used), "util": int(util), "pids": []})
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
    mapping = {g["uuid"]: g for g in gpus}
    for row in text.splitlines():
        uuid, pid = [s.strip() for s in row.split(",")]
        if uuid in mapping:
            mapping[uuid]["pids"].append(int(pid))
    return sorted(gpus, key=lambda g: g["index"])


def idle(gpu, plan):
    return not gpu["pids"] and gpu["used_mib"] <= plan["max_used_mib"] and gpu["util"] <= plan["max_util_percent"]


def process_group_pids(pgid):
    rows = subprocess.check_output(["ps", "-eo", "pid=,pgid=,sid="], text=True, timeout=10)
    return {int(pid) for pid, group, session in (r.split() for r in rows.splitlines())
            if int(group) == pgid and int(session) == pgid}


def tagged_processes(token, proc_root=Path("/proc")):
    """Identify this launch even when Nsight creates new sessions/reparents.

    Never infer ownership from command names. Check a random per-launch marker
    inherited in the environment and stable Linux start time around the read.
    """
    found = {}
    marker = ("LUMINA_QWEN35_JOB_TOKEN="+token).encode()
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            before = (entry/"stat").read_text().rsplit(")", 1)[1].split()[19]
            env = (entry/"environ").read_bytes().split(b"\0")
            after = (entry/"stat").read_text().rsplit(")", 1)[1].split()[19]
            if marker in env and before == after:
                found[int(entry.name)] = before
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return found


def owned_gpu_pids(process):
    if hasattr(process, "_watch_token"):
        return set(owned_process_records(process))
    return process_group_pids(process.pid)


def process_records(proc_root=Path("/proc")):
    records = {}
    for entry in proc_root.iterdir():
        if entry.name.isdigit():
            try:
                fields = (entry/"stat").read_text().rsplit(")", 1)[1].split()
                if fields[0] != "Z":
                    records[int(entry.name)] = (int(fields[1]), fields[19])
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    return records


def owned_process_records(process):
    records = process_records()
    previous = getattr(process, "_watch_owned", {})
    if hasattr(process, "_watch_start_ticks"):
        previous = previous | {process.pid: process._watch_start_ticks}
    trusted = {pid: ticks for pid, ticks in previous.items() if records.get(pid, (None, None))[1] == ticks}
    for pid, ticks in tagged_processes(process._watch_token).items():
        if records.get(pid, (None, None))[1] == ticks:
            trusted[pid] = ticks
    # Nsight can setsid/reparent; setproctitle can also relocate environment
    # memory. Retain proven descendants by PID+start time across both changes.
    while True:
        children = {pid: ticks for pid, (parent, ticks) in records.items()
                    if pid not in trusted and parent in trusted and int(ticks) >= int(trusted[parent])}
        if not children:
            break
        trusted.update(children)
    process._watch_owned = trusted
    return trusted


def stop_owned_group(process):
    if hasattr(process, "_watch_token"):
        # Individual PID signals cover detached Nsight workers. Revalidate
        # proven ownership + start time before every signal (PID reuse).
        for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
            members = owned_process_records(process)
            for pid in sorted(members, key=lambda p: p == process.pid):
                if owned_process_records(process).get(pid) != members[pid]:
                    continue
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic()+seconds
            while time.monotonic() < deadline:
                if not owned_process_records(process):
                    return
                time.sleep(.25)
        return
    # All signals target this Popen(start_new_session=True) session only.
    # A normal/abnormal parent exit may leave its own TP workers in that group.
    for sig, seconds in ((signal.SIGTERM, 10), (signal.SIGKILL, 5)):
        identity = Path(f"/proc/{process.pid}/stat")
        if identity.exists() and hasattr(process, "_watch_start_ticks"):
            try:
                ticks = identity.read_text().rsplit(")", 1)[1].split()[19]
            except FileNotFoundError:
                ticks = None
            if ticks is not None and ticks != process._watch_start_ticks:
                return  # PID was reused; never signal an unrelated session.
        if not process_group_pids(process.pid):
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if not process_group_pids(process.pid):
                return
            time.sleep(.25)


def choose_group(gpus, streaks, count, required, locked=None):
    locked = locked or set()
    eligible = [g for g in gpus if streaks.get(g["uuid"], 0) >= required and g["uuid"] not in locked]
    return next(itertools.combinations(eligible, count), None)


def event(directory, kind, **data):
    with (directory/"events.jsonl").open("a") as f:
        f.write(json.dumps({"time": now(), "kind": kind, **data}, ensure_ascii=False)+"\n")


def update(directory, plan, phase, **extra):
    bench.write_json(directory/"plan.json", plan)
    counts = {status: sum(c["status"] == status for c in plan["cases"])
              for status in sorted({c["status"] for c in plan["cases"]})}
    bench.write_json(directory/"status.json", {"updated": now(), "pid": os.getpid(), "phase": phase,
                     "counts": counts, **extra})
    lines = ["# Qwen3.5 自动实验进度", "", f"更新：{now()}；状态：{phase}", "",
             "吞吐是离线 burst 端到端 output tok/s；profile 不参与性能排名。未执行的配置没有性能结论。", "",
             "| 配置 | 状态 | Lumina tok/s | vLLM tok/s | 比值 L/V | 输出一致 |", "|---|---|---:|---:|---:|---|"]
    for c in plan["cases"]:
        result = c.get("comparison")
        if result:
            values = f"{result['lumina_output_tok_s']:.2f} | {result['vllm_output_tok_s']:.2f} | {result['lumina_over_vllm']:.3f} | {result['exact_sequences']}/{result['sequences']}"
        else:
            values = "— | — | — | —"
        lines.append(f"| {c['id']} | {c['status']} | {values} |")
    (directory/"实验汇总.md").write_text("\n".join(lines)+"\n")


def child_env(plan, group, directory):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g["index"]) for g in group)
    env["PYTHONPATH"] = plan["source"]
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # Caches in the data-volume environment, avoiding the nearly full root fs.
    cache = "/home/data/jhzhou/conda_envs/vllm/.benchmark-cache"
    env["VLLM_CACHE_ROOT"] = cache+"/vllm"
    env["TRITON_CACHE_DIR"] = cache+"/triton"
    env["TORCHINDUCTOR_CACHE_DIR"] = cache+"/inductor"
    return env


def execute(directory, plan, argv, env, log, group, timeout=None, watch_foreign=True):
    event(directory, "launch", argv=argv, gpu_indices=[g["index"] for g in group], log=str(log))
    with log.open("w") as stream:
        env = env.copy()
        token = uuid.uuid4().hex
        env["LUMINA_QWEN35_JOB_TOKEN"] = token
        process = subprocess.Popen(argv, cwd=plan["source"], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        process._watch_token = token
        try:
            process._watch_start_ticks = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            pass
        started = time.monotonic()
        try:
            while process.poll() is None:
                if (directory/"STOP").exists():
                    return {"returncode": None, "reason": "stop_requested", "pid": process.pid}
                if time.monotonic()-started > (timeout or plan["max_job_seconds"]):
                    return {"returncode": None, "reason": "timeout", "pid": process.pid}
                if shutil.disk_usage(directory).free < plan["min_free_disk_gib"]*1024**3:
                    return {"returncode": None, "reason": "disk_low", "pid": process.pid}
                if watch_foreign:
                    # Query GPU first, then process membership: a just-spawned
                    # worker isn't mistakenly treated as an external process.
                    observed = gpu_snapshot()
                    owned = owned_gpu_pids(process)
                    selected = {g["uuid"] for g in group}
                    outsiders = [pid for g in observed if g["uuid"] in selected for pid in g["pids"] if pid not in owned]
                    if outsiders:
                        return {"returncode": None, "reason": "foreign_gpu_process", "foreign_pids": outsiders, "pid": process.pid}
                time.sleep(5)
            return {"returncode": process.returncode, "reason": "exited", "pid": process.pid,
                    "seconds": time.monotonic()-started}
        finally:
            stop_owned_group(process)
            process.wait(timeout=10)


def run_case(directory, plan, case, group):
    attempt = directory/"runs"/case["id"]/f"attempt_{len(case['attempts'])+1:02d}"
    attempt.mkdir(parents=True)
    info = {"path": str(attempt), "started": now(), "gpu_indices": [g["index"] for g in group],
            "gpu_uuids": [g["uuid"] for g in group], "engines": {}}
    case["attempts"].append(info)
    case["status"] = "running"
    update(directory, plan, "running", case=case["id"], attempt=str(attempt))
    env = child_env(plan, group, directory)
    for engine in case["order"]:
        if package_versions(plan["pythons"][engine]) != plan["expected_packages"][engine]:
            raise RuntimeError("environment package versions changed: "+engine)
        selected = [g for g in gpu_snapshot() if g["uuid"] in info["gpu_uuids"]]
        if len(selected) != len(group) or not all(idle(g, plan) for g in selected):
            case["status"] = "pending"
            info["reason"] = "busy_before_engine"
            return
        output = attempt/engine
        argv = [plan["pythons"][engine], str(Path(plan["source"])/"benchmarks/qwen35_baseline.py"), "run",
                "--engine", engine, "--model", plan["model"], "--workload", case["workload"], "--output", str(output),
                "--tp", str(case["tp"]), "--capacity", str(case["capacity"]), "--max-model-len", "4096",
                "--token-budget", "2048", "--memory", ".85", "--warmups", "1", "--repeats", str(case["repeats"]),
                "--port", str(free_port())]
        if case["kind"] == "eager":
            argv.append("--eager")
        if case["kind"] == "profile":
            argv.append("--profile")
            argv = ["/usr/local/cuda/bin/nsys", "profile", "--trace=cuda,nvtx,osrt", "--trace-fork-before-exec=true",
                    "--cuda-graph-trace=node", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
                    "--output="+str(attempt/(engine+"_trace")), *argv]
        outcome = execute(directory, plan, argv, env, attempt/(engine+".log"), group)
        info["engines"][engine] = outcome
        if outcome["reason"] in ("foreign_gpu_process", "stop_requested", "disk_low"):
            info["reason"] = outcome["reason"]
            case["status"] = "pending"
            return
        path = output/"result.json"
        result = json.loads(path.read_text()) if path.exists() else {}
        if (outcome["returncode"] != 0 or not result.get("complete")
                or not result.get("repeat_outputs_equal") or not result.get("shutdown_called")):
            case["status"] = "failed"
            info["reason"] = "engine_failure"
            info["error"] = result.get("error", "see engine log")
            return
        # Explicitly wait for worker/context teardown without considering it
        # part of the measured generation time.
        for _ in range(12):
            if (directory/"STOP").exists():
                case["status"] = "pending"
                info["reason"] = "stop_requested"
                return
            selected = [g for g in gpu_snapshot() if g["uuid"] in info["gpu_uuids"]]
            if len(selected) == len(group) and all(idle(g, plan) for g in selected):
                break
            time.sleep(5)
        else:
            case["status"] = "pending"
            info["reason"] = "busy_after_engine"
            return
        update(directory, plan, "running", case=case["id"], engine_completed=engine)
    if case["kind"] != "profile":
        bench.compare(argparse.Namespace(lumina=attempt/"lumina/result.json", vllm=attempt/"vllm/result.json", output=attempt/"comparison.json"))
        case["comparison"] = json.loads((attempt/"comparison.json").read_text())
    else:
        coverage = []
        for engine in case["order"]:
            database = attempt/(engine+"_trace.sqlite")
            argv = ["/usr/local/cuda/bin/nsys", "export", "--type=sqlite", "--output="+str(database), str(attempt/(engine+"_trace.nsys-rep"))]
            outcome = execute(directory, plan, argv, env, attempt/(engine+"_export.log"), group, timeout=600, watch_foreign=False)
            if outcome["returncode"] != 0:
                case["status"] = "profile_incomplete"
                info["reason"] = "trace_export_failed"
                return
            argv = [plan["pythons"]["lumina"], str(Path(plan["source"])/"benchmarks/qwen35_profile_summary.py"),
                    str(database), "--output", str(attempt/(engine+"_profile.json"))]
            outcome = execute(directory, plan, argv, env, attempt/(engine+"_summary.log"), group, timeout=120, watch_foreign=False)
            if outcome["returncode"] != 0:
                case["status"] = "profile_incomplete"
                info["reason"] = "trace_summary_failed"
                return
            data = json.loads((attempt/(engine+"_profile.json")).read_text())
            coverage.append(len(data["devices"]) >= case["tp"])
        if not all(coverage):
            case["status"] = "profile_incomplete"
            info["reason"] = "missing_tp_device_trace"
            return
    case["status"] = "succeeded"
    info["finished"] = now()


def run(args):
    directory = args.directory.resolve()
    plan = json.loads((directory/"plan.json").read_text())
    lock = (directory/"runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (directory/"runner.pid").write_text(str(os.getpid())+"\n")
    def request_stop(*_):
        (directory/"STOP").write_text("Stop requested by signal.\n")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    for name, sha in plan["source_sha256"].items():
        if bench.digest(Path(plan["source"])/name) != sha:
            raise RuntimeError("frozen source changed: "+name)
    for c in plan["cases"]:
        if bench.digest(c["workload"]) != c["workload_sha256"]:
            raise RuntimeError("frozen workload changed: "+c["id"])
        if c["status"] == "running":
            # Never adopt/kill an unknown orphan from a previous daemon.
            c["status"] = "pending"
            (directory/"PAUSE").write_text("Previous runner interrupted mid-case; inspect owned/orphan processes before resuming.\n")
    streaks = {}
    try:
        while True:
            if (directory/"STOP").exists():
                update(directory, plan, "stopped")
                return
            if (directory/"PAUSE").exists():
                update(directory, plan, "paused")
                time.sleep(plan["poll_seconds"])
                continue
            if shutil.disk_usage(directory).free < plan["min_free_disk_gib"]*1024**3:
                update(directory, plan, "waiting_disk")
                time.sleep(plan["poll_seconds"])
                continue
            try:
                gpus = gpu_snapshot()
            except Exception as error:
                event(directory, "gpu_query_failed", error=repr(error))
                update(directory, plan, "waiting_gpu_query", error=repr(error))
                time.sleep(plan["poll_seconds"])
                continue
            for g in gpus:
                streaks[g["uuid"]] = streaks.get(g["uuid"], 0)+1 if idle(g, plan) else 0
            ready = []
            for c in plan["cases"]:
                if c["status"] != "pending":
                    continue
                smoke = next(s for s in plan["cases"] if s["kind"] == "smoke" and s["tp"] == c["tp"])
                if c["kind"] == "smoke" or smoke["status"] == "succeeded":
                    ready.append(c)
            if not any(c["status"] == "pending" for c in plan["cases"]):
                update(directory, plan, "finished")
                return
            choice = next(((c, group) for c in ready if (group := choose_group(gpus, streaks, c["tp"], plan["idle_samples"]))), None)
            if choice is None:
                update(directory, plan, "waiting_gpu", gpus=gpus, idle_streaks=streaks)
                time.sleep(plan["poll_seconds"])
                continue
            case, group = choice
            handles = []
            try:
                locks = Path(plan["gpu_lock_root"])
                locks.mkdir(exist_ok=True)
                for g in group:
                    handle = (locks/(g["uuid"]+".lock")).open("w")
                    handles.append(handle)
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                run_case(directory, plan, case, group)
                event(directory, "case_finished", case=case["id"], status=case["status"])
                # A bootstrap failure needs inspection, not 60 more model loads.
                if case["kind"] == "smoke" and case["status"] == "failed":
                    (directory/"PAUSE").write_text("Bootstrap failed; inspect runs/ logs before removing PAUSE.\n")
                elif case["status"] == "pending" and len(case["attempts"]) >= 3:
                    (directory/"PAUSE").write_text("Three interrupted attempts; inspect logs before removing PAUSE.\n")
                update(directory, plan, "waiting_gpu")
            except BlockingIOError:
                time.sleep(plan["poll_seconds"])
            except Exception as error:
                event(directory, "runner_error", case=case["id"], error=repr(error))
                case["status"] = "failed"
                (directory/"PAUSE").write_text("Runner error; inspect events.jsonl before removing PAUSE.\n")
                update(directory, plan, "paused", error=repr(error))
            finally:
                for handle in handles:
                    handle.close()
                streaks = {}
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init")
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--model", default="/home/data/jhzhou/huggingface/Qwen3.5-27B")
    p.set_defaults(function=init)
    p = sub.add_parser("run")
    p.add_argument("--directory", type=Path, required=True)
    p.set_defaults(function=run)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
