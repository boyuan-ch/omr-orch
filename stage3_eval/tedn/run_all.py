"""
RAM-safe parallel TEDn runner for every page in manifest.json.

    python run_all.py                       # auto-size from this machine's RAM / cores
    python run_all.py --workers 32 --ram-budget-gb 120 --floor-gb 40
    python run_all.py --summarize-only      # rebuild the summary from results/ without running

How memory is kept safe
-----------------------
1. One page = one short-lived worker process (worker.py). When a page finishes, the process exits
   and ALL of its memory goes back to the OS -- nothing accumulates across pages.
2. Admission control: every page carries an estimated peak RAM (from its zss matrix size, see
   manifest.json "est_gb"). A page is only started if
       sum(est_gb of running pages) + est_gb(page) <= --ram-budget-gb
   AND the machine's *actual* available RAM minus est_gb stays above --floor-gb.
3. Per-page hard cap: each worker gets RLIMIT_AS, and the watchdog here samples every worker's RSS
   several times a second and kills any worker above --per-page-cap-gb.
4. Machine-wide emergency brake: if available RAM ever drops below --floor-gb (for any reason, e.g.
   another user's job), admission stops and the largest running workers are killed until available
   RAM is back above the floor. Killed pages are re-queued, not lost.
5. Resumable: a page whose results/<id>.json has status "ok" is never recomputed. Ctrl-C (or a
   crash) loses at most the pages that were running at that moment.

Pages are started largest-first so the slow long-tail pages begin early instead of being the last
stragglers; small pages fill the remaining slots.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict

import psutil

from memguard import memory_status

HERE = os.path.dirname(os.path.abspath(__file__))
GB = 1024 ** 3


def log(msg, fh=None):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


def tree_rss(proc):
    total = 0
    try:
        procs = [proc] + proc.children(recursive=True)
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


def load_result(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def summarize(manifest, results_dir, out_dir, fh=None):
    by_sym = defaultdict(list)
    missing = []
    for p in manifest["pages"]:
        r = load_result(os.path.join(results_dir, p["id"] + ".json"))
        if r and r.get("status") == "ok":
            by_sym[p["sym"]].append((p, r))
        else:
            missing.append((p["id"], (r or {}).get("status", "not_run")))

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    # per-movement files in the same shape as the original eval_mvt{m}.json
    for sym, items in sorted(by_sym.items()):
        per_mvt = defaultdict(list)
        for p, r in items:
            per_mvt[p["mvt"]].append({"page": p["page_in_mvt"], "global_page": p["page"],
                                      "average_TEDn": r["TEDn"], "all_TEDn": [r["TEDn"]]})
        for mvt, lst in per_mvt.items():
            lst.sort(key=lambda e: e["page"])
            with open(os.path.join(out_dir, f"Bee_{sym}_eval_onesys_mvt{mvt}.json"), "w") as f:
                json.dump(lst, f, indent=2)
        vals = [r["TEDn"] for _, r in items]
        expected = sum(1 for p in manifest["pages"] if p["sym"] == sym)
        rows.append(dict(sym=sym, pages_scored=len(vals), pages_expected=expected,
                         mean_TEDn=sum(vals) / len(vals)))

    with open(os.path.join(out_dir, "per_page.csv"), "w") as f:
        f.write("symphony,global_page,movement,page_in_mvt,TEDn,seconds,peak_rss_gb\n")
        for sym, items in sorted(by_sym.items()):
            for p, r in sorted(items, key=lambda x: x[0]["page"]):
                f.write(f"Bee_{sym},{p['page']},{p['mvt']},{p['page_in_mvt']},{r['TEDn']:.6f},"
                        f"{r.get('seconds', '')},{r.get('peak_rss_gb', '')}\n")

    all_vals = [r["TEDn"] for items in by_sym.values() for _, r in items]
    summary = dict(per_symphony=rows,
                   overall=dict(pages_scored=len(all_vals), pages_expected=len(manifest["pages"]),
                                mean_TEDn=(sum(all_vals) / len(all_vals)) if all_vals else None),
                   missing=missing)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("=" * 60, fh)
    log(f"{'sym':<8}{'scored':>12}{'mean TEDn':>14}", fh)
    for r in rows:
        log(f"Bee_{r['sym']:<4}{r['pages_scored']:>6}/{r['pages_expected']:<5}{r['mean_TEDn']:>14.4f}", fh)
    if all_vals:
        log(f"{'ALL':<8}{len(all_vals):>6}/{len(manifest['pages']):<5}{sum(all_vals) / len(all_vals):>14.4f}", fh)
    if missing:
        log(f"missing / failed pages: {len(missing)}  e.g. {missing[:5]}", fh)
    log(f"summary -> {out_dir}", fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(HERE, "manifest.json"))
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--summary-dir", default=os.path.join(HERE, "summary"))
    ap.add_argument("--impl", choices=["lean", "stock"], default="lean")
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (cores - 2)")
    ap.add_argument("--ram-budget-gb", type=float, default=0, help="0 = auto (50%% of total RAM)")
    ap.add_argument("--floor-gb", type=float, default=0, help="0 = auto (max(16, 15%% of total RAM))")
    ap.add_argument("--per-page-cap-gb", type=float, default=0, help="0 = auto (25%% of total RAM)")
    ap.add_argument("--only-sym", type=int, nargs="*", default=None)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--rlimit-factor", type=float, default=1.5,
                    help="worker RLIMIT_AS = per-page-cap * this (backstop behind the RSS watchdog)")
    ap.add_argument("--min-gold-nodes", type=int, default=0,
                    help="skip pages whose reference tree has fewer than N nodes after cut_xml; "
                         "the paper uses 1000 (a small denominator amplifies minor errors)")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    os.makedirs(args.results, exist_ok=True)
    fh = open(os.path.join(HERE, "run_all.log"), "a")

    if args.summarize_only:
        summarize(manifest, args.results, args.summary_dir, fh)
        return

    mem = memory_status()          # host RAM, or the container's cgroup limit if tighter
    total_gb = mem["total_gb"]
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    budget = args.ram_budget_gb or 0.50 * total_gb
    floor = args.floor_gb or max(16.0, 0.15 * total_gb)
    cap = args.per_page_cap_gb or 0.25 * total_gb
    impl_factor = 1.0 if args.impl == "lean" else manifest.get("stock_factor", 50.0)
    log(f"machine: {total_gb:.1f}GB RAM ({mem['source']}), {os.cpu_count()} cores | workers={workers} budget={budget:.0f}GB "
        f"floor={floor:.0f}GB per-page-cap={cap:.2f}GB impl={args.impl}", fh)
    if budget + floor > total_gb:
        sys.exit(f"budget ({budget}) + floor ({floor}) exceeds total RAM ({total_gb:.0f}GB); lower them")

    pages = [p for p in manifest["pages"] if args.only_sym is None or p["sym"] in args.only_sym]
    if args.min_gold_nodes:
        before = len(pages)
        pages = [p for p in pages if p.get("gold_raw_nodes", 10 ** 9) >= args.min_gold_nodes]
        log(f"page filter: kept {len(pages)}/{before} pages with >= {args.min_gold_nodes} reference nodes", fh)
    pending = []
    for p in pages:
        r = load_result(os.path.join(args.results, p["id"] + ".json"))
        if not (r and r.get("status") == "ok"):
            pending.append(p)
    pending.sort(key=lambda p: -p["cells"])       # largest first
    done0 = len(pages) - len(pending)
    log(f"pages: {len(pages)} total, {done0} already done, {len(pending)} to run", fh)

    tries = defaultdict(int)
    running = {}          # pid -> dict(page, popen, ps, est, t0, peak)
    stop = {"flag": False}

    def on_signal(signum, frame):
        stop["flag"] = True
        log(f"signal {signum}: stopping -- killing {len(running)} running workers (they will be re-run next time)", fh)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    finished = 0
    failed = []
    t_start = time.time()
    last_report = 0.0

    def est_gb(p):
        return p["est_gb"] * impl_factor

    def launch(p):
        out = os.path.join(args.results, p["id"] + ".json")
        meta = {k: p[k] for k in ("id", "sym", "page", "mvt", "page_in_mvt", "cells")}
        cmd = [sys.executable, os.path.join(HERE, "worker.py"), "--gt", os.path.join(HERE, p["gt"]),
               "--pred", os.path.join(HERE, p["pred"]), "--out", out, "--impl", args.impl,
               "--mem-cap-gb", str(cap * args.rlimit_factor), "--meta", json.dumps(meta)]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=HERE,
                                preexec_fn=lambda: os.nice(5))
        running[proc.pid] = dict(page=p, popen=proc, ps=psutil.Process(proc.pid), est=est_gb(p),
                                 t0=time.time(), peak=0)

    def kill(pid, reason):
        info = running.pop(pid)
        try:
            info["popen"].kill()
            info["popen"].wait(timeout=30)
        except Exception:
            pass
        p = info["page"]
        tries[p["id"]] += 1
        log(f"KILLED {p['id']} ({reason}); try {tries[p['id']]}/{args.max_retries + 1}", fh)
        if tries[p["id"]] <= args.max_retries:
            pending.append(p)     # retry later (it goes to the back, after other large pages)
        else:
            failed.append((p["id"], reason))

    while (pending or running) and not stop["flag"]:
        # ---- watchdog -------------------------------------------------------------------
        avail = memory_status()["available_gb"]
        for pid in list(running):
            info = running[pid]
            rss = tree_rss(info["ps"]) / GB
            info["peak"] = max(info["peak"], rss)
            if rss > cap:
                kill(pid, f"rss {rss:.2f}GB > per-page cap {cap:.2f}GB")
        if avail < floor and running:
            # emergency brake: kill biggest workers until we are back above the floor
            for pid in sorted(running, key=lambda q: -running[q]["peak"]):
                kill(pid, f"machine available {avail:.1f}GB < floor {floor:.0f}GB")
                time.sleep(1)
                avail = memory_status()["available_gb"]
                if avail >= floor + 8:
                    break

        # ---- reap finished workers --------------------------------------------------------
        for pid in list(running):
            info = running[pid]
            if info["popen"].poll() is None:
                continue
            running.pop(pid)
            p = info["page"]
            out = os.path.join(args.results, p["id"] + ".json")
            r = load_result(out) or {"status": "no_result"}
            r["peak_rss_gb"] = round(info["peak"], 3)
            r["wall_s"] = round(time.time() - info["t0"], 2)
            with open(out + ".tmp", "w") as f:
                json.dump(r, f)
            os.replace(out + ".tmp", out)
            if r.get("status") == "ok":
                finished += 1
            else:
                tries[p["id"]] += 1
                log(f"page {p['id']} status={r.get('status')} {r.get('error', '')[:160]}", fh)
                if r.get("status") == "oom" and tries[p["id"]] <= args.max_retries:
                    pending.append(p)
                else:
                    failed.append((p["id"], r.get("status")))

        # ---- admission -------------------------------------------------------------------
        avail = memory_status()["available_gb"]
        committed = sum(info["est"] for info in running.values())
        started = True
        while started and pending and len(running) < workers and avail >= floor:
            started = False
            for k, p in enumerate(pending):
                e = est_gb(p)
                if committed + e <= budget and avail - e >= floor:
                    pending.pop(k)
                    launch(p)
                    committed += e
                    avail -= e
                    started = True
                    break
            if not started and not running and pending:
                # nothing fits even on an idle machine: run the smallest one alone
                k = min(range(len(pending)), key=lambda q: est_gb(pending[q]))
                p = pending.pop(k)
                log(f"{p['id']} est {est_gb(p):.1f}GB exceeds budget; running it alone", fh)
                launch(p)
                break

        # ---- progress --------------------------------------------------------------------
        now = time.time()
        if now - last_report > 30:
            last_report = now
            el = now - t_start
            rate = finished / el if el > 0 else 0
            eta = (len(pending) + len(running)) / rate / 3600 if rate > 0 else float("nan")
            log(f"done {done0 + finished}/{len(pages)} | running {len(running)} "
                f"(est {committed:.1f}GB, rss {sum(i['peak'] for i in running.values()):.1f}GB) | "
                f"pending {len(pending)} | failed {len(failed)} | avail {avail:.0f}GB | "
                f"{rate * 60:.1f} pages/min | ETA {eta:.2f}h", fh)
        time.sleep(0.25)

    if stop["flag"]:
        for pid in list(running):
            try:
                running[pid]["popen"].kill()
            except Exception:
                pass
        log("stopped by signal; re-run the same command to resume", fh)
        return

    log(f"finished: {done0 + finished}/{len(pages)} ok, {len(failed)} failed {failed[:10]}", fh)
    summarize(manifest, args.results, args.summary_dir, fh)


if __name__ == "__main__":
    main()
