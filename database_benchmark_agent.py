"""
Database Benchmark AI Agent — sysbench + HammerDB (TPROC-C)
=============================================================
An AI agent that takes a natural language prompt describing a database
workload and autonomously:
  1. Translates it into a sysbench OLTP command sequence and a HammerDB
     TPROC-C TCL script pair (schema build + run)
  2. Runs sysbench end to end (prepare -> run -> cleanup)
  3. Generates the HammerDB scripts for you to run (see note below)
  4. Parses sysbench's output for TPS/QPS/latency, and HammerDB's NOPM/TPM
  5. Compares results side by side

HammerDB scripts are generated but NOT auto-executed by default: HammerDB
ships as a downloadable archive (no apt/yum package), bundles its own Tcl
interpreter, and its CLI (`hammerdbcli auto <script>.tcl`) needs to be
pointed at wherever you installed it — there's no single binary name this
agent can reliably find with `shutil.which` the way it can for sysbench.
If HAMMERDB_CLI_PATH is set to your local `hammerdbcli` binary, this agent
WILL execute it for real; otherwise it hands you a ready-to-run script pair
and the exact command to run them, which is honest given HammerDB's
installation model — better than claiming a benchmark ran when it didn't.

Supports: MySQL/MariaDB and PostgreSQL (sysbench's two common drivers here;
HammerDB additionally supports Oracle, SQL Server, Db2 — not wired up here
since this agent focuses on the two open-source engines AgentLease targets).

Set your Gemini key:
    export GOOGLE_API_KEY="AIza..."

Dry-run mode (no sysbench/HammerDB/Gemini needed):
    export DRY_RUN=true
    python database_benchmark_agent.py

Author : vytasta — https://vytasta.github.io/uberzunn/benchmark_agent
License: MIT
"""

import os
import json
import time
import random
import subprocess
import re
import functools
import socket
import shutil
import threading
from typing import TypedDict
from datetime import datetime

import requests

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ─────────────────────────────────────────────────────────────
# AGENTLEASE CONTROL-PLANE WIRING
# Set these to register this run as a data plane against a real
# AgentLease lease and heartbeat while the benchmark runs. If
# CONTROL_PLANE_URL / PAIRING_TOKEN aren't set, this is a no-op —
# the agent still runs exactly as it did standalone.
#
#   export CONTROL_PLANE_URL="https://your-control-plane.run.app"
#   export PAIRING_TOKEN="uz_live_..."     # from your lease
#   export DATA_PLANE_NAME="my-benchmark-host"   # optional, defaults to hostname
#   export DATA_PLANE_ENV="bare-metal"           # optional: kubernetes|openstack|bare-metal|other
# ─────────────────────────────────────────────────────────────

CONTROL_PLANE_URL = os.environ.get("CONTROL_PLANE_URL", "").rstrip("/")
PAIRING_TOKEN     = os.environ.get("PAIRING_TOKEN", "")
DATA_PLANE_NAME   = os.environ.get("DATA_PLANE_NAME", socket.gethostname())
DATA_PLANE_ENV    = os.environ.get("DATA_PLANE_ENV", "other")
HEARTBEAT_SECONDS = int(os.environ.get("HEARTBEAT_SECONDS", "30"))

_heartbeat_stop = threading.Event()


def _control_plane_enabled():
    return bool(CONTROL_PLANE_URL and PAIRING_TOKEN)


def register_data_plane():
    """Register this run against the control plane. Returns the data_plane
    id on success, or None if disabled/unreachable (the benchmark still
    runs either way — a control plane outage shouldn't block your work)."""
    if not _control_plane_enabled():
        return None
    try:
        resp = requests.post(
            f"{CONTROL_PLANE_URL}/dataplanes/register",
            json={
                "pairing_token": PAIRING_TOKEN,
                "name": DATA_PLANE_NAME,
                "environment": DATA_PLANE_ENV,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data_plane_id = resp.json()["id"]
        print(f"[control-plane] registered as data plane {data_plane_id} ({DATA_PLANE_NAME})")
        return data_plane_id
    except Exception as e:
        print(f"[control-plane] registration failed, continuing without it: {e}")
        return None


def _heartbeat_loop(data_plane_id):
    while not _heartbeat_stop.wait(HEARTBEAT_SECONDS):
        try:
            requests.post(
                f"{CONTROL_PLANE_URL}/dataplanes/{data_plane_id}/heartbeat",
                json={"pairing_token": PAIRING_TOKEN},
                timeout=10,
            )
        except Exception as e:
            print(f"[control-plane] heartbeat failed (will retry): {e}")


def start_heartbeat(data_plane_id):
    if not data_plane_id:
        return None
    t = threading.Thread(target=_heartbeat_loop, args=(data_plane_id,), daemon=True)
    t.start()
    print(f"[control-plane] heartbeating every {HEARTBEAT_SECONDS}s")
    return t


def stop_heartbeat():
    _heartbeat_stop.set()

# ─────────────────────────────────────────────────────────────
# DRY-RUN FLAG
# Set DRY_RUN=true to simulate the full agent without running
# sysbench/HammerDB or consuming Gemini tokens. Great for CI and demos.
# ─────────────────────────────────────────────────────────────

DRY_RUN = os.environ.get("DRY_RUN", "false").lower() == "true"

if DRY_RUN:
    print("=" * 65)
    print("DRY-RUN MODE — no real tools or LLM calls will be made")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────
# RATE LIMIT PROTECTION — throttle only
# A fixed delay between every LLM call keeps us safely under
# Gemini free tier (15 RPM). No retry loops needed when you
# stay under the limit in the first place.
# ─────────────────────────────────────────────────────────────

THROTTLE_SECONDS = 5        # 5s gap = max 12 RPM — safely under 15 RPM limit
_last_call_time  = 0.0


def throttle():
    """Enforce minimum gap between LLM calls."""
    global _last_call_time
    if DRY_RUN:
        return                  # no throttle needed in dry-run
    elapsed = time.time() - _last_call_time
    if elapsed < THROTTLE_SECONDS:
        wait = THROTTLE_SECONDS - elapsed
        print(f"  [throttle] waiting {wait:.1f}s to stay under rate limit...")
        time.sleep(wait)
    _last_call_time = time.time()


MAX_RATE_RETRIES = 5


def rate_limited(fn):
    """Decorator: throttle before every LLM call, and retry with exponential
    backoff on rate-limit (429) or transient server overload (503) errors."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RATE_RETRIES + 1):
            throttle()
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                err = str(e).lower()
                is_retryable = (
                    "429" in err
                    or "503" in err
                    or "rate limit" in err
                    or "resource_exhausted" in err
                    or "quota" in err
                    or "unavailable" in err
                    or "high demand" in err
                )
                if is_retryable and attempt < MAX_RATE_RETRIES:
                    wait = (2 ** attempt) * 10 + random.uniform(0, 2)
                    print(f"  [retry] {e.__class__.__name__} — retry {attempt + 1}/{MAX_RATE_RETRIES} in {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    raise
    return wrapper


# ─────────────────────────────────────────────────────────────
# AGENT STATE
# ─────────────────────────────────────────────────────────────

class BenchmarkState(TypedDict):
    messages:             list   # conversation history
    workload_params:      dict   # parsed workload parameters
    sysbench_command:     str    # generated sysbench prepare/run/cleanup command block
    hammerdb_script:      str    # generated HammerDB build+run TCL script pair
    sysbench_raw_output:  str    # raw stdout from sysbench's run phase
    hammerdb_raw_output:  str    # raw output from hammerdbcli, or the "not run" placeholder
    sysbench_results:     dict   # parsed sysbench TPS/QPS/latency
    hammerdb_results:     dict   # parsed HammerDB NOPM/TPM, if it was actually run
    comparison:           str    # side-by-side comparison summary
    error:                str    # last error if any
    final_answer:         str    # clean output for the user


# ─────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────

@tool
def generate_sysbench_command(
    db_driver: str,
    host: str,
    port: int,
    user: str,
    password: str,
    db_name: str,
    tables: int = 4,
    table_size: int = 100000,
    threads: int = 8,
    runtime_seconds: int = 60,
    test_name: str = "oltp_read_write",
) -> str:
    """
    Generate the sysbench prepare/run/cleanup command sequence for an OLTP
    workload against MySQL or PostgreSQL.

    Args:
        db_driver:        "mysql" or "pgsql"
        host:             database host
        port:             database port (3306 for mysql, 5432 for pgsql)
        user:             database user
        password:         database password
        db_name:          database/schema name (must already exist — sysbench
                           creates tables inside it, not the database itself)
        tables:           number of sbtest tables to create
        table_size:       rows per table
        threads:          concurrent client threads
        runtime_seconds:  duration of the run phase in seconds
        test_name:        sysbench Lua test, e.g. oltp_read_write, oltp_read_only,
                           oltp_write_only, oltp_point_select

    Returns:
        Three newline-separated command lines: prepare, run, cleanup. Each
        line is a complete, independently runnable sysbench invocation —
        run_sysbench executes them in order and keeps only the "run" phase's
        output (that's the one with real numbers; prepare/cleanup are just
        setup and teardown).
    """
    if DRY_RUN:
        return f"[DRY-RUN] sysbench {test_name} for {db_driver}://{host}:{port}/{db_name} | tables={tables} | threads={threads} | {runtime_seconds}s"

    host_flag = f"--{db_driver}-host={host}"
    port_flag = f"--{db_driver}-port={port}"
    user_flag = f"--{db_driver}-user={user}"
    pass_flag = f"--{db_driver}-password={password}"
    db_flag   = f"--{db_driver}-db={db_name}"

    common = (
        f"sysbench {test_name} --db-driver={db_driver} {host_flag} {port_flag} "
        f"{user_flag} {pass_flag} {db_flag} --tables={tables} --table-size={table_size}"
    )

    prepare_cmd = f"{common} prepare"
    run_cmd     = f"{common} --threads={threads} --time={runtime_seconds} --report-interval=0 run"
    cleanup_cmd = f"{common} cleanup"

    return "\n".join([prepare_cmd, run_cmd, cleanup_cmd])


@tool
def run_sysbench(command_block: str) -> str:
    """
    Execute the sysbench prepare/run/cleanup sequence from
    generate_sysbench_command. Prepare and cleanup are run for their side
    effects only; the run phase's stdout (the one with real throughput and
    latency numbers) is what's returned.

    Args:
        command_block: the 3-line prepare/run/cleanup block from
                        generate_sysbench_command

    Returns:
        Raw stdout from the run phase.
    """
    if DRY_RUN or command_block.startswith("[DRY-RUN]"):
        return (
            "SQL statistics:\n"
            "    queries performed:\n"
            "        read:                            68516\n"
            "        write:                           19202\n"
            "        other:                           9991\n"
            "        total:                           97709\n"
            "    transactions:                        4812   (450.75 per sec.)\n"
            "    queries:                             97709  (9152.53 per sec.)\n"
            "    ignored errors:                      82     (7.68 per sec.)\n"
            "    reconnects:                          0      (0.00 per sec.)\n"
            "\n"
            "General statistics:\n"
            "    total time:                          10.6740s\n"
            "    total number of events:              4812\n"
            "\n"
            "Latency (ms):\n"
            "         min:                                    1.20\n"
            "         avg:                                    8.86\n"
            "         max:                                 1016.07\n"
            "         95th percentile:                        6.91\n"
            "         sum:                                42652.09\n"
        )

    lines = [l for l in command_block.strip().splitlines() if l.strip()]
    if len(lines) != 3:
        return json.dumps({"error": f"expected 3 commands (prepare/run/cleanup), got {len(lines)}"})
    prepare_cmd, run_cmd, cleanup_cmd = lines

    try:
        print("  [sysbench] preparing tables...")
        prep = subprocess.run(prepare_cmd.split(), capture_output=True, text=True, timeout=300)
        if prep.returncode != 0:
            return json.dumps({"error": f"prepare failed: {prep.stderr or prep.stdout}"})

        print("  [sysbench] running benchmark...")
        run = subprocess.run(run_cmd.split(), capture_output=True, text=True, timeout=600)
        run_output = run.stdout

        print("  [sysbench] cleaning up...")
        subprocess.run(cleanup_cmd.split(), capture_output=True, text=True, timeout=300)

        if run.returncode != 0 and "SQL statistics" not in run_output:
            return json.dumps({"error": run.stderr or "sysbench run failed"})
        return run_output
    except FileNotFoundError:
        return json.dumps({"error": "sysbench not found. Install: sudo apt install sysbench (Debian/Ubuntu) or brew install sysbench (macOS)"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "sysbench timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def parse_sysbench_output(raw_output: str) -> str:
    """
    Parse sysbench's "SQL statistics" / "Latency (ms)" run-phase output.

    Args:
        raw_output: stdout from the sysbench run phase

    Returns:
        JSON string with reads/writes/total_queries, transactions, tps, qps,
        ignored_errors, and latency min/avg/max/p95 in ms. Fields that don't
        appear in the output (e.g. a read-only test has no writes) come back
        as null rather than a fabricated zero.
    """
    try:
        if "SQL statistics" not in raw_output and "error" in raw_output.lower():
            return json.dumps({"tool": "sysbench", "error": "sysbench run reported an error", "raw_snippet": raw_output[:300]})

        def _num(pattern, text, cast=float):
            m = re.search(pattern, text)
            return cast(m.group(1)) if m else None

        result = {
            "tool":            "sysbench",
            "read_queries":    _num(r"read:\s+(\d+)", raw_output, int),
            "write_queries":   _num(r"write:\s+(\d+)", raw_output, int),
            "total_queries":   _num(r"total:\s+(\d+)", raw_output, int),
            "transactions":    _num(r"transactions:\s+(\d+)", raw_output, int),
            "tps":             _num(r"transactions:\s+\d+\s+\(([\d.]+) per sec\.\)", raw_output),
            "qps":             _num(r"queries:\s+\d+\s+\(([\d.]+) per sec\.\)", raw_output),
            "ignored_errors":  _num(r"ignored errors:\s+(\d+)", raw_output, int),
            "latency_min_ms":  _num(r"min:\s+([\d.]+)", raw_output),
            "latency_avg_ms":  _num(r"avg:\s+([\d.]+)", raw_output),
            "latency_max_ms":  _num(r"max:\s+([\d.]+)", raw_output),
            "latency_p95_ms":  _num(r"95th percentile:\s+([\d.]+)", raw_output),
            "total_time_s":    _num(r"total time:\s+([\d.]+)s", raw_output),
        }
        if result["transactions"] is None:
            return json.dumps({"tool": "sysbench", "error": "could not find transaction stats in output", "raw_snippet": raw_output[:300]})
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"tool": "sysbench", "error": str(e), "raw_snippet": raw_output[:300]})


@tool
def generate_hammerdb_script(
    db_type: str,
    host: str,
    port: int,
    user: str,
    password: str,
    warehouses: int = 10,
    virtual_users: int = 4,
    runtime_minutes: int = 2,
    rampup_minutes: int = 1,
) -> str:
    """
    Generate a HammerDB TPROC-C (TPC-C-style) CLI script pair for MySQL/
    MariaDB or PostgreSQL: one script to build the schema, one to run the
    timed test.

    HammerDB is NOT auto-executed unless HAMMERDB_CLI_PATH points at a real
    hammerdbcli binary (see module docstring) — it has no package-manager
    install and no fixed binary name this agent can discover on its own, so
    by default this returns ready-to-run scripts and the exact command to
    invoke them rather than pretending to run something that likely isn't
    installed.

    Args:
        db_type:          "mysql" or "postgresql" (HammerDB's own names: this
                           agent maps "mysql"->mysql, "postgresql"->pg)
        host:              database host
        port:              database port
        user:              database user
        password:          database password
        warehouses:        TPC-C warehouse count (scales data size — bigger
                            warehouses = bigger dataset, more realistic but
                            slower to build)
        virtual_users:     number of HammerDB virtual users (concurrent
                            simulated clients)
        runtime_minutes:   timed test duration in minutes
        rampup_minutes:    ramp-up period before measurement starts, in minutes

    Returns:
        Two Tcl scripts concatenated with a marker line between them:
        "-- BUILD SCHEMA --" then the buildschema script, "-- RUN TEST --"
        then the run script. run_hammerdb splits on these markers.
    """
    if DRY_RUN:
        return f"[DRY-RUN] hammerdb {db_type} script for {host}:{port} | warehouses={warehouses} | vu={virtual_users} | {runtime_minutes}m"

    db_key = "mysql" if db_type.lower() in ("mysql", "mariadb") else "pg"

    build_script = f"""dbset db {db_key}
diset connection {db_key}_host {host}
diset connection {db_key}_port {port}
diset connection {db_key}_user {user}
diset connection {db_key}_password {password}
diset tpcc {db_key}_count_ware {warehouses}
diset tpcc {db_key}_num_vu {virtual_users}
buildschema
waittocomplete
quit
""".strip()

    run_script = f"""dbset db {db_key}
diset connection {db_key}_host {host}
diset connection {db_key}_port {port}
diset connection {db_key}_user {user}
diset connection {db_key}_password {password}
diset tpcc {db_key}_driver timed
diset tpcc {db_key}_rampup {rampup_minutes}
diset tpcc {db_key}_duration {runtime_minutes}
diset tpcc {db_key}_allwarehouses true
vuset logtotemp 1
loadscript
vuset vu {virtual_users}
vucreate
vurun
vudestroy
quit
""".strip()

    return f"-- BUILD SCHEMA --\n{build_script}\n-- RUN TEST --\n{run_script}"


@tool
def run_hammerdb(script_content: str) -> str:
    """
    Execute the HammerDB scripts from generate_hammerdb_script, if a real
    hammerdbcli is available via HAMMERDB_CLI_PATH. Otherwise returns the
    scripts as files ready to run manually, with the exact command.

    Args:
        script_content: the two-script block from generate_hammerdb_script

    Returns:
        Real hammerdbcli output if HAMMERDB_CLI_PATH is set and the binary
        runs successfully; otherwise a "not auto-executed" message
        containing the file paths written and the manual command to run.
    """
    if DRY_RUN or script_content.startswith("[DRY-RUN]"):
        return "TEST RESULT : System achieved 121771 NOPM from 283024 MySQL TPM"

    try:
        build_part, run_part = script_content.split("-- RUN TEST --")
        build_script = build_part.replace("-- BUILD SCHEMA --", "").strip()
        run_script   = run_part.strip()
    except ValueError:
        return json.dumps({"error": "malformed script_content — expected the two-script format from generate_hammerdb_script"})

    build_path = "/tmp/hammerdb_build.tcl"
    run_path   = "/tmp/hammerdb_run.tcl"
    with open(build_path, "w") as f:
        f.write(build_script)
    with open(run_path, "w") as f:
        f.write(run_script)

    hammerdb_cli = os.environ.get("HAMMERDB_CLI_PATH", "")
    if not hammerdb_cli:
        return (
            f"[NOT AUTO-EXECUTED] HammerDB scripts written to {build_path} and {run_path}.\n"
            f"HammerDB has no package-manager install this agent can rely on — download it from "
            f"https://www.hammerdb.com/download.html, then either set HAMMERDB_CLI_PATH to your "
            f"hammerdbcli binary and re-run, or run manually:\n"
            f"  <hammerdb-dir>/hammerdbcli auto {build_path}\n"
            f"  <hammerdb-dir>/hammerdbcli auto {run_path}"
        )

    if shutil.which(hammerdb_cli) is None and not os.path.isfile(hammerdb_cli):
        return json.dumps({"error": f"HAMMERDB_CLI_PATH is set to '{hammerdb_cli}' but that binary was not found"})

    try:
        print("  [hammerdb] building schema (this can take a while for larger warehouse counts)...")
        build = subprocess.run([hammerdb_cli, "auto", build_path], capture_output=True, text=True, timeout=3600)
        if build.returncode != 0:
            return json.dumps({"error": f"schema build failed: {build.stderr or build.stdout}"})

        print("  [hammerdb] running timed TPROC-C test...")
        run = subprocess.run([hammerdb_cli, "auto", run_path], capture_output=True, text=True, timeout=3600)
        return run.stdout or run.stderr
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "hammerdbcli timed out"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def parse_hammerdb_output(raw_output: str) -> str:
    """
    Parse HammerDB's "TEST RESULT" line for NOPM and TPM, or recognize the
    "not auto-executed" placeholder and pass it through as such rather than
    reporting a fake result.

    Args:
        raw_output: output from run_hammerdb

    Returns:
        JSON string with nopm and tpm on a real run, or
        {"tool": "hammerdb", "not_run": true, "note": "..."} if HammerDB
        wasn't actually executed.
    """
    if raw_output.startswith("[NOT AUTO-EXECUTED]"):
        return json.dumps({"tool": "hammerdb", "not_run": True, "note": raw_output})

    try:
        match = re.search(
            r"System achieved\s+(\d+)\s+NOPM from\s+(\d+)\s+\w*\s*TPM",
            raw_output,
        )
        if not match:
            return json.dumps({"tool": "hammerdb", "error": "could not find TEST RESULT line", "raw_snippet": raw_output[:300]})
        return json.dumps({
            "tool": "hammerdb",
            "nopm": int(match.group(1)),
            "tpm":  int(match.group(2)),
        })
    except Exception as e:
        return json.dumps({"tool": "hammerdb", "error": str(e), "raw_snippet": raw_output[:300]})


ALL_TOOLS = [
    generate_sysbench_command, run_sysbench, parse_sysbench_output,
    generate_hammerdb_script, run_hammerdb, parse_hammerdb_output,
]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ─────────────────────────────────────────────────────────────
# LLM SETUP
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a database benchmarking assistant. Given a natural
language description of a database workload, drive it through sysbench and
generate matching HammerDB scripts, in this order:

1. Call generate_sysbench_command, then run_sysbench, then parse_sysbench_output.
2. Call generate_hammerdb_script, then run_hammerdb, then parse_hammerdb_output.

Call ONE tool at a time, in the order above: sysbench triple first, then
HammerDB triple.

Default values if not specified:
  db_driver/db_type: mysql | tables: 4 | table_size: 100000 | threads: 8
  runtime_seconds: 60 | test_name: oltp_read_write
  warehouses: 10 | virtual_users: 4 | runtime_minutes: 2 | rampup_minutes: 1

Note: HammerDB will only actually execute if HAMMERDB_CLI_PATH is set on
this host — otherwise run_hammerdb hands back ready-to-run scripts instead
of fabricated results. Don't claim HammerDB numbers you didn't get.
"""


def _make_llm():
    """Create LLM — returns a stub in dry-run mode."""
    if DRY_RUN:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    ).bind_tools(ALL_TOOLS)


def _make_llm_plain():
    if DRY_RUN:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        temperature=0,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )


LLM       = _make_llm()
LLM_PLAIN = _make_llm_plain()


# ─────────────────────────────────────────────────────────────
# DRY-RUN SIMULATION — scripted tool sequence, no LLM needed
# ─────────────────────────────────────────────────────────────

DRY_RUN_SEQUENCE = [
    ("generate_sysbench_command", {"db_driver": "mysql", "host": "127.0.0.1", "port": 3306, "user": "root", "password": "x", "db_name": "sbtest", "tables": 4, "table_size": 100000, "threads": 8, "runtime_seconds": 60}),
    ("run_sysbench",              {"command_block": "[DRY-RUN] sysbench command"}),
    ("parse_sysbench_output",     {"raw_output": (
        "SQL statistics:\n    queries performed:\n        read: 68516\n        write: 19202\n        other: 9991\n        total: 97709\n"
        "    transactions: 4812 (450.75 per sec.)\n    queries: 97709 (9152.53 per sec.)\n    ignored errors: 82 (7.68 per sec.)\n"
        "General statistics:\n    total time: 10.6740s\nLatency (ms):\n     min: 1.20\n     avg: 8.86\n     max: 1016.07\n     95th percentile: 6.91\n"
    )}),
    ("generate_hammerdb_script",  {"db_type": "mysql", "host": "127.0.0.1", "port": 3306, "user": "root", "password": "x", "warehouses": 10, "virtual_users": 4, "runtime_minutes": 2, "rampup_minutes": 1}),
    ("run_hammerdb",              {"script_content": "[DRY-RUN] hammerdb script"}),
    ("parse_hammerdb_output",     {"raw_output": "TEST RESULT : System achieved 121771 NOPM from 283024 MySQL TPM"}),
]

_dry_run_step = 0


class _DryRunMessage(AIMessage):
    """Mimics an AIMessage with tool_calls for dry-run mode."""
    def __init__(self, tool_name, tool_args, call_id):
        super().__init__(
            content=f"[DRY-RUN] calling {tool_name}",
            tool_calls=[{"name": tool_name, "args": tool_args, "id": call_id}],
        )


# ─────────────────────────────────────────────────────────────
# GRAPH NODES
# ─────────────────────────────────────────────────────────────

@rate_limited
def _llm_invoke(messages):
    return LLM.invoke(messages)


def planner_node(state: BenchmarkState) -> dict:
    """LLM (or dry-run stub) decides the next tool to call."""
    global _dry_run_step

    if DRY_RUN:
        if _dry_run_step >= len(DRY_RUN_SEQUENCE):
            return {"messages": state["messages"] + [AIMessage(content="[DRY-RUN] all tools complete", tool_calls=[])]}

        tool_name, tool_args = DRY_RUN_SEQUENCE[_dry_run_step]
        _dry_run_step += 1
        print(f"  [dry-run] step {_dry_run_step}/{len(DRY_RUN_SEQUENCE)}: {tool_name}")
        time.sleep(0.3)
        msg = _DryRunMessage(tool_name, tool_args, f"dry-call-{_dry_run_step}")
        return {"messages": state["messages"] + [msg]}

    messages = state["messages"]
    if len(messages) == 1:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    response = _llm_invoke(messages)
    return {"messages": state["messages"] + [response]}


def tool_executor_node(state: BenchmarkState) -> dict:
    """Execute whatever tool call the planner just requested and fold its
    result into state."""
    last = state["messages"][-1]
    new_messages = []
    updates = {}

    for tc in last.tool_calls:
        fn = TOOL_MAP[tc["name"]]
        result = fn.invoke(tc["args"])
        new_messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        if tc["name"] == "generate_sysbench_command":
            updates["sysbench_command"] = result
        elif tc["name"] == "generate_hammerdb_script":
            updates["hammerdb_script"] = result
        elif tc["name"] == "run_sysbench":
            updates["sysbench_raw_output"] = result
        elif tc["name"] == "run_hammerdb":
            updates["hammerdb_raw_output"] = result
        elif tc["name"] == "parse_sysbench_output":
            try:    updates["sysbench_results"] = json.loads(result)
            except: updates["sysbench_results"] = {"raw": result}
        elif tc["name"] == "parse_hammerdb_output":
            try:    updates["hammerdb_results"] = json.loads(result)
            except: updates["hammerdb_results"] = {"raw": result}

    return {"messages": new_messages, **updates}


@rate_limited
def _compare_invoke(messages):
    return LLM_PLAIN.invoke(messages)


def comparison_node(state: BenchmarkState) -> dict:
    """Generate a side-by-side summary of sysbench vs HammerDB results.
    HammerDB's numbers (NOPM/TPM) aren't directly comparable to sysbench's
    (TPS/QPS/latency) — different workload shapes (TPC-C vs generic OLTP) —
    so this reports both rather than forcing a misleading apples-to-apples
    table the way the storage agent can for IOPS/throughput."""
    sysbench = state.get("sysbench_results", {})
    hammerdb = state.get("hammerdb_results", {})
    hammerdb_ran = bool(hammerdb) and not hammerdb.get("not_run")

    if DRY_RUN:
        hammerdb_line = (
            f"HammerDB TPROC-C: {hammerdb.get('nopm', 'N/A')} NOPM / {hammerdb.get('tpm', 'N/A')} TPM"
            if hammerdb_ran else
            "HammerDB: scripts generated but not auto-executed — see hammerdb_script in the saved results and run manually."
        )
        answer = f"""
DATABASE BENCHMARK SUMMARY  [DRY-RUN SIMULATION]
====================================================
sysbench (generic OLTP, oltp_read_write)
-----------------------------------------
Transactions/sec (TPS) : {sysbench.get('tps', 'N/A')}
Queries/sec (QPS)       : {sysbench.get('qps', 'N/A')}
Total transactions      : {sysbench.get('transactions', 'N/A')}
Avg latency (ms)        : {sysbench.get('latency_avg_ms', 'N/A')}
95th pct latency (ms)   : {sysbench.get('latency_p95_ms', 'N/A')}

HammerDB (TPC-C style)
-----------------------------------------
{hammerdb_line}

ANALYSIS
--------
[DRY-RUN] sysbench's oltp_read_write and HammerDB's TPROC-C are different
workload shapes (generic key-value OLTP vs the TPC-C order-entry schema),
so treat these as two independent signals rather than one number to
reconcile — a system can lead on one and trail on the other.

RECOMMENDATION
--------------
Use sysbench for quick, repeatable regression checks (fast to prepare/run/
cleanup). Use HammerDB's TPC-C result when you need an industry-recognized
number to quote externally.
""".strip()
        return {"comparison": answer, "final_answer": answer}

    hammerdb_note = (
        f"HammerDB achieved {hammerdb.get('nopm')} NOPM / {hammerdb.get('tpm')} TPM."
        if hammerdb_ran else
        "HammerDB was not auto-executed on this host (no HAMMERDB_CLI_PATH set) — "
        "its build/run scripts were generated and saved for the user to run manually."
    )

    prompt = f"""
You are a database performance expert. Summarize these benchmark results
for a database engineer. sysbench and HammerDB run different workload
shapes (generic OLTP vs TPC-C order-entry), so do NOT force them into one
comparison table — report each clearly, then comment on what they each
suggest.

sysbench (oltp_read_write) results:
{json.dumps(sysbench, indent=2)}

HammerDB (TPROC-C) status:
{hammerdb_note}

Format:
DATABASE BENCHMARK SUMMARY
==============================
sysbench: TPS, QPS, avg/p95 latency
HammerDB: NOPM/TPM if available, or the manual-run note if not

ANALYSIS
--------
2-3 sentences on what the numbers suggest, noting they're not directly comparable.

RECOMMENDATION
--------------
When to use which tool going forward.
"""
    response = _compare_invoke([{"role": "user", "content": prompt}])
    return {"comparison": response.content, "final_answer": response.content}


# ─────────────────────────────────────────────────────────────
# EDGE ROUTING
# ─────────────────────────────────────────────────────────────

def route_after_planner(state: BenchmarkState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "execute_tool"
    return "compare"


def route_after_tool(state: BenchmarkState) -> str:
    sysbench_done = bool(state.get("sysbench_results"))
    hammerdb_done = bool(state.get("hammerdb_results"))
    if sysbench_done and hammerdb_done:
        return "compare"
    return "planner"


# ─────────────────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────

def build_benchmark_agent():
    graph = StateGraph(BenchmarkState)
    graph.add_node("planner",      planner_node)
    graph.add_node("execute_tool", tool_executor_node)
    graph.add_node("compare",      comparison_node)

    graph.set_entry_point("planner")
    graph.add_conditional_edges("planner", route_after_planner, {
        "execute_tool": "execute_tool",
        "compare":      "compare",
    })
    graph.add_conditional_edges("execute_tool", route_after_tool, {
        "planner": "planner",
        "compare": "compare",
    })
    graph.add_edge("compare", END)
    return graph.compile(checkpointer=MemorySaver())


# ─────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────

def run_benchmark_agent(user_prompt: str):
    global _dry_run_step
    _dry_run_step = 0

    agent = build_benchmark_agent()
    config = {"configurable": {"thread_id": "db-benchmark-1"}}

    initial_state = {
        "messages":            [HumanMessage(content=user_prompt)],
        "workload_params":     {},
        "sysbench_command":    "",
        "hammerdb_script":     "",
        "sysbench_raw_output": "",
        "hammerdb_raw_output": "",
        "sysbench_results":    {},
        "hammerdb_results":    {},
        "comparison":          "",
        "error":               "",
        "final_answer":        "",
    }

    print("\n" + "=" * 65)
    print("DATABASE BENCHMARK AI AGENT")
    print("=" * 65)
    print(f"Prompt : {user_prompt}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode   : {'DRY-RUN (simulated)' if DRY_RUN else 'LIVE'}")
    print("=" * 65 + "\n")

    result = agent.invoke(initial_state, config=config)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(result.get("final_answer", "(no comparison produced)"))
    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def check_prerequisites():
    """
    Check for sysbench BEFORE calling the LLM at all — a live run with it
    missing would still spend a real Gemini call just to have the model
    report back that the binary was missing, something `shutil.which`
    answers instantly and for free.

    HammerDB is deliberately NOT checked here: it has no fixed binary name
    or package-manager install (see module docstring), so there's nothing
    reliable for shutil.which to look for. Its availability is decided at
    call time by whether HAMMERDB_CLI_PATH is set.
    """
    if shutil.which("sysbench") is not None:
        return
    print("ERROR: required benchmark tool not found on this host: sysbench")
    print("  Install sysbench:")
    print("    Ubuntu/Debian:  sudo apt install sysbench")
    print("    RHEL/CentOS:    sudo yum install sysbench")
    print("    macOS:          brew install sysbench")
    print("Or test the pipeline without real tools: export DRY_RUN=true")


if __name__ == "__main__":
    if not DRY_RUN and not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set.")
        print("Get a free key at https://aistudio.google.com")
        print("Or run in dry-run mode: export DRY_RUN=true")
        exit(1)

    if not DRY_RUN:
        check_prerequisites()
        if shutil.which("sysbench") is None:
            exit(1)

    data_plane_id = register_data_plane()
    start_heartbeat(data_plane_id)

    # ── Pick a prompt ──────────────────────────────────────
    prompt = (
        "Run an OLTP read/write benchmark against the MySQL database at "
        "127.0.0.1:3306, db sbtest, user root, 4 tables of 100000 rows, "
        "8 threads, 60 seconds. Also generate a HammerDB TPROC-C script "
        "with 10 warehouses and 4 virtual users."
    )
    # prompt = "Run a read-only OLTP benchmark against Postgres at 127.0.0.1:5432, db sbtest, user postgres, 16 threads, 120 seconds."

    try:
        result = run_benchmark_agent(prompt)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = {
            "prompt":            prompt,
            "timestamp":         timestamp,
            "dry_run":           DRY_RUN,
            "sysbench_command":  result.get("sysbench_command"),
            "hammerdb_script":   result.get("hammerdb_script"),
            "sysbench_results":  result.get("sysbench_results"),
            "hammerdb_results":  result.get("hammerdb_results"),
            "comparison":        result.get("comparison"),
        }
        out_path = f"/tmp/db_benchmark_results_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nFull results saved to: {out_path}")
    finally:
        stop_heartbeat()
