# Database Benchmark Agent

An AI agent that runs a `sysbench` OLTP benchmark against MySQL or
PostgreSQL from a plain-English workload description, and generates a
ready-to-run `HammerDB` TPROC-C (TPC-C style) script pair — via LangGraph +
Gemini.

> Rename this file to `README.md` in its own repo (`database_benchmark_agent.py`
> lives alongside `storage_benchmark_agent.py` and `network_benchmark_agent.py`
> in this output bundle, but each is meant to be its own GitHub repo).

## Prerequisites — install this BEFORE running for real

**sysbench:**
```bash
# Ubuntu/Debian
sudo apt install sysbench

# RHEL/CentOS
sudo yum install sysbench

# macOS
brew install sysbench
```

You also need a MySQL or PostgreSQL server reachable from this host, with a
database already created (sysbench creates tables inside it, not the
database itself) and a user with permission to create/drop tables there.

**Want to test the agent itself without sysbench or a database?** Use
dry-run mode instead (see below).

## About HammerDB

HammerDB is **not auto-executed by default**. Unlike sysbench, it has no
package-manager install and no fixed binary name this agent can reliably
find — it ships as a downloadable archive with its own bundled Tcl
interpreter. By default, this agent generates the two scripts HammerDB
needs (a schema-build script and a timed-run script, using real `dbset`/
`diset`/`buildschema`/`vucreate`/`vurun` syntax) and hands you the exact
command to run them yourself:

```bash
<hammerdb-dir>/hammerdbcli auto /tmp/hammerdb_build.tcl
<hammerdb-dir>/hammerdbcli auto /tmp/hammerdb_run.tcl
```

If you want the agent to run HammerDB for you, download HammerDB
(https://www.hammerdb.com/download.html), then set:

```bash
export HAMMERDB_CLI_PATH="/path/to/hammerdbcli"
```

With that set, the agent runs both scripts for real and parses the
resulting NOPM/TPM numbers from HammerDB's own `TEST RESULT` line.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

**Dry-run (no tools, no database, no API key, no cost):**
```bash
export DRY_RUN=true
python3 database_benchmark_agent.py
```

**Live run (real sysbench, real Gemini calls):**
```bash
export GOOGLE_API_KEY="your-key"   # free tier: https://aistudio.google.com
python3 database_benchmark_agent.py
```

Edit the `prompt` variable near the bottom of the file (in the
`if __name__ == "__main__":` block) to describe your own workload — db
driver (mysql/pgsql), host/port/user/password/db name, table count and
size, thread count, duration, and HammerDB warehouse/virtual-user counts.

## AgentLease control-plane integration (optional)

If you're running this agent as part of an AgentLease lease rather than
standalone, set these before running so it registers itself and
heartbeats:

```bash
export CONTROL_PLANE_URL="https://your-control-plane-url"
export PAIRING_TOKEN="uz_live_..."   # from your lease
export DATA_PLANE_NAME="my-host"     # optional, defaults to hostname
```

Leave these unset to run the agent completely standalone — it works
identically either way; the control-plane calls are a no-op without them.

## Rate limits

Gemini's free tier has both per-minute and per-day request limits. This
agent throttles and retries automatically on transient rate-limit (429)
and server-overload (503) errors, but a genuinely exhausted **daily**
quota can't be retried around — check https://ai.dev/rate-limit if a run
keeps failing with `RESOURCE_EXHAUSTED` despite several retries.

## Output

Full results (sysbench command, HammerDB scripts, parsed metrics for both,
and the comparison summary) are saved as JSON to
`/tmp/db_benchmark_results_<timestamp>.json` after every run.

Note: sysbench's `oltp_read_write` and HammerDB's TPROC-C measure different
workload shapes (generic key-value OLTP vs. the TPC-C order-entry schema),
so the comparison summary deliberately reports them separately rather than
forcing them into one misleading table.
