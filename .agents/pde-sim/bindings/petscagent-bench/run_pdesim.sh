#!/usr/bin/env bash
# Run one or more pde-sim configs over one or more benchmark problems with no
# state carried between configs, so no config's score can be influenced by
# another's.
#
# Before each config this clears the three things that leak between runs:
#   artifacts/          code generation cribs a prior solution from a study dir
#   case-index.json     the pde-sim case index remembers earlier cases
#   purple_agent_cache/ Green serves a recorded Purple response from here
# Nothing is deleted: prior results and study dirs move into $ARCHIVE.
#
# Usage: ./run_pdesim.sh [-p|--problems LIST] [-c|--configs LIST] [-m|--model NAME]
#                        [-i|--progress SECS] [-b|--bench DIR] [-o|--output DIR]
#
# PETSC_DIR and PETSC_ARCH must both be set: they select the PETSc build, and the
# pde-sim binding and artifacts/ are taken from PETSC_DIR. The bench repo is a
# separate checkout, so it comes from --bench, else $PETSCAGENT_BENCH, else
# ~/petscagent-bench.
#
#   -p, --problems LIST  comma-separated problem terms, as main.py --problems
#                        takes them, e.g. darcy or darcy,robertson (default advection)
#   -c, --configs LIST   comma-separated pde-sim configs, e.g. 1,2   (default 1)
#   -m, --model NAME     model the Purple generates with, as PDESIM_CLAUDE_MODEL
#                        takes it, e.g. claudeopus48 or 'claudeopus48[1m]'
#   -i, --progress SECS  seconds between progress lines, 0 to silence (default 30)
#   -o, --output DIR     where Green writes results, as main.py --output takes it
#                        (default $BENCH/output/pdesim_<timestamp>, one per run)
#
# -p takes a glob, so -p '*' is the full matrix: ./run_pdesim.sh -p '*' -c 1,2,3
#
# Quote a model id containing brackets, or the shell globs it away:
#   -m 'claudeopus48[1m]'   Opus 4.8 with a 1M context window rather than 200K.
# The bare id takes the 200K default, and a pde-sim run has been seen to peak at
# ~220K tokens of context, so the 1M form is what avoids mid-run compaction.
#
# Each run writes to its own output directory, so results never collide with an
# earlier run's and nothing has to be parked first. Green puts the aggregate JSON
# there, one per config, beside the code/ and scores/ trees it writes per variant.
# Building a report from them is a separate step.
#
# Logs: this script to stdout, per-config detail to /tmp/pdesim_{purple,launch}_c<N>.log
set -u

PROBLEMS=advection
CONFIGS=1
MODEL=claudeopus48
PROGRESS_EVERY=30
BENCH=${PETSCAGENT_BENCH:-$HOME/petscagent-bench}
OUTPUT=
RUN_ID=$(date +%Y%m%d-%H%M%S)

usage() {
  cat <<'EOF'
Usage: run_pdesim.sh [-p|--problems LIST] [-c|--configs LIST] [-m|--model NAME]
                     [-i|--progress SECS] [-b|--bench DIR]

  -p, --problems LIST  comma-separated problem terms, as main.py --problems takes
                       them, e.g. darcy or darcy,robertson  (default: advection)
                       a term may be a glob, so '*' runs the full matrix
  -c, --configs LIST   comma-separated pde-sim configs, e.g. 1,2    (default: 1)
  -m, --model NAME     model the Purple generates with, as PDESIM_CLAUDE_MODEL
                       takes it, e.g. claudeopus48 or 'claudeopus48[1m]'
                                                       (default: claudeopus48)
  -i, --progress SECS  seconds between progress lines, 0 to silence (default: 30)
  -b, --bench DIR      petscagent-bench repo root
                             (default: $PETSCAGENT_BENCH, else ~/petscagent-bench)
  -o, --output DIR     where Green writes results, as main.py --output takes it
                       (default: $BENCH/output/pdesim_<timestamp>, one per run)
  -h, --help           this message

PETSC_DIR and PETSC_ARCH select the PETSc build to run against. Both must be set
in the environment; the script never picks one for you.

Note: quote a model id containing brackets, or the shell globs it away:

    -m 'claudeopus48[1m]'    Opus 4.8 with a 1M context window rather than 200K

The bare id takes the 200K default. A pde-sim run has been seen to peak at ~220K
tokens of context, so the 1M form is what avoids mid-run compaction.
EOF
}

die() { echo "run_pdesim.sh: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case $1 in
    -p | --problems) [ $# -ge 2 ] || die "$1 needs a value"; PROBLEMS=$2; shift 2 ;;
    --problems=*) PROBLEMS=${1#*=}; shift ;;
    -c | --configs) [ $# -ge 2 ] || die "$1 needs a value"; CONFIGS=$2; shift 2 ;;
    --configs=*) CONFIGS=${1#*=}; shift ;;
    -m | --model) [ $# -ge 2 ] || die "$1 needs a value"; MODEL=$2; shift 2 ;;
    --model=*) MODEL=${1#*=}; shift ;;
    -i | --progress) [ $# -ge 2 ] || die "$1 needs a value"; PROGRESS_EVERY=$2; shift 2 ;;
    --progress=*) PROGRESS_EVERY=${1#*=}; shift ;;
    -b | --bench) [ $# -ge 2 ] || die "$1 needs a value"; BENCH=$2; shift 2 ;;
    --bench=*) BENCH=${1#*=}; shift ;;
    -o | --output) [ $# -ge 2 ] || die "$1 needs a value"; OUTPUT=$2; shift 2 ;;
    --output=*) OUTPUT=${1#*=}; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "run_pdesim.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# The config loop wants whitespace separation; main.py --problems wants the
# comma-separated form as typed.
CONFIG_LIST=$(echo "$CONFIGS" | tr ',' ' ')
[ -n "${PROBLEMS//[[:space:],]/}" ] || die "--problems is empty"
[ -n "${CONFIG_LIST//[[:space:]]/}" ] || die "--configs is empty"
[ -n "${MODEL//[[:space:]]/}" ] || die "--model is empty"
[ -n "${BENCH//[[:space:]]/}" ] || die "--bench is empty"
case $PROGRESS_EVERY in
  "" | *[!0-9]*) die "--progress wants a whole number of seconds, got '$PROGRESS_EVERY'" ;;
esac
# Unset is fine — the default applies below — but a blank value is a typo.
case $OUTPUT in "") ;; *[![:space:]]*) ;; *) die "--output is empty" ;; esac

# The PETSc build is whichever one the environment selects: choosing a root or an
# arch here would quietly run against a build the caller did not pick.
[ -n "${PETSC_DIR:-}" ] || die "PETSC_DIR is not set; set it to the PETSc root"
[ -n "${PETSC_ARCH:-}" ] || die "PETSC_ARCH is not set; set it to the build to run against"

# Make both absolute before anything is derived from them. The script cd's to
# $BENCH further down, after which a relative path would re-resolve against the
# new directory — quietly, because the parking globs would just match nothing.
# Logical pwd, so a path given through a symlink stays as the caller wrote it.
abs=$(cd "$PETSC_DIR" 2>/dev/null && pwd) || die "PETSC_DIR=$PETSC_DIR is not a directory"
PETSC_DIR=$abs
abs=$(cd "$BENCH" 2>/dev/null && pwd) || die "no petscagent-bench at $BENCH; set --bench or PETSCAGENT_BENCH"
BENCH=$abs

PURPLE=$PETSC_DIR/.agents/pde-sim/bindings/petscagent-bench/pdesim_purple.py
ARTIFACTS=$PETSC_DIR/artifacts
CASE_INDEX=$PETSC_DIR/.agents/pde-sim/components/case-index.json
ARCHIVE=$PETSC_DIR/artifacts_runs/run_$RUN_ID
# One output directory per invocation, so a run's results stand alone without
# having to park the previous run's out of the way first. Green creates it and
# writes the aggregate JSON plus its code/ and scores/ trees there, one set per
# config (the configs differ in their model slug, pdesim-<model>-c<N>, so they
# share a directory without colliding). Same stamp as $ARCHIVE, to pair them.
OUTPUT=${OUTPUT:-$BENCH/output/pdesim_$RUN_ID}
[ -f "$PURPLE" ] || die "no pde-sim binding under PETSC_DIR=$PETSC_DIR (looked for $PURPLE)"
[ -f "$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables" ] ||
  die "PETSC_ARCH=$PETSC_ARCH is not a configured build of $PETSC_DIR"
[ -f "$BENCH/main.py" ] || die "no petscagent-bench at $BENCH (looked for main.py); set --bench or PETSCAGENT_BENCH"

PURPLE_PORT=9002
PURPLE_URL=http://localhost:$PURPLE_PORT/

# PETSC_DIR and PETSC_ARCH need no export: they came from the environment, so
# the Purple and the pipeline below it already inherit them.
export PDESIM_CLAUDE_MODEL=$MODEL

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

# Run a command in a new session so it leads its own process group and stop_purple
# can signal the whole subtree. macOS has no setsid binary; os.setsid() is the same
# POSIX call, so fall back to a one-line Python exec wrapper that does it in-process.
if command -v setsid >/dev/null 2>&1; then
  new_session() { setsid "$@"; }
else
  new_session() { python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])' "$@"; }
fi

# port_free reports whether a TCP port has no listener; port_pids lists the PIDs
# of its listeners (newline-separated, empty if none). Both share one backend so
# they never disagree. lsof is present on macOS and on most Linux boxes; ss
# (iproute2) covers Linux installs without lsof. In the interactive shell `ss` is
# aliased to start_ssh, so call the binary by its path.
if command -v lsof >/dev/null 2>&1; then
  port_free() { ! lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
  port_pids() { lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null; }
elif [ -x /usr/sbin/ss ] || [ -x /usr/bin/ss ]; then
  SS=$([ -x /usr/sbin/ss ] && echo /usr/sbin/ss || echo /usr/bin/ss)
  port_free() { ! $SS -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$1\$"; }
  port_pids() { $SS -ltnp 2>/dev/null | grep -E "[:.]$1 " | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u; }
else
  die "need lsof or ss to check ports; install one and retry"
fi

# Move rather than delete, so a run is always recoverable from $ARCHIVE.
stash() {                                # stash <dest-subdir> <path>...
  local dest=$ARCHIVE/$1; shift
  [ $# -eq 0 ] && return 0
  mkdir -p "$dest"
  mv "$@" "$dest"/ 2>/dev/null
}

PURPLE_PID=""
PURPLE_PGID=""
PROGRESS_PID=""
MY_PGID=$(ps -o pgid= -p $$ | tr -d ' ')
# One marker per finished study, so a problem is reported once however often the
# watcher and the post-launch sweep both look at it.
DONE_DIR=$(mktemp -d)

# Report every problem that has finished since the last look. The per-problem
# figures come from the `result` event the CLI writes at the end of a study's
# transcript.stream.jsonl; a run killed at PDESIM_TIMEOUT never emits one, so it
# stays unreported here rather than being shown with invented numbers.
report_done() {                          # report_done <config>
  local c=$1
  python3 - "$ARTIFACTS" "$DONE_DIR" <<'PY' | while IFS= read -r line; do say "  done c$c $line"; done
import json, os, sys

artifacts, marks = sys.argv[1], sys.argv[2]
if not os.path.isdir(artifacts):
    sys.exit()

for study in sorted(os.listdir(artifacts)):
    if not study.startswith("bench-"):
        continue
    mark = os.path.join(marks, study)
    if os.path.exists(mark):
        continue
    result = None
    try:
        with open(os.path.join(artifacts, study, "transcript.stream.jsonl"), errors="replace") as fh:
            for line in fh:
                try:
                    event = json.loads(line)
                except ValueError:        # a partially flushed trailing line
                    continue
                if event.get("type") == "result":
                    result = event
    except OSError:                       # no transcript yet; still running
        continue
    if result is None:
        continue
    open(mark, "w").close()

    problem = study[len("bench-"):].rsplit("-", 1)[0]
    secs = int((result.get("duration_ms") or 0) / 1000)
    cost = result.get("total_cost_usd")
    cost = f"${cost:,.2f}" if isinstance(cost, (int, float)) else "cost unreported"
    status = result.get("subtype") or result.get("terminal_reason") or "?"
    print(f"{problem}  {secs // 60}m{secs % 60:02d}s  {cost}  {status}")
PY
}

# The pde-sim pipeline flushes one JSON event per line into the study dir's
# transcript.stream.jsonl expressly so it can be followed live, and it is the
# only thing that moves during a run: the launch log goes quiet between
# "Sending..." and "Terminating agents...", which can be the better part of an
# hour. Summarize the newest study dir rather than tail the raw stream, since a
# run writes one dir per problem and several transcripts per dir.
summarize_stream() {                     # summarize_stream <study-dir>
  python3 - "$1" <<'PY'
import json, os, sys

study = sys.argv[1].rstrip("/")
path = os.path.join(study, "transcript.stream.jsonl")

def descr(block):
    """Shortest useful identifier for a tool call: what it acts on."""
    args = block.get("input") or {}
    for key in ("command", "file_path", "path", "pattern", "description"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return " " + val.strip().replace("\n", " ")[:56]
    return ""

events = 0
last = ""
try:
    with open(path, errors="replace") as fh:
        for line in fh:
            events += 1
            try:
                ev = json.loads(line)
            except ValueError:            # a partially flushed trailing line
                continue
            for block in (ev.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    last = f"{block.get('name')}{descr(block)}"
                elif block.get("type") == "text" and block.get("text", "").strip():
                    last = "say " + block["text"].strip().split("\n")[0][:56]
except FileNotFoundError:
    print(f"{os.path.basename(study)}  starting up")
    sys.exit()

print(f"{os.path.basename(study)}  {events} events  last: {last or '-'}")
PY
}

elapsed() { local s=$((SECONDS - $1)); printf '%dm%02ds' $((s / 60)) $((s % 60)); }

start_progress() {                       # start_progress <config>
  PROGRESS_PID=""
  [ "$PROGRESS_EVERY" -gt 0 ] || return 0
  local c=$1 start=$SECONDS
  (
    while sleep "$PROGRESS_EVERY"; do
      # Newest first, so this follows the pipeline from one problem to the next.
      report_done "$c"                   # announce problems as they finish
      # ls -dt sorts by mtime newest-first on both BSD and GNU; find -printf is
      # GNU-only. nullglob keeps the glob from passing a literal 'bench-*' to ls
      # when nothing has been created yet.
      shopt -s nullglob
      newest=$(ls -dt "$ARTIFACTS"/bench-*/ 2>/dev/null | head -1)
      shopt -u nullglob
      if [ -n "$newest" ]; then
        say "  c$c $(elapsed $start)  $(summarize_stream "$newest")"
      else
        say "  c$c $(elapsed $start)  waiting for a study dir"
      fi
    done
  ) &
  PROGRESS_PID=$!
}

stop_progress() {
  [ -n "$PROGRESS_PID" ] || return 0
  kill "$PROGRESS_PID" 2>/dev/null
  wait "$PROGRESS_PID" 2>/dev/null
  PROGRESS_PID=""
}

start_purple() {                         # start_purple <config>
  local c=$1
  # setsid puts the Purple in its own process group so stop_purple can take down
  # the claude CLI it spawns too; without that a killed Purple leaves the
  # generation running, which is how an orphan once finished after a Ctrl-C.
  # No PDESIM_USAGE_DIR: that would make the Purple append a usage-*.jsonl
  # sidecar, which nothing reads. Green takes its telemetry in-band, from the
  # petscagent.telemetry.v1 data Part the Purple returns over A2A.
  PDESIM_CONFIG=$c \
    new_session uv run python "$PURPLE" > "/tmp/pdesim_purple_c$c.log" 2>&1 &
  PURPLE_PID=$!
  # Advisory only: new_session()'s setsid() may not have run yet, so this can read
  # the inherited (script) group. stop_purple re-reads the pgid when it actually
  # matters, and reaps by port regardless.
  PURPLE_PGID=$(ps -o pgid= -p "$PURPLE_PID" 2>/dev/null | tr -d ' ')
  say "started Purple (config=$c) pid=$PURPLE_PID pgid=${PURPLE_PGID:-?}"
}

stop_purple() {
  [ -z "$PURPLE_PID" ] && return 0
  # Re-read the pgid now rather than trusting start_purple's snapshot: by this
  # point setsid() has long since run, so the launcher's real group is readable.
  # Signal that group when it is distinct from ours; killing our own group would
  # take down this script.
  local pgid target
  pgid=$(ps -o pgid= -p "$PURPLE_PID" 2>/dev/null | tr -d ' ')
  target=$PURPLE_PID
  if [ -n "$pgid" ] && [ "$pgid" != "$MY_PGID" ]; then target=-$pgid; fi
  kill -TERM "$target" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "$PURPLE_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$PURPLE_PID" 2>/dev/null; then
    say "Purple $PURPLE_PID ignored SIGTERM; sending SIGKILL"
    kill -KILL "$target" 2>/dev/null
  fi
  wait "$PURPLE_PID" 2>/dev/null
  PURPLE_PID=""; PURPLE_PGID=""
  port_free "$PURPLE_PORT" && return 0
  # `uv run` can place the A2A server in a process group of its own, escaping the
  # group kill above. Reap by port (ours only, matched on the pdesim_purple.py
  # command line) so 9002 is free for the next config instead of being left for
  # the next run's preflight to clean up.
  reap_orphan_purple && port_free "$PURPLE_PORT" && return 0
  say "WARNING: port $PURPLE_PORT still bound after teardown"
}

# A run killed uncleanly (Ctrl-C before setsid, SIGKILL, a crash) can leave its
# Purple alive, holding $PURPLE_PORT and blocking the next run's preflight. Reap
# such an orphan, but ONLY when the listener is one of our own Purples: match the
# pdesim_purple.py path in its command line so an unrelated listener (an editor's
# port forwarder, another service) is never touched and still fails the preflight.
reap_orphan_purple() {
  local pid pgid ours=0
  for pid in $(port_pids "$PURPLE_PORT"); do
    ps -p "$pid" -o command= 2>/dev/null | grep -qF "$PURPLE" || continue
    ours=1
    pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    say "reaping orphaned Purple on port $PURPLE_PORT (pid=$pid pgid=${pgid:-?})"
    # Kill the whole group when it is its own, matching stop_purple; never signal
    # our own group, which would take down this script.
    if [ -n "$pgid" ] && [ "$pgid" != "$MY_PGID" ]; then kill -TERM "-$pgid" 2>/dev/null; else kill -TERM "$pid" 2>/dev/null; fi
    for _ in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      say "orphan $pid ignored SIGTERM; sending SIGKILL"
      if [ -n "$pgid" ] && [ "$pgid" != "$MY_PGID" ]; then kill -KILL "-$pgid" 2>/dev/null; else kill -KILL "$pid" 2>/dev/null; fi
    fi
  done
  [ "$ours" -eq 1 ] || return 0
  # Confirm the port actually freed before the caller's preflight re-checks it.
  for _ in $(seq 1 15); do
    port_free "$PURPLE_PORT" && return 0
    sleep 1
  done
  return 1
}

# Ctrl-C and errors must not leave a Purple generating in the background.
trap 'echo; say "interrupted - stopping Purple"; stop_progress; stop_purple; exit 130' INT TERM
trap 'stop_progress; stop_purple; rm -rf "$DONE_DIR"' EXIT

cd "$BENCH" || exit 1
mkdir -p "$ARCHIVE" "$OUTPUT"

say "=== $PROBLEMS x configs $CONFIGS on model $MODEL ==="
say "output  -> $(cd "$OUTPUT" && pwd)"   # resolve portably; readlink -f is GNU-only
say "archive -> $ARCHIVE"

# Resolve the problem terms now: a typo should fail here, not after a server is up.
uv run main.py problems "$PROBLEMS" || exit 1

# Ports must be clear, or `launch` binds nothing and every compile fails. On
# $PURPLE_PORT a busy port may be one of our own orphaned Purples from a run that
# died without teardown; reap that before giving up. 8080 and 9001 are bound by
# `launch`, never by us, so a listener there is always someone else's — fail.
for p in 8080 9001 $PURPLE_PORT; do
  port_free "$p" && continue
  if [ "$p" = "$PURPLE_PORT" ] && reap_orphan_purple && port_free "$p"; then continue; fi
  say "ERROR: port $p is already in use"
  exit 1
done

for c in $CONFIG_LIST; do
  say "===== CONFIG $c ====="

  # Clean slate for this config.
  shopt -s nullglob
  stash "stale_c$c" "$ARTIFACTS"/bench-*
  shopt -u nullglob
  printf '{\n  "schema_version": "1.0",\n  "cases": []\n}\n' > "$CASE_INDEX"
  rm -rf "${BENCH:?}/purple_agent_cache"/*

  start_purple "$c"
  up=0
  for _ in $(seq 1 150); do               # up to 5 min
    port_free "$PURPLE_PORT" || { up=1; break; }
    sleep 2
  done
  if [ $up -ne 1 ]; then
    say "ERROR: Purple did not come up for config $c; skipping"
    stop_purple
    continue
  fi
  say "Purple up; running $PROBLEMS"

  start_progress "$c"
  uv run main.py launch --purple-url "$PURPLE_URL" --problems "$PROBLEMS" \
    --output "$OUTPUT" > "/tmp/pdesim_launch_c$c.log" 2>&1
  rc=$?                                  # capture before stop_progress clobbers it
  stop_progress
  # Catch whatever finished after the watcher's last look, and cover -i 0.
  report_done "$c"
  say "config $c launch exited rc=$rc"

  stop_purple
  shopt -s nullglob
  stash "c$c" "$ARTIFACTS"/bench-*        # keep this config's study dirs
  shopt -u nullglob
  say "config $c done; study dirs in $ARCHIVE/c$c"
done

say "=== complete ==="
say "results:"; ls -1 "$OUTPUT"/*.json 2>/dev/null
