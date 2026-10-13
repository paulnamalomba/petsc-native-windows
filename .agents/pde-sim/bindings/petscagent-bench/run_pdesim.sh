#!/usr/bin/env bash
# Run one or more pde-sim configs over one or more petscagent-bench problems,
# one case (problem x config) at a time, and keep every case's artifacts in a
# directory outside $PETSC_DIR.
#
# This does not use the bench's main.py, Green agent or A2A server. It reads the
# problem descriptions from the bench's data/*.json and runs the pde-sim pipeline
# on each one directly through `pdesim_purple.py run`. Nothing is graded.
#
# No state is carried from one case to the next, so no case can be influenced by
# another. Before each case this clears the two things that leak between runs:
#   artifacts/          code generation cribs a prior solution from a study dir
#   case-index.json     the pde-sim case index remembers earlier cases
# Nothing is deleted: anything in artifacts/ is moved into the output directory.
#
# Usage: ./run_pdesim.sh [-p|--problems LIST] [-c|--configs LIST] [-m|--model NAME]
#                        [-i|--progress SECS] [-t|--timeout MIN] [-b|--bench DIR]
#                        [-o|--output DIR]
#
# PETSC_DIR and PETSC_ARCH must both be set: they select the PETSc build, and the
# pde-sim pipeline runs from PETSC_DIR. The bench repo is a separate checkout
# that supplies only the problem files; it comes from --bench, else
# $PETSCAGENT_BENCH, else ~/petscagent-bench.
#
# Each case's study dir ends up in $OUTPUT/c<config>/<problem>/, holding the
# generated source, the specs, results-manifest.json, the transcripts and
# case-result.json (the run parameters, sources and usage this script reports).
# $OUTPUT also gets the usage sidecar usage-pdesim-<model>-c<config>.jsonl and
# summary.tsv, one line per case.
#
# If the LLM stops being usable partway (the API credit balance or the usage
# limit runs out), the case is paused, not failed: its study, case index and
# Claude session are parked in $OUTPUT/paused/c<config>-<problem>/ and the CLI's
# message is shown. At a terminal the script waits for Enter, after funds are
# added, and then continues that same session where it stopped; otherwise, or on
# q, it exits with status 75 and prints the command that resumes. Rerunning with
# the same --output resumes the paused case and skips the finished ones.
#
# Logs: this script to stdout, per-case detail to $OUTPUT/logs/c<N>-<problem>.log
set -u

PROBLEMS=advection
CONFIGS=1
MODEL=claudeopus48
PROGRESS_EVERY=30
TIMEOUT_MIN=1440          # 24h: effectively off unless asked for
BENCH=${PETSCAGENT_BENCH:-$HOME/petscagent-bench}
OUTPUT=
RUN_ID=$(date +%Y%m%d-%H%M%S)
ORIG_ARGS=("$@")          # to print the command that resumes a paused run
# Ask before resuming a case paused for budget only when someone is there to answer.
INTERACTIVE=
[ -t 0 ] && INTERACTIVE=1

usage() {
  cat <<'EOF'
Usage: run_pdesim.sh [-p|--problems LIST] [-c|--configs LIST] [-m|--model NAME]
                     [-i|--progress SECS] [-t|--timeout MIN] [-b|--bench DIR]
                     [-o|--output DIR]

  -p, --problems LIST  comma-separated terms matched against the bench's problem
                       names, ignoring case: a plain term matches a substring, a
                       term with a wildcard is a glob, so '*' runs every problem
                       e.g. darcy or darcy,robertson        (default: advection)
  -c, --configs LIST   comma-separated pde-sim configs, e.g. 1,2    (default: 1)
  -m, --model NAME     model the pipeline generates with, as PDESIM_CLAUDE_MODEL
                       takes it, e.g. claudeopus48 or 'claudeopus48[1m]'
                                                       (default: claudeopus48)
  -i, --progress SECS  seconds between progress lines, 0 to silence (default: 30)
  -t, --timeout MIN    stop a case after MIN minutes and keep whatever it
                       finished                          (default: 1440, i.e. 24h)
  -b, --bench DIR      petscagent-bench checkout whose data/*.json are the problems
                             (default: $PETSCAGENT_BENCH, else ~/petscagent-bench)
  -o, --output DIR     where every case's artifacts are kept; must be outside
                       PETSC_DIR    (default: ~/pdesim-runs/<timestamp>, one per run)
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
    -t | --timeout) [ $# -ge 2 ] || die "$1 needs a value"; TIMEOUT_MIN=$2; shift 2 ;;
    --timeout=*) TIMEOUT_MIN=${1#*=}; shift ;;
    -b | --bench) [ $# -ge 2 ] || die "$1 needs a value"; BENCH=$2; shift 2 ;;
    --bench=*) BENCH=${1#*=}; shift ;;
    -o | --output) [ $# -ge 2 ] || die "$1 needs a value"; OUTPUT=$2; shift 2 ;;
    --output=*) OUTPUT=${1#*=}; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "run_pdesim.sh: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

CONFIG_LIST=$(echo "$CONFIGS" | tr ',' ' ')
[ -n "${PROBLEMS//[[:space:],]/}" ] || die "--problems is empty"
[ -n "${CONFIG_LIST//[[:space:]]/}" ] || die "--configs is empty"
for c in $CONFIG_LIST; do
  case $c in 1 | 2 | 3) ;; *) die "--configs takes 1, 2 or 3, got '$c'" ;; esac
done
[ -n "${MODEL//[[:space:]]/}" ] || die "--model is empty"
[ -n "${BENCH//[[:space:]]/}" ] || die "--bench is empty"
case $TIMEOUT_MIN in
  "" | *[!0-9]* | 0) die "--timeout wants a whole number of minutes, got '$TIMEOUT_MIN'" ;;
esac
case $PROGRESS_EVERY in
  "" | *[!0-9]*) die "--progress wants a whole number of seconds, got '$PROGRESS_EVERY'" ;;
esac
# Unset is fine — the default applies below — but a blank value is a typo.
case $OUTPUT in "") ;; *[![:space:]]*) ;; *) die "--output is empty" ;; esac

# The PETSc build is whichever one the environment selects: choosing a root or an
# arch here would quietly run against a build the caller did not pick.
[ -n "${PETSC_DIR:-}" ] || die "PETSC_DIR is not set; set it to the PETSc root"
[ -n "${PETSC_ARCH:-}" ] || die "PETSC_ARCH is not set; set it to the build to run against"

# Physical paths, so the outside-PETSC_DIR check below cannot be fooled by a
# symlink into the tree.
abs=$(cd "$PETSC_DIR" 2>/dev/null && pwd -P) || die "PETSC_DIR=$PETSC_DIR is not a directory"
PETSC_DIR=$abs
abs=$(cd "$BENCH" 2>/dev/null && pwd -P) || die "no petscagent-bench at $BENCH; set --bench or PETSCAGENT_BENCH"
BENCH=$abs
DATA=$BENCH/data

RUNNER=$PETSC_DIR/.agents/pde-sim/bindings/petscagent-bench/pdesim_purple.py
ARTIFACTS=$PETSC_DIR/artifacts
CASE_INDEX=$PETSC_DIR/.agents/pde-sim/components/case-index.json
[ -f "$RUNNER" ] || die "no pde-sim runner under PETSC_DIR=$PETSC_DIR (looked for $RUNNER)"
[ -f "$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables" ] ||
  die "PETSC_ARCH=$PETSC_ARCH is not a configured build of $PETSC_DIR"
[ -d "$DATA" ] || die "no problem files at $DATA; set --bench or PETSCAGENT_BENCH"

OUTPUT_GIVEN=$OUTPUT
OUTPUT=${OUTPUT:-$HOME/pdesim-runs/$RUN_ID}
created=
[ -d "$OUTPUT" ] || created=1
mkdir -p "$OUTPUT" || die "cannot create --output $OUTPUT"
OUTPUT=$(cd "$OUTPUT" && pwd -P)
case $OUTPUT/ in
  "$PETSC_DIR"/*)
    [ -n "$created" ] && rmdir "$OUTPUT"
    die "--output $OUTPUT is inside PETSC_DIR=$PETSC_DIR; choose a directory outside it" ;;
esac
LOGS=$OUTPUT/logs
SUMMARY=$OUTPUT/summary.tsv
mkdir -p "$LOGS"

# PETSC_DIR and PETSC_ARCH need no export: they came from the environment, so
# the runner and the pipeline below it already inherit them.
export PDESIM_CLAUDE_MODEL=$MODEL
# Minutes here, seconds in the runner. A case stopped at the limit still keeps
# whatever it had finished, so this is a budget rather than a failure.
export PDESIM_TIMEOUT=$((TIMEOUT_MIN * 60))
# The runner appends one usage record per case to
# $OUTPUT/usage-pdesim-<model>-c<config>.jsonl.
export PDESIM_USAGE_DIR=$OUTPUT

ts() { date '+%F %T'; }
say() { echo "[$(ts)] $*"; }

# Resolve the problem terms now, so a typo fails before anything runs. Prints
# one "<file><TAB><problem_name>" line per selected problem, in file order. The
# matching is the bench's own: case-insensitive substring, or glob when the term
# holds a wildcard; hidden and non-.json files are ignored.
select_problems() {
  python3 -I - "$DATA" "$PROBLEMS" <<'PY'
import fnmatch, json, sys
from pathlib import Path

data, spec = Path(sys.argv[1]), sys.argv[2]
problems = []
for f in sorted(p for p in data.iterdir()
                if p.is_file() and p.suffix.lower() == ".json" and not p.name.startswith(".")):
    try:
        prob = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"{f} cannot be read: {e}")
    if not isinstance(prob, dict) or not prob.get("problem_name") or not prob.get("problem_description"):
        sys.exit(f"{f} has no problem_name or problem_description")
    problems.append((f, prob["problem_name"]))

terms = [t.strip().lower() for t in spec.split(",") if t.strip()]
match = lambda t, n: fnmatch.fnmatch(n, t) if any(c in t for c in "*?[") else t in n  # noqa: E731
unmatched = [t for t in terms if not any(match(t, n.lower()) for _, n in problems)]
if unmatched:
    sys.exit(f"no problem matches: {', '.join(unmatched)}\n"
             f"available: {', '.join(sorted(n for _, n in problems))}")
for f, n in problems:
    if any(match(t, n.lower()) for t in terms):
        print(f"{f}\t{n}")
PY
}

# Move whatever is in artifacts/ into <dest>. Nothing is deleted, so a stray
# study is always recoverable from $OUTPUT.
stash_artifacts() {                      # stash_artifacts <dest>
  local dest=$1 entries
  shopt -s nullglob dotglob
  entries=("$ARTIFACTS"/*)
  shopt -u nullglob dotglob
  [ ${#entries[@]} -eq 0 ] && return 0
  mkdir -p "$dest"
  mv "${entries[@]}" "$dest"/
}

reset_case_index() {
  printf '{\n  "schema_version": "1.0",\n  "cases": []\n}\n' > "$CASE_INDEX"
}

# A paused case keeps exactly what it had: everything in artifacts/ (its study,
# and anything else the pipeline wrote there) and the case index, so a resumed
# session finds its files at the same paths.
park_case() {                            # park_case <dir>
  mkdir -p "$1"
  cp "$CASE_INDEX" "$1/case-index.json"
  stash_artifacts "$1/artifacts"
}

unpark_case() {                          # unpark_case <dir>
  local entries
  shopt -s nullglob dotglob
  entries=("$1"/artifacts/*)
  shopt -u nullglob dotglob
  mkdir -p "$ARTIFACTS"
  [ ${#entries[@]} -eq 0 ] || mv "${entries[@]}" "$ARTIFACTS"/
  cp "$1/case-index.json" "$CASE_INDEX"
  rm -f "$1/case-index.json"
  rmdir "$1/artifacts" "$1" 2>/dev/null
  rmdir "$OUTPUT/paused" 2>/dev/null
  return 0
}

# A field of a study's case-result.json, empty when absent.
case_field() {                           # case_field <study-dir> <key>
  python3 -I -c 'import json, sys
try:
    v = json.load(open(sys.argv[1] + "/case-result.json")).get(sys.argv[2])
except (OSError, ValueError):
    v = None
print("" if v is None else v)' "$1" "$2"
}

# The command that picks this run up again: the same arguments and output dir.
resume_command() {
  local args=("${ORIG_ARGS[@]}")
  [ -n "$OUTPUT_GIVEN" ] || args+=(-o "$OUTPUT")
  printf 'PETSC_DIR=%q PETSC_ARCH=%q %q' "$PETSC_DIR" "$PETSC_ARCH" "$0"
  printf ' %q' "${args[@]}"
  echo
}

# The pipeline flushes one JSON event per line into the study dir's
# transcript.stream.jsonl expressly so it can be followed live. Summarize it
# rather than tail the raw stream.
summarize_stream() {                     # summarize_stream <study-dir>
  python3 -I - "$1" <<'PY'
import json, os, sys

path = os.path.join(sys.argv[1], "transcript.stream.jsonl")

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
    print("starting up")
    sys.exit()

print(f"{events} events  last: {last or '-'}")
PY
}

# One line for the case: wall time and cost from the transcript's final `result`
# event (absent for a run killed at PDESIM_TIMEOUT), and the outcome from
# case-result.json. Also appended to $SUMMARY.
report_case() {                          # report_case <config> <problem> <study-dir> <rc>
  python3 -I - "$@" "$SUMMARY" <<'PY'
import json, os, sys

c, problem, study, rc, summary = sys.argv[1:6]
result = None
try:
    with open(os.path.join(study, "transcript.stream.jsonl"), errors="replace") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "result":
                result = event
except OSError:
    pass
try:
    case = json.load(open(os.path.join(study, "case-result.json")))
except (OSError, ValueError):
    case = {}

secs = case.get("wall_sec")             # the runner's own clock, else the CLI's
secs = int(secs if secs is not None else ((result or {}).get("duration_ms") or 0) / 1000)
cost = (result or {}).get("total_cost_usd")
cost = f"${cost:,.2f}" if isinstance(cost, (int, float)) else "-"
if case.get("error"):
    outcome = "failed: " + case["error"][:80]
elif case:
    outcome = "ok" + (f" ({case['partial']})" if case.get("partial") else "")
else:
    outcome = f"no case-result.json (runner exit {rc})"
cmd = case.get("command") or "-"
print(f"c{c} {problem}  {secs // 60}m{secs % 60:02d}s  {cost}  {outcome}  {cmd}")
new = not os.path.exists(summary)
with open(summary, "a") as fh:
    if new:
        fh.write("config\tproblem\twall_sec\tcost_usd\toutcome\tcommand\tstudy_dir\n")
    fh.write("\t".join([c, problem, str(secs), cost.lstrip("$").replace(",", ""),
                        outcome, cmd, study]) + "\n")
PY
}

elapsed() { local s=$((SECONDS - $1)); printf '%dm%02ds' $((s / 60)) $((s % 60)); }

PROGRESS_PID=""
RUNNER_PID=""
EXIT_PAUSED=75            # the runner's status for a case paused for LLM budget

start_progress() {                       # start_progress <label> <study-dir>
  PROGRESS_PID=""
  [ "$PROGRESS_EVERY" -gt 0 ] || return 0
  local label=$1 study=$2 start=$SECONDS
  (
    while sleep "$PROGRESS_EVERY"; do
      say "  $label $(elapsed $start)  $(summarize_stream "$study")"
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

# Prefix that runs a command in a new session, so it leads its own process group
# and an interrupt can signal the whole subtree, including the claude CLI it
# spawns. macOS has no setsid binary; os.setsid() is the same POSIX call. An
# array rather than a function: a backgrounded function runs in a forked
# subshell, so $! would name the subshell and not the group leader.
if command -v setsid >/dev/null 2>&1; then
  NEW_SESSION=(setsid)
else
  NEW_SESSION=(python3 -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])')
fi

stop_runner() {
  [ -n "$RUNNER_PID" ] || return 0
  kill -TERM "-$RUNNER_PID" 2>/dev/null || kill -TERM "$RUNNER_PID" 2>/dev/null
  for _ in $(seq 1 20); do
    kill -0 "$RUNNER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "-$RUNNER_PID" 2>/dev/null
  wait "$RUNNER_PID" 2>/dev/null
  RUNNER_PID=""
}

# Ctrl-C must not leave a pipeline generating in the background, and whatever
# the interrupted case wrote is still moved out of the tree.
trap 'echo; say "interrupted - stopping the current case"; stop_progress; stop_runner;
      stash_artifacts "$OUTPUT/interrupted"; exit 130' INT TERM
trap 'stop_progress; stop_runner' EXIT

SELECTED=$(select_problems) || exit 1
NPROB=$(printf '%s\n' "$SELECTED" | grep -c .)

say "=== $NPROB problem(s) x configs $CONFIGS on model $MODEL ==="
say "problems from $DATA"
say "output -> $OUTPUT"

# Studies left over from an earlier run would be cribbed from; park them first.
if [ -n "$(ls -A "$ARTIFACTS" 2>/dev/null)" ]; then
  say "moving what was already in $ARTIFACTS to $OUTPUT/stale"
  stash_artifacts "$OUTPUT/stale"
fi

for c in $CONFIG_LIST; do
  say "===== CONFIG $c ====="
  while IFS=$'\t' read -r pfile pname; do
    [ -n "$pfile" ] || continue
    slug=$(printf '%s' "$pname" | tr -c 'A-Za-z0-9._-' '_')
    final=$OUTPUT/c$c/$slug
    park=$OUTPUT/paused/c$c-$slug
    log=$LOGS/c$c-$slug.log
    if [ -e "$final" ]; then
      say "c$c $pname: $final already exists; skipping"
      continue
    fi
    resume=()
    if [ -d "$park" ]; then
      # Back where it ran, so the resumed session finds its files.
      unpark_case "$park"
      study=$(case_field "$(ls -d "$ARTIFACTS"/bench-"$slug"-* | head -1)" run_dir)
      case $study in
        "$ARTIFACTS"/bench-*) [ -d "$study" ] || die "paused case c$c $pname: no study at $study" ;;
        *) die "paused case c$c $pname ran at '$study', not under $ARTIFACTS; resume it with the same PETSC_DIR" ;;
      esac
      resume=(--resume)
      say "c$c $pname: resuming the paused session (log $log)"
    else
      # The pipeline runs inside the tree, as the orchestration skill expects its
      # studies under artifacts/; the study is moved out once the case ends.
      study=$ARTIFACTS/bench-$slug-$(date +%s)
      reset_case_index
      mkdir -p "$ARTIFACTS"
      say "c$c $pname: running (log $log)"
    fi

    while :; do
      start_progress "c$c $pname" "$study"
      PDESIM_CONFIG=$c "${NEW_SESSION[@]}" python3 "$RUNNER" run --problem "$pfile" --study-dir "$study" \
        ${resume[@]+"${resume[@]}"} >> "$log" 2>&1 &
      RUNNER_PID=$!
      wait "$RUNNER_PID"
      rc=$?
      RUNNER_PID=""
      stop_progress
      [ "$rc" -eq "$EXIT_PAUSED" ] || break

      # Out of LLM budget: the case is intact, so keep it and say why.
      ran=$(case_field "$study" active_sec)
      ran=${ran%.*}
      why=$(case_field "$study" pause_reason)
      park_case "$park"
      say "c$c $pname PAUSED after $((ran / 60))m$((ran % 60))s of work: the LLM can no longer be used:"
      say "    ${why:-no reason reported; see $log}"
      say "  Its study and Claude session are saved in $park; nothing is lost."
      if [ -n "$INTERACTIVE" ]; then
        say "  Add funds (or wait for the usage limit to reset), then press Enter to"
        say "  continue this case where it stopped, or type q and Enter to stop here."
        answer=q
        read -r answer < /dev/tty || answer=q
      else
        answer=q
      fi
      if [ "$answer" = q ]; then
        say "  To continue later, run:"
        echo "    $(resume_command)"
        exit "$EXIT_PAUSED"
      fi
      unpark_case "$park"
      resume=(--resume)
      say "c$c $pname: resuming the paused session"
    done

    # The study dir becomes the case's final directory; anything else the
    # pipeline left in artifacts/ goes beside it rather than being lost.
    mkdir -p "$OUTPUT/c$c"
    if [ -d "$study" ]; then mv "$study" "$final"; else mkdir -p "$final"; fi
    stash_artifacts "$final/other-artifacts"
    report_case "$c" "$pname" "$final" "$rc" | while IFS= read -r line; do say "  done $line"; done
  done <<<"$SELECTED"
done

reset_case_index
say "=== complete ==="
say "summary: $SUMMARY"
