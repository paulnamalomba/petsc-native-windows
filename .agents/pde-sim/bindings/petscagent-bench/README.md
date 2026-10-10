# pde-sim runner for petscagent-bench problems

Runs the pde-sim pipeline on the problems of
[petscagent-bench](https://github.com/petsc/petscagent-bench), one case (problem ×
ablation config) at a time, and keeps each case's artifacts in a directory
outside `$PETSC_DIR`. `run_pdesim.sh` does not use the bench's `main.py`, Green
agent, MCP server or A2A protocol; it reads the problem descriptions from the
bench's `data/*.json` and nothing else. Nothing is graded: comparing results is
a separate step.

## How it works

For each case, `run_pdesim.sh` calls

```bash
PDESIM_CONFIG=<N> python3 pdesim_purple.py run --problem <bench>/data/<Problem>.json \
    --study-dir $PETSC_DIR/artifacts/bench-<problem>-<time>
```

which

1. builds the prompt for config `<N>` from the problem's `problem_description`,
2. drives pde-sim non-interactively with `claude -p` in **autonomous** mode from
   the repo root, streaming the transcript into the study dir,
3. reads the pipeline's `results-manifest.json`, picks the run that answers the
   problem, and writes `case-result.json` into the study dir: `nsize` (the run's
   `mpi_ranks`), `command`, `cli_args` (the command without the
   `mpiexec -n N ./exe` prefix), the generated source files, usage, and whether
   the answer is partial or the case failed.

The pipeline works inside the tree, because the orchestration skill keeps its
studies under `artifacts/`. When the case ends the script moves the study dir to
its final place, and anything else the pipeline left in `artifacts/` with it.

`run` mode needs only the Python standard library and the `claude` CLI.
`pdesim_purple.py serve` still offers the same run as an A2A Purple agent for
the bench's own Green agent; see [Serving the bench's Green agent](#serving-the-benchs-green-agent).

## Prerequisites

- The `claude` CLI on `PATH` (or set `PDESIM_CLAUDE_BIN`), logged in, able to run
  the `/pde-sim` pipeline in this repo.
- A working PETSc build selected by `PETSC_DIR` / `PETSC_ARCH` in the environment.
- Python 3.9 or later as `python3`.
- A petscagent-bench checkout, for its `data/*.json` problem files:
  `git clone https://github.com/petsc/petscagent-bench`. No `uv sync` is needed
  for `run_pdesim.sh`.

## Driver: `run_pdesim.sh`

```bash
RUN=$PETSC_DIR/.agents/pde-sim/bindings/petscagent-bench/run_pdesim.sh

$RUN -p '*' -c 1,2,3          # the full matrix: every problem on every rung
$RUN -p darcy -c 3            # one problem, one config
$RUN -p darcy,robertson -c 1  # two problems on config 1
$RUN                          # the defaults: advection on config 1, the cheapest
$RUN -p darcy -c 3 -b ~/some/other/bench   # a different bench checkout
$RUN -p darcy -c 3 -o ~/runs/darcy-opus    # a chosen output directory
```

The script works from any directory. Without `-b`, the bench checkout comes from
`$PETSCAGENT_BENCH`, else `$HOME/petscagent-bench`.

A `-p` term matches part of a problem name, ignoring case, and may be a glob, so
`-p darcy` selects `DarcyFlow2D_Steady` and `-p '*'` selects every problem. A
term that matches nothing stops the script before anything runs, listing the
available names. Configs run one after another; within a config, problems run
in file order.

Less common, but useful when comparing models or running unattended:

```bash
$RUN -p darcy -c 3 -m 'claudeopus48[1m]'   # quote: the brackets are shell globs
$RUN -p '*' -c 1,2,3 -i 300                # a progress line every 5 min
$RUN -p darcy -c 3 -i 0                    # no progress lines at all
$RUN -p '*' -c 3 -t 90                     # stop each case after 90 minutes
```

No state is carried between cases, so no case can be influenced by another.
Before each case the script resets the pde-sim case index and makes sure
`artifacts/` is empty: on start-up, anything already there is moved to
`<output>/stale/`. Nothing is deleted.

### Output

`-o DIR` names the output directory. It must be outside `$PETSC_DIR` (compared
by physical path, so a symlink into the tree is refused too). The default is
`~/pdesim-runs/<timestamp>`, one per invocation. It holds:

```
<output>/
  c<N>/<problem>/            one study dir per case: generated source, makefile,
                             specs, results-manifest.json, transcripts,
                             case-result.json, and other-artifacts/ for anything
                             else the pipeline wrote under artifacts/
  logs/c<N>-<problem>.log    the runner's own log for the case
  usage-pdesim-<model>-c<N>.jsonl   one usage record per case
  summary.tsv                one line per case: config, problem, wall time,
                             cost, outcome, run command, study dir
  stale/                     what was in artifacts/ before the run, if anything
  interrupted/               the study of a case stopped by Ctrl-C
  paused/c<N>-<problem>/     a case paused because the LLM budget ran out
```

A case whose `c<N>/<problem>/` already exists is skipped, so rerunning with the
same `-o` fills in only the missing cases.

### Progress

While a case runs, the script prints a progress line every `-i` seconds
summarizing its `transcript.stream.jsonl`: elapsed time, event count, and the
most recent tool call or message. When the case ends it prints its wall time,
`total_cost_usd` (from the `result` event the CLI writes at the end; absent for a
case stopped at `PDESIM_TIMEOUT`), outcome, and run command.

Ctrl-C stops the running case and its `claude` process, and moves what the case
had written to `<output>/interrupted/`.

### When the LLM budget runs out

If the LLM stops being usable partway through a case — the API key's credit
balance is exhausted (`Credit balance is too low`) or a subscription's usage
limit is reached — the CLI ends the run with a `billing_error` or `rate_limit`
error, which it does not retry. The case is then **paused, not failed**:

- The script shows the CLI's message, for example
  `Credit balance is too low · Add funds: https://platform.claude.com/settings/billing`.
- The case's study, everything else it wrote under `artifacts/`, and the pde-sim
  case index are moved to `<output>/paused/c<N>-<problem>/`, and its
  `case-result.json` records the Claude session to continue. Nothing is left in
  the tree, and no later case is started.
- At a terminal, the script waits: add funds (or wait for the limit to reset)
  and press Enter, and it moves the case back to where it ran and continues the
  same session with `claude --resume`, telling the agent to carry on from what is
  on disk instead of starting over. If the budget is still out, it pauses again.
  Type `q` to stop instead.
- Without a terminal, or after `q`, the script exits with status 75 and prints
  the command that continues the run. Rerunning with the same `-o` resumes the
  paused case, skips the finished ones, and runs the rest.

What is kept is the top-level conversation. In config 3, a subagent cut off by
the limit loses its own unfinished work; the resumed orchestrator re-dispatches
that one stage. Time spent paused counts toward neither `wall_sec` nor
`PDESIM_TIMEOUT`.

## Serving the bench's Green agent

`pdesim_purple.py serve` exposes the same run behind the A2A "Purple agent"
contract, for scoring pde-sim with the bench's own Green agent. `run_pdesim.sh`
does not use it. Green sends a plain-text `problem_description`; the server runs
the case under `$PETSC_DIR/artifacts/bench-<context>-<time>` and returns a
`TextPart` (`Code generation successful` / `nsize:` / `cli_args:`) and a
`FilePart` per generated source, main source first. The makefile is not sent;
the bench builds its own.

This mode imports the A2A SDK and the bench's `src.util` shims, so run it from the
bench repo root inside the bench's `uv` env (`uv sync` there first), and start the
bench's components by hand:

```bash
export PETSC_DIR=/path/to/petsc PETSC_ARCH=your-arch ARGO_API_KEY=your_anl_username

# 1) Green agent — :9001
cd ~/petscagent-bench && uv run src/green_agent/server.py &

# 2) PETSc compile/run MCP server — :8080
cd ~/petscagent-bench && \
  uv run python -c "from petsc_compile_run_mcp_server import main; main()" &

# 3) This binding as the Purple agent — :9002
cd ~/petscagent-bench && \
  uv run python $PETSC_DIR/.agents/pde-sim/bindings/petscagent-bench/pdesim_purple.py serve &

# 4) Trigger the run, pointing Green at this binding's URL
cd ~/petscagent-bench && uv run src/client_cli.py \
  --green-url http://localhost:9001 \
  --purple-url http://localhost:9002 \
  --mcp-server-url http://localhost:8080/mcp
```

Results land in the bench's `output/`. Green keeps its own budget for one Purple
call, `purple_budget_sec` in `config/green_agent_config.yaml`; it must exceed
`PDESIM_TIMEOUT`, or Green abandons the call before a stopped case delivers its
partial answer.

## Configuration (environment variables)

| Variable                                | Default                                               | Purpose                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PDESIM_CLAUDE_BIN`                   | `claude`                                            | Claude CLI binary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `PDESIM_REPO_DIR`                     | repo root (inferred)                                  | cwd for the pipeline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `PDESIM_ARTIFACTS_DIR`                | `<repo>/artifacts`                                  | where studies/manifests are searched                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `PDESIM_TIMEOUT` | `86400` (24 h) | seconds a problem may take before the pipeline is stopped; `run_pdesim.sh -t MIN` sets it in minutes and defaults to the same day. Off in practice: what ends a run that has genuinely stopped producing is `CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS`, not this. When it does fire it is a **budget, not a failure**: the agent is told how long it has, and a stopped run still keeps and reports what it finished, marked `(partial: …)` — first its recorded run, else the generated source alone. Only a run that wrote no source fails, leaving a stub manifest saying why. Under `serve`, Green's `purple_budget_sec` must exceed this; see above |
| `CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS` | `1200000` (this binding; CLI default is `600000`) | Claude CLI knob (milliseconds): a Task subagent with no streamed progress for this long is aborted and reported to the orchestrator as a stall. The binding raises the default to 20 min via`setdefault` so a long code-generation step is not reaped mid-run; export your own value to override (e.g. `3600000` for 1 h). There is no documented value that fully disables it — set it large.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `PDESIM_PERMISSION_FLAG`              | `--permission-mode bypassPermissions`               | Claude CLI permission flag                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `PDESIM_CLAUDE_MODEL`                 | *(unset)*                                           | optional`--model` override                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `PDESIM_CLAUDE_EXTRA_ARGS`            | *(unset)*                                           | extra CLI args (shell-split)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `PDESIM_MAX_NSIZE`                    | `64`                                                | cap on returned`nsize` (bench rejects larger)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `PDESIM_CONFIG`                       | `3`                                                 | ablation rung:`1` general agent (no skills, no subagents), `2` general + skills (no subagents), `3` full pde-sim pipeline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `PDESIM_DISALLOWED_TOOLS`             | per-config                                            | Claude CLI`--disallowed-tools` list gating the config; default `Task Skill` (1), `Task` (2), empty (3). Override if the installed CLI names these tools differently.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `PDESIM_USAGE_DIR`                    | *(unset)*                                           | if set, append one JSON record per case (tokens,`total_cost_usd`, `num_turns`, subagent counts, `model_name`, and per-stage attempts — see below) to `<dir>/usage-pdesim-<model>-c<config>.jsonl` — named from the self-reported model; `run_pdesim.sh` sets it to the run's output directory                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `PDESIM_USAGE_LOG`                    | *(unset)*                                           | explicit single-file override for the usage record (used when`PDESIM_USAGE_DIR` is not set)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `PDESIM_SAVE_TRANSCRIPT`              | `1`                                                 | (also gates spec companions) save the run's model I/O to the study dir,**live**, and — at the end — a companion `<name>.md` for each math-bearing spec JSON the pipeline wrote (`problem-spec.json`, `numerical-plan.json`, …), rendering its `*_latex`/formula fields as `$$` blocks so the equations are readable in a Markdown-math viewer. Model I/O: The CLI runs with `--output-format stream-json`, so `transcript.stream.jsonl` is appended one event per line as the run happens (`tail -f` it to watch it grow). Readable `.md` files are re-rendered every ~3s and once at the end: each turn timestamped `[HH:MM:SS]` local, each tool result showing the tool and its duration (e.g. `(Bash, 5.4s)`), plus a same-named `.json` sibling with the full, UNtruncated events (look up any `…[truncated]` there), plus `transcript-stderr.log`. Single-agent runs get `transcript.{md,json}`; config 3 splits by role into `transcript-{orchestrator,pde-modeling,numerical-analysis,code-generation,visualization}.{md,json}` (split by `parent_tool_use_id`, roles from the dispatch prompt). Each per-role file grows **live**: it appears the moment its subagent is dispatched and gains one timestamped progress line per tool the subagent runs (from the CLI's `task_*` events, which are the only material available before the subagent's full turns arrive), so a long code-generation step is visible step-by-step and a stall shows as a growing time gap; the coarse progress lines are superseded by the full detailed turns once the subagent completes. Set `0` to disable |

### What each metric covers

A config-3 run is an orchestrator plus the specialists it dispatches, so a figure
is only meaningful once you know which of them it counts. "Whole run" below means
the orchestrator **and** its subagents; under configs 1 and 2, which dispatch
none, whole-run and orchestrator-only coincide.

| metric | covers | how it combines |
| --- | --- | --- |
| `input_tokens`, `output_tokens`, `total_tokens`, `cached_tokens` | whole run | summed, from the CLI's `modelUsage` breakdown |
| `cost_usd` | whole run | the CLI's `total_cost_usd`, which the `modelUsage` costs sum to |
| `model_calls` | whole run | one per assistant message, counted over the transcript |
| `tool_calls` | whole run | one per `tool_use` block |
| `peak_context_tokens` | whole run | a **maximum** over single turns, not a sum |
| `stage_attempts` | whole run | per-specialist dispatch counts; empty without subagents |
| `code_generation_failures` | the code-generation stage only | failed `build`/`run` journal entries |
| `skill_tool_calls`, `subagent_tool_calls`, `pdesim_skills_read` | whole run | counted over the transcript |
| `subagents_spawned`, `subagents_completed`, `subagents_failed` | the orchestrator's view | as the CLI reports them |
| `num_turns`, `model_responses` | orchestrator only / whole run | the CLI's own turn count, kept for reference, and the response count `model_calls` reports |
| `wall_sec`, `returncode`, `config`, `context_id`, `ts` | the request | per-request context, not a metric |

`cached_tokens` is the cache-read **part of** `input_tokens`, not an addition to
it. A run killed at `PDESIM_TIMEOUT` reports neither `cost_usd` nor a `modelUsage`
breakdown, so its record is flagged `partial` and its token counts fall back to
per-turn sums that overstate input and understate output.

The sidecar spells three of these differently, since it predates the telemetry
contract: `prompt_tokens` is `input_tokens`, `completion_tokens` is
`output_tokens`, and `model_name` is the model the CLI reported, where the
telemetry's `model` is the label built from it (`pdesim-<model>-c<N>`).

### Per-stage attempts in the usage sidecar

Each `PDESIM_USAGE_DIR` record carries two attempt counts, recorded **binding-side
only** (they are deliberately *not* part of the bench `petscagent.telemetry.v1`
contract, so `_build_telemetry` does not forward them to the Green agent — how the
bench should aggregate attempts is left to the bench maintainers):

- `stage_attempts` — a map of how many times the orchestrator dispatched each
  specialist, keyed by the same `role[-stage]` label as the split transcripts
  (e.g. `{"pde-modeling": 1, "numerical-analysis-plan": 1, "numerical-analysis-assess": 1, "code-generation": 1}`). Normally 1 per stage (2
  for numerical-analysis: plan and assess); a value `>1` means the orchestrator
  re-dispatched that stage, e.g. a revise-plan feedback loop. Empty for configs 1–2
  (no subagents).
- `code_generation_failures` — how many entries in the manifest's `attempts[]`
  journal are a failed `build` or `run`, i.e. how often code generation had to fix
  something before the program worked. This counts failures within a dispatch,
  where `stage_attempts["code-generation"]` counts the dispatches themselves. The
  journal's third action, `format`, is not counted: it runs PETSc's formatter and
  succeeds, so it says nothing about the generated code.
  Every config is asked for the journal — config 3 through
  `results-manifest.schema.json`, configs 1–2 through the same shape stated in
  their own contract — so the count compares across the ladder. **Omitted, rather
  than zero, when a manifest carries no `attempts[]`**, since a zero would read as
  a run that failed nothing instead of one that recorded nothing. Runs made before
  configs 1–2 were asked to journal have no entry for this field.

A study stopped before the pipeline wrote `results-manifest.json` gets a stub
one in its place, marked `"written_by": "pdesim_purple binding, not the pde-sim
pipeline"` with the reason it stopped and the sources found. It carries no
`attempts` and no `runs` entries: nothing there was attempted or run by the
agent, and an empty journal would make `code_generation_failures` report zero
rather than stay absent. That also leaves it short of
`results-manifest.schema.json`, which requires both — a stub that conformed
would have to invent exactly the fields that matter. An existing manifest is
never overwritten.

A usage record is written on **every** path — success, timeout, failure, or a
pause for budget — so a run's consumption is never silently lost. Each part of a
resumed case gets its own record, numbered by `part` and flagged `paused` when
that part ended for budget. The CLI's cost and token totals are cumulative over
the session, so the case's last record covers all of its parts: use it alone,
rather than summing the parts. A run killed at `PDESIM_TIMEOUT` emits
no final `result` event and therefore reports no cost, so its record carries
`"partial": true` and `"total_cost_usd": null`; its token counts are summed from
the individual turns, which is how billing actually accrues, so the run can still
be priced after the fact.

## Ablation configurations (`PDESIM_CONFIG`)

The binding doubles as the agent-under-test for the three-rung ablation in
`EXPERIMENT_PLAN.md` (repo root). One code path, selected by `PDESIM_CONFIG`:

| # | Config                      | Gating (`--disallowed-tools`)         | Prompt                                                                                                    |
| - | --------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| 1 | General coding agent        | `Task` (subagent) + `Skill` blocked | plain "write PETSc code"                                                                                  |
| 2 | Single-context pipeline     | `Task` blocked                        | run the pde-sim workflow itself: read the orchestration`SKILL.md` (steps) + domain skills; no subagents |
| 3 | Specialized + scoped skills | none                                    | full pde-sim pipeline, orchestrator dispatches role agents (default)                                      |

Config 2 reads the **same** pde-sim knowledge as config 3 — the orchestration
`SKILL.md` for the end-to-end workflow plus the domain skills — but performs every
step itself in one context (the dispatch parts of the orchestration skill are inert
without the Agent tool). So **2→3 isolates agent specialization alone**, not the
knowledge available; **1→2 isolates the pde-sim knowledge** (workflow + skills).

All three run at the **repo root**, so they share identical project context
(`CLAUDE.md`/`AGENTS.md`); the only thing that changes across the ladder is the
tool gating. The output/deliverable contract (output discipline, direct run
command, manifest, `PDESIM_DONE` marker) is shared verbatim across all three so
scoring differences reflect the wiring, not prompt wording.

Gating **verified** against Claude CLI 2.1.281: `--disallowed-tools "Task Skill"`
removes both the subagent tool (displayed as `Agent`; `Task` is its disallow alias)
and `Skill`; `--disallowed-tools Task` removes the subagent tool while keeping
`Skill`. Names stay overridable via `PDESIM_DISALLOWED_TOOLS` should a future CLI
rename them.

## Known limitations / things to watch

- **Grader mismatch is the main risk.** The bench's `numerical_accuracy` metric
  reads the *trailing N numeric lines* of stdout (N = length of the problem's
  hidden `expected_output`) and compares them to that reference; only
  `problem_description` is given to the agent — `test_cases` (`args` /
  `expected_output`) are not. The binding's prompt therefore tells pde-sim to
  emit **only** the output the description asks for (e.g. the final solution via
  `VecView()`) and to keep extra numeric diagnostics off stdout. A correct
  solution can still miss the metric if the description's output format is
  ambiguous; remaining cases need per-problem handling in the bench evaluators.
- **Run selection is a heuristic.** The bench runs the reported `cli_args` with
  each test case's own arguments appended, so this choice decides how the graded
  program is invoked. Of the runs the pipeline recorded, the binding drops those
  that verify the method rather than answer the problem (an `-mms` run reports an
  error norm, not the solution), prefers one the agent labelled `production` or
  `deliverable`, then takes the most `mpi_ranks` — ties to the earlier run, and a
  run above `PDESIM_MAX_NSIZE` passed over where possible. Each filter yields if
  it would leave nothing. The reported `mpi_ranks` is used only by a test case
  that declares no `nsize`. If a problem needs a different run, adjust
  `_pick_run`.
- **cli_args are passed verbatim** after stripping the launcher/exe prefix,
  including MMS-specific flags (e.g. `-mms_levels 4`, `-mms_csv convergence.csv`);
  the latter just writes a file in the bench work dir.
- **A2A SDK / telemetry (`serve` only).** The bench uses the protobuf-based
  **a2a-sdk 1.x**; `serve` imports its A2A types and helpers from the bench's own
  shims (`src.util.a2a_v1`, `src.util.telemetry`), so it must be launched from the
  bench repo root (it adds the cwd to `sys.path`). `run` imports none of them. It reports token/cost/effort to the
  bench in a versioned data Part (`petscagent.telemetry.v1`) via `_build_telemetry()`.
- **Cost/time.** A full pde-sim pipeline (multi-agent, MMS studies, retries) uses
  far more tokens/time per problem than a single-shot purple agent. Both the
  reported token counts and `cost_usd` cover the **whole run, subagents included**:
  they are read from the CLI's `modelUsage` breakdown, whose per-model costs sum to
  `total_cost_usd`. For a config-3 study that is several times the top-level
  agent's own usage — one Darcy run spent 7.2M input tokens against the top-level
  agent's 0.5M.
- **`bypassPermissions` runs arbitrary Bash** — only run against a sandboxed
  PETSc build.
