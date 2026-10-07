"""pde-sim Purple-agent binding for petscagent-bench.

Exposes the pde-sim multi-agent pipeline behind the A2A "Purple agent" contract
that petscagent-bench's Green agent expects, without modifying any bench code.

The Green agent talks to a Purple agent purely over A2A/HTTP by URL: it sends a
plain-text problem description and expects back a single message containing

  * exactly one TextPart shaped as
        Code generation successful ...
        nsize: <int>
        cli_args: <string>
  * one or more FileParts holding the generated PETSc source (first = main).

This server satisfies that contract by driving pde-sim non-interactively with
the Claude CLI (``claude -p`` in autonomous mode), then harvesting the generated
source and the run parameters from the pipeline's ``results-manifest.json``
(schema: .agents/pde-sim/contracts/results-manifest.schema.json).

Run it from an environment that has the A2A server deps installed (the simplest
is petscagent-bench's own uv env); see the sibling README.md for the full recipe.
"""

import argparse
import asyncio
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from starlette.applications import Starlette
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore

# The bench migrated to the protobuf-based A2A 1.x SDK and provides its types and
# the telemetry contract through local shims (src.util.*). This binding is
# launched from the bench repo root (see README), so add the cwd to sys.path to
# resolve `src.*`, exactly as the bench's own agents rely on.
sys.path.insert(0, os.getcwd())
from src.util.a2a_v1 import (  # noqa: E402  (import after sys.path fix-up)
    AgentCapabilities, AgentCard, AgentInterface, AgentSkill,
    new_agent_parts_message, new_data_part, new_raw_part, new_text_part,
)
from src.util.telemetry import PURPLE_TELEMETRY_SCHEMA  # noqa: E402

# Repo root: this file lives at <repo>/.agents/pde-sim/bindings/petscagent-bench/pdesim_purple.py
REPO_DIR = Path(__file__).resolve().parents[4]

# Source suffixes forwarded to the bench's MCP uploader as compile dependencies
# (see petscagent-bench src/green_agent/agent.py::_create_files_on_server). A
# second ``.c`` would collide with the main after renaming, so only these extra
# kinds are forwarded as dependency files.
#
# KNOWN LIMITATION: the bench only *compiles* the dependency kinds it special-
# cases, and those are ``.cu`` and ``.kokkos.cpp`` — NOT ``.kokkos.cxx``. PETSc's
# own Kokkos convention is ``.kokkos.cxx`` (the tree has 28 of those and zero
# ``.kokkos.cpp``), so a generated ``.kokkos.cxx`` is uploaded but never compiled
# by the bench, yielding an undiagnosed link failure. Fixing that needs a change
# in the bench source, out of scope for this binding; until the bench recognizes
# ``.kokkos.cxx``, emit Kokkos dependencies as ``.kokkos.cpp``.
DEP_SUFFIXES = (".cu",)
DEP_COMPOUND_SUFFIXES = (".kokkos.cxx", ".kokkos.cpp")
MAIN_SUFFIXES = (".c", ".cxx", ".cpp", ".cu")

# Build-system drivers that must NOT appear as the executable in runs[].command.
# The bench compiles the source itself and runs the resulting binary directly, so
# the command must be the real invocation (e.g. 'mpiexec -n 4 ./app -opt val'),
# not a makefile target like 'make run' — parsing that would yield cli_args='run'
# and run the program default-configured, silently corrupting the bench score.
BUILD_TOOLS = {"make", "gmake", "bmake", "cmake", "ninja", "meson", "bear"}


def _env(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val else default


class Settings:
    """Runtime knobs, all overridable by environment variable."""

    def __init__(self) -> None:
        self.claude_bin = _env("PDESIM_CLAUDE_BIN", "claude")
        self.repo_dir = Path(_env("PDESIM_REPO_DIR", str(REPO_DIR)))
        self.artifacts_dir = Path(_env("PDESIM_ARTIFACTS_DIR", str(self.repo_dir / "artifacts")))
        # Hard ceiling: the bench's Green->Purple HTTP read timeout, hardcoded at
        # read=3000.0 in src/util/a2a_comm.py with no env override. Past that Green
        # gives up on the request, so the budget here must leave room to kill the
        # CLI, save the transcript and answer. 2900s keeps ~100s of headroom.
        # NOTE the full config-3 pipeline can exceed even this on the heavier
        # problems; raising it further requires raising that bench-side read timeout.
        self.timeout_sec = float(_env("PDESIM_TIMEOUT", "2900"))
        # Use os.environ.get, NOT _env: an explicit PDESIM_PERMISSION_FLAG="" must
        # DROP the flag (run under Claude's default, safer permissions), not fall
        # back to the default and silently reinstate bypassPermissions.
        self.permission_flag = os.environ.get("PDESIM_PERMISSION_FLAG", "--permission-mode bypassPermissions")
        self.claude_model = os.environ.get("PDESIM_CLAUDE_MODEL")  # optional --model
        self.extra_args = os.environ.get("PDESIM_CLAUDE_EXTRA_ARGS", "")  # optional passthrough
        # Bench rejects nsize outside [1, max_nsize] (default 64).
        self.max_nsize = int(_env("PDESIM_MAX_NSIZE", "64"))

        # Ablation configuration selecting how much pde-sim machinery is enabled
        # (see EXPERIMENT_PLAN.md in petscagent-bench):
        #   1 = general coding agent: one context, NO skills, NO subagents
        #   2 = general agent + reusable skills: one context, skills, NO subagents
        #   3 = specialized agents + scoped skills: full pde-sim pipeline (default)
        self.config = _env("PDESIM_CONFIG", "3")
        if self.config not in ("1", "2", "3"):
            raise ValueError(f"PDESIM_CONFIG must be 1, 2, or 3; got {self.config!r}")
        # Tools blocked when launching the Claude CLI, gating the ablation:
        # blocking the subagent tool forces a single context (configs 1-2); blocking
        # the Skill tool additionally removes skills (config 1). Verified against
        # Claude CLI 2.1.281: the subagent tool is displayed as "Agent" but the
        # disallow name "Task" still removes it (Task is its alias), and "Skill"
        # removes the Skill tool; "--disallowed-tools" accepts a space-separated
        # value, so the single token below is split by the CLI. The defaults stay
        # overridable in case a future CLI renames these tools. Use os.environ.get
        # so an explicit "" DROPS all gating rather than falling back to the default.
        _default_disallowed = {"1": "Task Skill", "2": "Task", "3": ""}[self.config]
        self.disallowed_tools = os.environ.get("PDESIM_DISALLOWED_TOOLS", _default_disallowed)


SETTINGS = Settings()


def _build_prompt(problem_description: str, study_dir: Path) -> str:
    """Compose the autonomous pde-sim driver prompt."""
    return (
        "You are running the PETSc pde-sim multi-agent pipeline non-interactively as the "
        "code-generation target for an external benchmark.\n\n"
        "Phenomenon / problem description:\n"
        "<<<\n"
        f"{problem_description}\n"
        ">>>\n\n"
        "Instructions:\n"
        "- Follow the pde-sim pipeline defined by .claude/commands/pde-sim.md and "
        ".agents/pde-sim/skills/orchestration/SKILL.md.\n"
        "- Run with autonomy = autonomous: do NOT stop at any human-in-the-loop gate; make "
        "reasonable default choices and proceed to a verified PETSc program.\n"
        "- There is NO human in this run. NEVER end your turn to ask a question or wait for input. "
        "If a dispatched subagent returns something like '[Request interrupted by user for tool use]', "
        "stalls, hits its no-progress watchdog, times out, or fails, treat it as a transient signal — "
        "do NOT stop and do NOT ask the user: re-dispatch that stage once with a tighter instruction, "
        "then continue the pipeline.\n"
        "- To avoid the subagent no-progress stall watchdog on large programs, instruct code-generation "
        "to build incrementally — several smaller Write/Edit steps with frequent build checkpoints — "
        "rather than emitting one very large file in a single step.\n"
        "- ALWAYS finish by writing results-manifest.json and printing the PDESIM_DONE line, EVEN IF a "
        "stage ultimately fails: if you cannot produce a verified program, still emit a "
        "results-manifest.json that conforms to the schema with the build/run status set to \"failure\" "
        "and the \"attempts\" journal populated, so the run always ends with a manifest rather than an "
        "unanswered question.\n"
        f"- Write ALL artifacts (problem-spec, numerical-plan, generated source, makefile, "
        f"results-manifest.json, logs) under this directory: {study_dir}\n"
        "- The deliverable is a compiled, MMS-verified PETSc program plus a results-manifest.json "
        "that conforms to .agents/pde-sim/contracts/results-manifest.schema.json, including at "
        'least one entry in "runs" with its full "command" and "mpi_ranks".\n'
        '- The "command" MUST be the direct program invocation (e.g. "mpiexec -n 4 ./app -opt val"), '
        'NOT a makefile target like "make run": the benchmark parses it to recover the executable and '
        "its CLI arguments, and a build-system target would be mis-run with the wrong arguments. Build "
        "through the makefile, but record the underlying run command here.\n"
        "- Prefer a single self-contained PETSc C source file for the main program.\n"
        "- Output discipline: an external grader compares the program's TRAILING numeric stdout "
        "lines against a hidden reference. Emit on stdout ONLY the program output the problem "
        "description explicitly asks for (e.g. the final solution via VecView()). Do NOT print extra "
        "numeric lines to stdout (MMS/convergence errors, norms, timings, iteration counts, "
        "residuals); send any such diagnostics to stderr or a file, or omit them. Match the "
        "description's stated grid/size, scheme, and boundary/initial conditions exactly.\n"
        "- Do NOT launch the visualization stage (it is opt-in per D20); no plots or renders are needed.\n"
        "- When finished, print EXACTLY one final line:\n"
        "      PDESIM_DONE <absolute path to results-manifest.json>\n"
    )


# Output/deliverable rules shared VERBATIM by every ablation config, so that
# scoring differences reflect the agent wiring under test and not incidental
# prompt wording. The grader compares the program's trailing numeric stdout lines
# against a hidden reference, so the output-discipline text below must be identical
# for configs 1, 2, and 3.
_SHARED_CONTRACT = (
    "- Write ALL artifacts (generated source, makefile, results-manifest.json, logs) "
    f"under this directory: {{study_dir}}\n"
    "- Prefer a single self-contained PETSc C source file for the main program.\n"
    '- Build through a makefile that uses $PETSC_DIR/$PETSC_ARCH, but record in the '
    'manifest the DIRECT program invocation (e.g. "mpiexec -n 4 ./app -opt val"), '
    'NOT a makefile target like "make run": the benchmark parses the command to '
    "recover the executable and its CLI arguments.\n"
    "- Output discipline: an external grader compares the program's TRAILING numeric stdout "
    "lines against a hidden reference. Emit on stdout ONLY the program output the problem "
    "description explicitly asks for (e.g. the final solution via VecView()). Do NOT print extra "
    "numeric lines to stdout (MMS/convergence errors, norms, timings, iteration counts, "
    "residuals); send any such diagnostics to stderr or a file, or omit them. Match the "
    "description's stated grid/size, scheme, and boundary/initial conditions exactly.\n"
    "- Do NOT produce any plots, renders, or visualizations.\n"
    '- Write a results-manifest.json in that directory with a "runs" array; each run entry '
    'must have "command" (the direct invocation string above), "mpi_ranks" (integer), and '
    '"status": "success" once it runs cleanly.\n'
    "- When finished, print EXACTLY one final line:\n"
    "      PDESIM_DONE <absolute path to results-manifest.json>\n"
)


def _build_prompt_general(problem_description: str, study_dir: Path, with_skills: bool) -> str:
    """Compose the single-context (config 1 / config 2) driver prompt.

    Config 1 is a plain PETSc coding agent. Config 2 additionally consults the
    pde-sim knowledge: the orchestration SKILL.md for the end-to-end WORKFLOW and
    the domain skills for method detail. It performs every step ITSELF in one
    context (the dispatch parts of the orchestration skill do not apply) — so the
    only difference from config 3 is agent specialization, not the knowledge
    available. Neither config dispatches specialized subagents.
    """
    skills_clause = (
        "- Follow the pde-sim workflow in THIS SINGLE context. Read "
        ".agents/pde-sim/skills/orchestration/SKILL.md for the end-to-end steps "
        "(modeling, numerical analysis / discretization, scheme & solver choice, "
        "code generation, MMS verification), and read these domain skills as needed:\n"
        "    .agents/pde-sim/skills/pde-formulation/SKILL.md\n"
        "    .agents/pde-sim/skills/numerical-methods/SKILL.md\n"
        "    .agents/pde-sim/skills/petsc-codegen/SKILL.md\n"
        "    .agents/pde-sim/skills/petsc-solvers/SKILL.md\n"
        "- Perform EVERY step yourself: do NOT dispatch or coordinate subagents, and "
        "ignore the parts of the orchestration skill about delegating to separate agents.\n"
        if with_skills
        else "- Work from your own PETSc knowledge in this single context; do NOT consult "
        "external skills and do NOT delegate to subagents.\n"
    )
    return (
        "You are a PETSc code-generation agent producing a verified program for an "
        "external benchmark.\n\n"
        "Problem description:\n"
        "<<<\n"
        f"{problem_description}\n"
        ">>>\n\n"
        "Instructions:\n"
        "- Write a PETSc program that solves the problem, build it, run it, and confirm it "
        "runs cleanly.\n"
        f"{skills_clause}"
        + _SHARED_CONTRACT.format(study_dir=study_dir)
    )


def _prompt_for(problem_description: str, study_dir: Path) -> str:
    """Select the driver prompt for the active ablation configuration."""
    if SETTINGS.config == "3":
        return _build_prompt(problem_description, study_dir)
    return _build_prompt_general(problem_description, study_dir, with_skills=SETTINGS.config == "2")


def _scan_events(events: List[Any]) -> Dict[str, Any]:
    """Evidence scanned from the transcript's assistant turns.

    Shared by the normal path (which takes authoritative totals from the final
    ``result`` event) and the timeout path (which has no result event, so these
    per-turn sums are the only record of what the killed run consumed). Summing
    per-turn usage is the correct billing aggregation: every request re-bills its
    fresh and cached input, which is how the CLI derives its own cost figure.

    Collects positive evidence of what actually happened, so the ablation can
    VERIFY a config rather than trust the gating: Skill-tool calls, subagent
    (Agent/Task) tool calls, and Reads of pde-sim SKILL.md files (the pde-sim
    skills are loaded by reading files, not via the Skill tool).
    """
    skill_tool_calls = 0
    subagent_tool_calls = 0
    tool_calls_total = 0
    peak_context = 0
    sum_input = 0
    sum_cache_creation = 0
    sum_cache_read = 0
    sum_output = 0
    skills_read: set = set()
    # Per-stage attempt counts: how many times the orchestrator dispatched each
    # specialist (keyed by the same role[-stage] label as the split transcripts, so
    # numerical-analysis's plan and assess are counted separately). Normally 1 per
    # stage (2 for numerical-analysis); a value >1 for a stage means the orchestrator
    # re-dispatched it, e.g. a revise-plan feedback loop. Empty for configs 1-2 (no
    # subagents). This is our-side bookkeeping for the sidecar; it is NOT part of the
    # bench telemetry contract, so _build_telemetry does not forward it to Green.
    stage_attempts: Dict[str, int] = {}
    marker = ".agents/pde-sim/skills/"
    for e in events:
        if not (isinstance(e, dict) and e.get("type") == "assistant"):
            continue
        msg = e.get("message", {}) or {}
        for c in msg.get("content", []) or []:
            if not (isinstance(c, dict) and c.get("type") == "tool_use"):
                continue
            tool_calls_total += 1
            name = c.get("name")
            if name == "Skill":
                skill_tool_calls += 1
            elif name in ("Agent", "Task"):
                subagent_tool_calls += 1
                inp_a = c.get("input") or {}
                role = _subagent_role(inp_a) or "subagent"
                st = _role_stage(role, inp_a)
                label = f"{role}-{st}" if st else role
                stage_attempts[label] = stage_attempts.get(label, 0) + 1
            elif name == "Read":
                fp = (c.get("input") or {}).get("file_path", "") or ""
                if marker in fp and fp.endswith("SKILL.md"):
                    skills_read.add(fp.split(marker, 1)[1].split("/", 1)[0])
        mu = msg.get("usage") or {}
        turn_in = int(mu.get("input_tokens", 0) or 0)
        turn_cc = int(mu.get("cache_creation_input_tokens", 0) or 0)
        turn_cr = int(mu.get("cache_read_input_tokens", 0) or 0)
        sum_input += turn_in
        sum_cache_creation += turn_cc
        sum_cache_read += turn_cr
        sum_output += int(mu.get("output_tokens", 0) or 0)
        # Peak context = largest single-turn input (fresh + cached), i.e. how big
        # the context window got — the "context size" efficiency metric.
        ctx = turn_in + turn_cr + turn_cc
        if ctx > peak_context:
            peak_context = ctx

    return {
        "skill_tool_calls": skill_tool_calls,
        "subagent_tool_calls": subagent_tool_calls,
        "tool_calls": tool_calls_total,
        "peak_context_tokens": peak_context,
        "pdesim_skills_read": sorted(skills_read),
        "stage_attempts": stage_attempts,
        "sum_input_tokens": sum_input,
        "sum_cache_creation_tokens": sum_cache_creation,
        "sum_cache_read_tokens": sum_cache_read,
        "sum_output_tokens": sum_output,
    }


def _partial_usage(events: List[Any]) -> Optional[Dict[str, Any]]:
    """Usage for a run that was killed before emitting its final ``result`` event.

    A timed-out run never reports ``total_cost_usd``, so that stays None and the
    record is flagged ``partial``; the token sums still price the run after the
    fact, which a bare timeout would otherwise leave unrecorded entirely.
    """
    if not isinstance(events, list) or not events:
        return None
    scan = _scan_events(events)
    sysinit = next((e for e in events if isinstance(e, dict) and e.get("type") == "system"), {})
    prompt_tokens = (scan["sum_input_tokens"] + scan["sum_cache_creation_tokens"]
                     + scan["sum_cache_read_tokens"])
    return {
        "partial": True,  # killed run: token sums are per-request totals, cost unknown
        "prompt_tokens": prompt_tokens,
        "completion_tokens": scan["sum_output_tokens"],
        "total_tokens": prompt_tokens + scan["sum_output_tokens"],
        "cached_tokens": scan["sum_cache_read_tokens"],
        "total_cost_usd": None,  # the CLI reports cost only in the result event
        "num_turns": None,
        "subagents_spawned": None,
        "subagents_completed": None,
        "subagents_failed": None,
        "tool_calls": scan["tool_calls"],
        "peak_context_tokens": scan["peak_context_tokens"],
        "model_name": sysinit.get("model"),
        "skill_tool_calls": scan["skill_tool_calls"],
        "subagent_tool_calls": scan["subagent_tool_calls"],
        "pdesim_skills_read": scan["pdesim_skills_read"],
        "stage_attempts": scan["stage_attempts"],
    }


class PdeSimTimeout(RuntimeError):
    """Pipeline exceeded PDESIM_TIMEOUT. Carries the killed run's partial usage so
    the caller can still record what it consumed."""

    def __init__(self, message: str, usage: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.usage = usage


def _extract_result(events: List[Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Extract (final_text, usage_info) from the parsed transcript events.

    Events are the ``stream-json`` lines (system / assistant / user / result). The
    final ``type=="result"`` event carries the assistant's final text (``result``),
    token ``usage``, and ``total_cost_usd``; usage_info normalizes
    tokens/cost/turns/subagent counts. Returns ("", None) if there is no result.
    """
    if not isinstance(events, list) or not events:
        return "", None
    result = next((e for e in reversed(events)
                   if isinstance(e, dict) and e.get("type") == "result"), None)
    if result is None:
        return "", None

    text = result.get("result")
    if not isinstance(text, str):
        # Fall back to concatenating assistant text parts if 'result' is absent.
        parts: List[str] = []
        for e in events:
            if isinstance(e, dict) and e.get("type") == "assistant":
                for c in (e.get("message", {}) or {}).get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
        text = "\n".join(parts)

    # The actual model that ran (even when unpinned): the CLI reports it in the
    # system init event, or as the modelUsage key. Prefer this over the configured
    # PDESIM_CLAUDE_MODEL so the label records what really executed.
    sysinit = next((e for e in events if isinstance(e, dict) and e.get("type") == "system"), {})
    model_name = sysinit.get("model") or next(iter(result.get("modelUsage") or {}), None)

    u = result.get("usage") or {}
    inp = int(u.get("input_tokens", 0) or 0)
    cc = int(u.get("cache_creation_input_tokens", 0) or 0)
    cr = int(u.get("cache_read_input_tokens", 0) or 0)
    out = int(u.get("output_tokens", 0) or 0)
    prompt_tokens = inp + cc + cr  # all input, including cached, for Green's "prompt"
    stats = result.get("subagent_stats") or {}

    scan = _scan_events(events)

    usage_info = {
        # Token counts are the TOP-LEVEL agent only; subagents (config 3) are not
        # summed into usage, so total_cost_usd is the trustworthy total-run figure.
        "prompt_tokens": prompt_tokens,
        "completion_tokens": out,
        "total_tokens": prompt_tokens + out,
        "cached_tokens": cr,
        "total_cost_usd": result.get("total_cost_usd"),
        "num_turns": result.get("num_turns"),
        "subagents_spawned": stats.get("spawned"),
        "subagents_completed": stats.get("completed"),
        "subagents_failed": stats.get("failed"),
        "tool_calls": scan["tool_calls"],
        "peak_context_tokens": scan["peak_context_tokens"] or prompt_tokens,
        "model_name": model_name,  # actual model that ran (from the transcript)
        # Positive evidence of skill / subagent usage (for ablation verification).
        "skill_tool_calls": scan["skill_tool_calls"],
        "subagent_tool_calls": scan["subagent_tool_calls"],
        "pdesim_skills_read": scan["pdesim_skills_read"],
        # Per-stage dispatch counts (see _scan_events). code_generation's own
        # build/run/fix iteration count is added later from the manifest's attempts[].
        "stage_attempts": scan["stage_attempts"],
    }
    return text, usage_info


# ANSI/VT100 escape sequences (e.g. color codes like "\x1b[01;34m" from `ls`),
# stripped from the rendered Markdown so captured terminal output stays readable.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _parse_ts(s: Any) -> Optional[datetime]:
    """Parse an ISO-8601 event timestamp (…Z); return None if unparseable."""
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


# System-event subtypes the CLI streams for a dispatched subagent (Task): the
# dispatch, per-step progress (each tool the subagent runs), and completion. They
# arrive tagged with the dispatch tool_use_id LIVE, before the subagent's full turns
# (which appear only at completion), so they are the only material available to grow
# a per-subagent transcript while it runs.
_TASK_SUBTYPES = ("task_started", "task_progress", "task_updated", "task_notification")


def _render_transcript_md(events: List[Any]) -> str:
    """Render the CLI transcript events as a readable Markdown conversation.

    Headers carry the local event time [HH:MM:SS]; each tool result shows the tool
    name and how long the call took (result time − tool_use time). While a subagent
    is still running, its coarse per-step progress lines stand in for the detailed
    turns; once those turns arrive at completion they supersede the progress lines.
    """
    def trunc(s: Any, n: int = 2000) -> str:
        s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False, indent=2)
        s = _ANSI_RE.sub("", s)
        return s if len(s) <= n else s[:n] + f"\n…[truncated {len(s) - n} chars]"

    def clk(dt: Optional[datetime]) -> str:
        # timestamps are UTC (…Z); astimezone() with no arg converts to local time.
        return dt.astimezone().strftime("%H:%M:%S") if dt else "--:--:--"

    tool_start: Dict[str, tuple] = {}  # tool_use_id -> (name, start_dt)
    # The system init event carries no timestamp, so use the earliest event that does.
    run_start = next((dt for dt in (_parse_ts(x.get("timestamp") or x.get("_recv_ts")) for x in events
                                    if isinstance(x, dict)) if dt), None)
    # Once the subagent's detailed turns are present, drop the coarse progress lines.
    has_detail = any(isinstance(x, dict) and x.get("type") in ("assistant", "user") for x in events)
    out: List[str] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        t = e.get("type")
        ts = _parse_ts(e.get("timestamp") or e.get("_recv_ts"))
        if t == "system":
            sub = e.get("subtype")
            if sub in _TASK_SUBTYPES:
                if sub == "task_started":
                    out += [f"### [{clk(ts)}] ▷ dispatched · {e.get('description', '')}", ""]
                elif sub == "task_progress" and not has_detail:
                    u = e.get("usage", {}) or {}
                    tn = e.get("last_tool_name")
                    tail = f" ({tn})" if tn else ""
                    meta = f" · tools={u.get('tool_uses')} tokens={u.get('total_tokens')}" if u else ""
                    out += [f"- [{clk(ts)}] {e.get('description', '')}{tail}{meta}"]
                elif sub == "task_updated":  # task_notification duplicates this; skip it
                    patch = e.get("patch", {}) or {}
                    status = patch.get("status") or e.get("status") or "?"
                    line = f"### [{clk(ts)}] ■ status={status}"
                    if patch.get("error"):
                        line += f" — {patch['error']}"
                    out += [line, ""]
                continue
            # The init event (carries model/cwd/tools) heads the transcript.
            start_local = run_start.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if run_start else "?"
            out += [f"# pde-sim run transcript", "",
                    f"- model: `{e.get('model')}`",
                    f"- cwd: `{e.get('cwd')}`",
                    f"- tools: {len(e.get('tools') or [])}",
                    f"- start: {start_local} (times below are local)", ""]
        elif t in ("assistant", "user"):
            content = (e.get("message", {}) or {}).get("content")
            if isinstance(content, str):
                out += [f"### [{clk(ts)}] {t}", trunc(content), ""]
                continue
            for c in content or []:
                if not isinstance(c, dict):
                    continue
                ct = c.get("type")
                if ct == "text":
                    out += [f"### [{clk(ts)}] {t}", trunc(c.get("text", "")), ""]
                elif ct == "tool_use":
                    if c.get("id"):
                        tool_start[c["id"]] = (c.get("name"), ts)
                    out += [f"### [{clk(ts)}] {t} → tool: {c.get('name')}", "```json",
                            trunc(c.get("input", {}), 1500), "```", ""]
                elif ct == "tool_result":
                    name, start = tool_start.get(c.get("tool_use_id"), (None, None))
                    dur = f"{(ts - start).total_seconds():.1f}s" if (ts and start) else "?"
                    meta = f" ({name}, {dur})" if name else ""
                    res = c.get("content")
                    if isinstance(res, list):
                        res = "\n".join(x.get("text", "") for x in res
                                        if isinstance(x, dict) and x.get("type") == "text")
                    out += [f"### [{clk(ts)}] tool result{meta}", "```", trunc(res, 1500), "```", ""]
        elif t == "result":
            u = e.get("usage", {}) or {}
            wall = f" · wall={e['duration_ms'] / 1000:.1f}s" if isinstance(e.get("duration_ms"), (int, float)) else ""
            out += ["---",
                    f"**result**: {e.get('subtype')} · turns={e.get('num_turns')} · "
                    f"cost=${e.get('total_cost_usd')}{wall} · in={u.get('input_tokens')} "
                    f"out={u.get('output_tokens')} cache_read={u.get('cache_read_input_tokens')}",
                    "", "**final text:**", trunc(e.get("result", ""), 4000)]
    return "\n".join(out) + "\n"


# pde-sim subagent role names, used verbatim in transcript filenames. The
# orchestrator dispatches each role as a general-purpose subagent whose prompt
# names the role and its .agents/pde-sim/agents/<role>.md, so the role is
# detectable from the prompt text.
_ROLE_NAMES = ("pde-modeling", "numerical-analysis", "code-generation", "visualization")


def _subagent_role(agent_input: Dict[str, Any]) -> Optional[str]:
    """Role name for a dispatched subagent, from its Agent-tool input.

    Uses the authoritative self-identification first — "You are the <role>
    specialist" and the agents/<role>.md path in the role prompt — because a role's
    prompt also *references* other roles (e.g. code-generation reads the
    numerical-analysis plan), which a bare substring scan would match by mistake.
    """
    text = ((agent_input.get("description") or "") + " " + (agent_input.get("prompt") or "")).lower()
    m = re.search(r"you are the \*{0,2}([a-z][a-z-]+?)\*{0,2}\s+specialist", text)
    if m and m.group(1) in _ROLE_NAMES:
        return m.group(1)
    m = re.search(r"agents/([a-z-]+)\.md", text)
    if m and m.group(1) in _ROLE_NAMES:
        return m.group(1)
    for name in _ROLE_NAMES:                       # last resort: first role mentioned
        if name in text:
            return name
    return None


def _role_stage(role: Optional[str], agent_input: Dict[str, Any]) -> Optional[str]:
    """Pipeline-stage suffix for a role dispatched in more than one stage.

    numerical-analysis runs twice: once as the 'plan' step (before code-generation)
    and once as the 'assess' step (after the run), so its two transcripts are named
    ...-plan and ...-assess rather than a bare counter. The orchestrator's short
    dispatch description is the most reliable signal, so check it before the prompt.
    """
    if role != "numerical-analysis":
        return None
    for src in (agent_input.get("description") or "", agent_input.get("prompt") or ""):
        s = src.lower()
        if "assess" in s:
            return "assess"
        if "plan" in s:
            return "plan"
    return None


def _write_transcripts(study_dir: Path, events: List[Any]) -> None:
    """(Re)write the readable transcripts from the current events.

    Every Markdown file has a same-named ``.json`` sibling holding the full,
    UNtruncated events, so any ``…[truncated]`` marker can be looked up. Single-agent
    runs get ``transcript.{md,json}``; when subagents were dispatched (config 3) the
    inlined subagent turns (tagged ``parent_tool_use_id``) are split by role into
    ``transcript-<role>.{md,json}`` (orchestrator, pde-modeling, numerical-analysis,
    code-generation, visualization). Prior rendered files are cleared first so the
    set always reflects the current state (e.g. as the first subagent appears).
    """
    if not events:
        return
    # Clear previously-rendered files; keep the live stream and stderr logs.
    for p in [study_dir / "transcript.md", study_dir / "transcript.json"]:
        p.unlink(missing_ok=True)
    for p in list(study_dir.glob("transcript-*.md")) + list(study_dir.glob("transcript-*.json")):
        p.unlink(missing_ok=True)

    def _dump(base: Path, evs: List[Any]) -> None:
        base.with_suffix(".json").write_text(json.dumps(evs, indent=2, ensure_ascii=False))
        base.with_suffix(".md").write_text(_render_transcript_md(evs))

    # Map each Agent/Task tool_use id -> label. A role dispatched in distinct stages
    # (numerical-analysis: plan vs assess) gets a stage suffix; any other repeat of
    # the same label gets a numeric suffix.
    id_role: Dict[str, str] = {}
    label_counts: Dict[str, int] = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        for c in ((e.get("message", {}) or {}).get("content") or []):
            if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") in ("Agent", "Task"):
                inp = c.get("input") or {}
                role = _subagent_role(inp) or "subagent"
                stage = _role_stage(role, inp)
                base = f"{role}-{stage}" if stage else role
                label_counts[base] = label_counts.get(base, 0) + 1
                tag = base if label_counts[base] == 1 else f"{base}-{label_counts[base]}"
                if c.get("id"):
                    id_role[c["id"]] = tag

    # Partition events into role groups. A subagent's full turns arrive at completion
    # tagged with parent_tool_use_id; its coarse per-step progress arrives LIVE as
    # system task_* events tagged with the dispatch tool_use_id (and no
    # parent_tool_use_id). Route BOTH to the subagent's group so its transcript grows
    # while it runs; everything else (the orchestrator's own turns, the init/result
    # events) stays at top level.
    groups: Dict[str, List[Any]] = {}
    for e in events:
        if not isinstance(e, dict):
            continue
        pid = e.get("parent_tool_use_id")
        if pid:
            label = id_role.get(pid, "subagent")
        elif e.get("type") == "system" and e.get("subtype") in _TASK_SUBTYPES and e.get("tool_use_id") in id_role:
            label = id_role[e["tool_use_id"]]
        else:
            label = "orchestrator"
        groups.setdefault(label, []).append(e)

    if len(groups) <= 1:
        _dump(study_dir / "transcript.md", events)          # single-agent run
    else:
        for label, evs in groups.items():
            # Skip groups with neither a conversation turn nor live progress (e.g. a
            # stray event whose parent wasn't a role-detected dispatch).
            if not any(isinstance(e, dict) and (e.get("type") in ("assistant", "user")
                       or (e.get("type") == "system" and e.get("subtype") in _TASK_SUBTYPES))
                       for e in evs):
                continue
            _dump(study_dir / f"transcript-{label}.md", evs)  # one .md + .json per role


def _save_transcript(study_dir: Path, events: List[Any], err_b: bytes) -> None:
    """Write stderr + the readable transcripts, best-effort (never fails the run)."""
    try:
        study_dir.mkdir(parents=True, exist_ok=True)
        if err_b:
            (study_dir / "transcript-stderr.log").write_bytes(err_b)
        _write_transcripts(study_dir, events)
    except OSError as e:  # noqa: BLE001 - diagnostics must not fail the request
        print(f"@@@ pde-sim purple: could not save transcript to {study_dir}: {e}", flush=True)


# Keys whose string is a standalone formula worth typesetting as a $$ block. Prose
# keys (notes/details/description) are deliberately excluded: they are sentences with
# embedded LaTeX fragments, which must not be wrapped whole in $$ (it breaks MathJax).
_MATH_VALUE_KEYS = {"value", "expression"}


def _json_math_fields(obj: Any) -> List[tuple]:
    """Collect (label, latex) for every standalone-equation field in a spec JSON.

    A field qualifies if its key ends in 'latex', or it is a value/expression key
    whose string looks like LaTeX. Labels come from the JSON path; identical formulas
    (e.g. the four u=0 boundary faces) are de-duped.
    """
    out: List[tuple] = []

    def label_from_path(p: str) -> str:
        p = re.sub(r"\[(\d+)\]", r" \1", p)
        return p.replace("_", " ").replace("/", " › ").replace("latex", "").strip()

    def walk(o: Any, path: str) -> None:
        if isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}/{k}" if path else k)
        elif isinstance(o, str):
            key = path.split("/")[-1].split("[")[0]
            if key.lower().endswith("latex") or (key in _MATH_VALUE_KEYS and "\\" in o):
                out.append((label_from_path(path) or key, o))

    walk(obj, "")
    seen: set = set()
    uniq: List[tuple] = []
    for label, tex in out:
        if tex in seen:
            continue
        seen.add(tex)
        uniq.append((label, tex))
    return uniq


def _render_json_math_md(data: Any, title: str) -> Optional[str]:
    """Render a spec JSON's math fields as Markdown ($$…$$ blocks). None if no math."""
    fields = _json_math_fields(data)
    if not fields:
        return None
    lines = [f"# {title}", ""]
    desc = data.get("description") if isinstance(data, dict) else None
    if isinstance(desc, str) and "\\" not in desc:
        lines += [desc, ""]
    for label, tex in fields:
        lines += [f"**{label}**", "", "$$", tex, "$$", ""]
    return "\n".join(lines) + "\n"


def _render_spec_companions(study_dir: Path) -> None:
    """Write a companion <name>.md for each math-bearing spec JSON in the study dir.

    Renders the LaTeX (governing equations, BCs, coefficients, QoI, ...) as Markdown
    so the equations are readable in a Markdown-math viewer without opening the JSON.
    Skips our own transcript JSONs and any JSON with no math (e.g. results-manifest).
    Best-effort: never fails the run.
    """
    try:
        for jf in sorted(study_dir.glob("*.json")):
            if jf.name.startswith("transcript"):
                continue
            try:
                data = json.loads(jf.read_text())
            except (json.JSONDecodeError, ValueError, OSError):
                continue
            title = None
            if isinstance(data, dict):
                title = data.get("problem_name") or data.get("name") or data.get("title")
            md = _render_json_math_md(data, title or jf.stem.replace("-", " "))
            if md:
                jf.with_suffix(".md").write_text(md)
    except OSError as e:  # noqa: BLE001 - diagnostics must not fail the request
        print(f"@@@ pde-sim purple: could not render spec companions in {study_dir}: {e}", flush=True)


async def _run_claude(prompt: str, study_dir: Path, cwd: Path) -> Tuple[int, str, str, Optional[Dict[str, Any]]]:
    """Drive the Claude CLI for the active config, streaming the transcript live.

    Returns (returncode, final_text, stderr, usage_info). Runs with
    ``--output-format stream-json --verbose`` so each transcript event arrives as
    its own line: they are appended to ``<study_dir>/transcript.stream.jsonl`` as
    they happen (tail it to watch the run grow), and the readable per-role
    transcripts are re-rendered periodically and once more at the end.
    """
    argv: List[str] = [SETTINGS.claude_bin, "-p", prompt, "--output-format", "stream-json", "--verbose"]
    argv += shlex.split(SETTINGS.permission_flag)
    if SETTINGS.claude_model:
        argv += ["--model", SETTINGS.claude_model]
    if SETTINGS.disallowed_tools.strip():
        argv += ["--disallowed-tools", SETTINGS.disallowed_tools.strip()]
    if SETTINGS.extra_args:
        argv += shlex.split(SETTINGS.extra_args)

    save = _env("PDESIM_SAVE_TRANSCRIPT", "1") != "0"
    print(f"@@@ pde-sim purple: launching config {SETTINGS.config} "
          f"(cwd={cwd}, study={study_dir}, disallowed={SETTINGS.disallowed_tools!r})", flush=True)
    if save:
        study_dir.mkdir(parents=True, exist_ok=True)
    # The CLI aborts a Task subagent after CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS with no
    # streamed progress (CLI default 600000 = 10 min). A large code-generation step that
    # composes one big file can go silent longer than that and be reaped mid-run (the
    # orchestrator then sees a spurious "[Request interrupted...]"). Raise the window to
    # 20 min here; setdefault yields to an explicit value the caller exported.
    child_env = os.environ.copy()
    child_env.setdefault("CLAUDE_ASYNC_AGENT_STALL_TIMEOUT_MS", "1200000")
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_env,
    )

    events: List[Any] = []
    stderr_chunks: List[bytes] = []

    async def _drain_stderr() -> None:
        async for chunk in proc.stderr:
            stderr_chunks.append(chunk)

    async def _read_stdout() -> None:
        stream_f = open(study_dir / "transcript.stream.jsonl", "w") if save else None
        last_render = 0.0
        try:
            async for raw in proc.stdout:                        # one JSON event per line
                if stream_f is not None:
                    stream_f.write(raw.decode("utf-8", "replace"))
                    stream_f.flush()                              # so `tail -f` sees it live
                s = raw.strip()
                if not s:
                    continue
                try:
                    ev = json.loads(s)
                except (json.JSONDecodeError, ValueError):
                    continue
                # Stamp wall-clock receive time: the CLI's live task_* progress events
                # carry no timestamp, so this is what lets them render a real [HH:MM:SS]
                # and makes a stall visible as a growing gap between lines.
                if isinstance(ev, dict) and not ev.get("timestamp"):
                    ev["_recv_ts"] = datetime.now().astimezone().isoformat()
                events.append(ev)
                if save and time.monotonic() - last_render > 3.0:
                    _write_transcripts(study_dir, events)         # near-live readable render
                    last_render = time.monotonic()
        finally:
            if stream_f is not None:
                stream_f.close()

    try:
        await asyncio.wait_for(asyncio.gather(_read_stdout(), _drain_stderr()),
                               timeout=SETTINGS.timeout_sec)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        if save:
            _save_transcript(study_dir, events, b"".join(stderr_chunks))  # keep partial transcript
            _render_spec_companions(study_dir)
        raise PdeSimTimeout(
            f"pde-sim pipeline exceeded PDESIM_TIMEOUT={SETTINGS.timeout_sec:.0f}s",
            usage=_partial_usage(events))

    await proc.wait()
    err_b = b"".join(stderr_chunks)
    if save:
        _save_transcript(study_dir, events, err_b)               # final complete render
        _render_spec_companions(study_dir)                        # <name>.md for each spec JSON
    text, usage = _extract_result(events)
    return proc.returncode, text, err_b.decode("utf-8", "replace"), usage


def _find_manifest(stdout: str, study_dir: Path, since_ts: float) -> Optional[Path]:
    """Locate the results-manifest.json produced by this run.

    Prefers the explicit ``PDESIM_DONE <path>`` marker the prompt asks for, then
    falls back to the newest results-manifest.json (mtime >= run start) under the
    requested study dir or the repo artifacts tree, since the orchestrator may
    place artifacts under artifacts/<study-id>/ instead.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("PDESIM_DONE"):
            cand = Path(line.split(None, 1)[1].strip()) if len(line.split(None, 1)) > 1 else None
            if cand and cand.is_file():
                return cand
            break

    search_roots = [study_dir, SETTINGS.artifacts_dir]
    newest: Optional[Path] = None
    newest_mtime = since_ts - 1.0
    for root in search_roots:
        if not root.exists():
            continue
        for m in root.rglob("results-manifest.json"):
            try:
                mt = m.stat().st_mtime
            except OSError:
                continue
            if mt >= since_ts and mt > newest_mtime:
                newest, newest_mtime = m, mt
    return newest


def _pick_run(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Choose a representative run: first successful, else first listed."""
    runs = manifest.get("runs") or []
    for r in runs:
        if r.get("status") == "success" and r.get("command"):
            return r
    for r in runs:
        if r.get("command"):
            return r
    return None


def _split_command(command: str) -> Tuple[Optional[str], str]:
    """From a run command, return (exe_basename, cli_args_string).

    e.g. 'mpiexec -n 4 ./poisson -pc_type gamg' -> ('poisson', '-pc_type gamg').

    The command must be the direct program invocation the bench will run, not a
    build-system target: a 'make run' command is rejected with ValueError so the
    caller reports the failure instead of mis-parsing 'run' as CLI arguments.
    """
    toks = shlex.split(command)
    exe_idx = next((i for i, t in enumerate(toks) if t.startswith("./")), None)
    if exe_idx is None:
        # No './exe' token: skip the launcher and any options it consumes, so the
        # value of a flag like '-n 4' is never mistaken for the executable.
        launchers = {"mpiexec", "mpirun", "srun", "petscmpiexec"}
        # Launcher flags that take a separate following value token.
        value_flags = {
            "-n", "-np", "--n", "--np", "-c", "-N", "--ntasks", "-ppn",
            "-host", "--host", "-hosts", "-hostfile", "--hostfile",
            "-machinefile", "-f", "-rf", "-wdir", "--wdir", "-path",
            "-bind-to", "-map-by", "-rmk", "-launcher", "-x", "-genv",
        }
        i = 1 if toks and os.path.basename(toks[0]) in launchers else 0
        while i < len(toks):
            t = toks[i]
            if t.startswith("-"):
                i += 2 if t in value_flags else 1
            else:
                break
        exe_idx = i if i < len(toks) else None
    if exe_idx is None:
        return None, ""
    exe = os.path.basename(toks[exe_idx])
    if exe.startswith("./"):
        exe = exe[2:]
    if exe in BUILD_TOOLS:
        raise ValueError(
            f"runs[].command invokes a build system ({exe!r}); it must be the direct "
            f"program invocation the bench runs (e.g. 'mpiexec -n 4 ./app -opt val'), "
            f"not a makefile target. Command was: {command!r}"
        )
    cli_args = " ".join(toks[exe_idx + 1:])
    return exe, cli_args


def _collect_sources(manifest_dir: Path, exe: Optional[str]) -> List[Tuple[str, bytes]]:
    """Gather the main source (+ recognized dependency sources) from the study dir."""
    sources: List[Tuple[str, bytes]] = []

    main_path: Optional[Path] = None
    if exe:
        for suf in MAIN_SUFFIXES:
            cand = manifest_dir / f"{exe}{suf}"
            if cand.is_file():
                main_path = cand
                break
    if main_path is None:
        # Fall back across every recognized main-source suffix, not just .c, so a
        # C++/Kokkos study (e.g. main.cxx) whose exe name does not match the source
        # stem is still found.
        candidates = sorted(p for suf in MAIN_SUFFIXES for p in manifest_dir.glob(f"*{suf}"))
        if len(candidates) == 1:
            main_path = candidates[0]
        elif candidates:
            # Ambiguous: prefer one whose stem appears in the exe name if known.
            main_path = next((p for p in candidates if exe and exe in p.stem), candidates[0])
    if main_path is None:
        return sources

    sources.append((main_path.name, main_path.read_bytes()))

    for p in sorted(manifest_dir.iterdir()):
        if not p.is_file() or p == main_path:
            continue
        name = p.name
        is_dep = name.endswith(DEP_COMPOUND_SUFFIXES) or p.suffix in DEP_SUFFIXES
        if is_dep:
            sources.append((name, p.read_bytes()))
    return sources


def _purple_model_label(model_name: Optional[str] = None) -> str:
    """This binding's self-reported model name: pdesim-<model>-c<config>.

    Prefers the ACTUAL model that ran (from the transcript), then the configured
    PDESIM_CLAUDE_MODEL, then "clidefault" only if neither is known. The model is
    slugged (last path component, alphanumerics only). Encoding the config keeps the
    three ablation rungs in distinct output files even though the model is held
    constant across them. Green reads this from the telemetry data Part to name the
    output file (see src/util/telemetry.py PURPLE_TELEMETRY_MODEL_FIELD).
    """
    m = model_name or SETTINGS.claude_model
    tag = "".join(ch for ch in m.lower().split("/")[-1] if ch.isalnum()) if m else "clidefault"
    return f"pdesim-{tag}-c{SETTINGS.config}"


def _build_telemetry(usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Map captured usage onto the bench's petscagent.telemetry.v1 data-Part contract.

    The bench aggregates these fields (all optional) from a data Part tagged with
    PURPLE_TELEMETRY_SCHEMA; see src/util/telemetry.py. Token counts are the
    top-level agent's (subagents excluded), while cost_usd is the whole-run figure.
    Also self-reports this agent's model name for output naming.
    """
    if not usage:
        # Still report identity even when usage capture failed, so the run is named.
        return {"schema_version": PURPLE_TELEMETRY_SCHEMA, "model": _purple_model_label()}
    t: Dict[str, Any] = {"schema_version": PURPLE_TELEMETRY_SCHEMA,
                         "model": _purple_model_label(usage.get("model_name"))}
    t["input_tokens"] = usage["prompt_tokens"]         # fresh + cached input
    t["output_tokens"] = usage["completion_tokens"]
    t["total_tokens"] = usage["total_tokens"]
    t["cached_tokens"] = usage["cached_tokens"]
    t["peak_context_tokens"] = usage.get("peak_context_tokens") or usage["prompt_tokens"]
    if usage.get("num_turns") is not None:
        t["model_calls"] = usage["num_turns"]
    if usage.get("tool_calls") is not None:
        t["tool_calls"] = usage["tool_calls"]
    if usage.get("total_cost_usd") is not None:
        t["cost_usd"] = usage["total_cost_usd"]
    return t


def _append_usage_log(usage: Optional[Dict[str, Any]], context_id: str, since: float, rc: int) -> None:
    """Append one JSON record of this request's usage to the per-config sidecar.

    Captures total_cost_usd (the trustworthy subagent-inclusive figure for config 3)
    alongside the top-level token counts. When PDESIM_USAGE_DIR is set, the file is
    named from this agent's self-reported label, matching the Green output file:
    ``<dir>/usage-pdesim-<model>-c<config>.jsonl``. PDESIM_USAGE_LOG still forces an
    explicit path. No-op when neither is set; failures are logged, not raised.
    """
    usage_dir = os.environ.get("PDESIM_USAGE_DIR")
    if usage_dir:
        label = _purple_model_label(usage.get("model_name") if usage else None)
        path = str(Path(usage_dir) / f"usage-{label}.jsonl")
    else:
        path = os.environ.get("PDESIM_USAGE_LOG")
    if not path:
        return
    rec: Dict[str, Any] = {
        "ts": time.time(), "config": SETTINGS.config, "context_id": context_id,
        "returncode": rc, "wall_sec": round(time.time() - since, 1),
    }
    if usage:
        rec.update(usage)
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError as e:  # noqa: BLE001 - a usage-log failure must not fail the request
        print(f"@@@ pde-sim purple: could not write usage log {path}: {e}", flush=True)


class PdeSimPurpleExecutor(AgentExecutor):
    """A2A executor that maps a problem description to a pde-sim study."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        problem_description = context.get_user_input()
        ctx = (context.context_id or "study").replace("/", "_")
        study_dir = SETTINGS.artifacts_dir / f"bench-{ctx}-{int(time.time())}"
        study_dir.mkdir(parents=True, exist_ok=True)

        try:
            since = time.time()
            usage: Optional[Dict[str, Any]] = None
            rc = -1
            prompt = _prompt_for(problem_description, study_dir)
            # All configs run at the repo root so they share identical project
            # context (CLAUDE.md/AGENTS.md); the ONLY difference across the ladder
            # is the tool gating (PDESIM_DISALLOWED_TOOLS): config 1 blocks the
            # subagent (Task/Agent) and Skill tools, config 2 blocks only the
            # subagent tool, config 3 blocks nothing. Running config 1 elsewhere
            # would also strip project context, confounding the 1->2 comparison.
            try:
                rc, stdout, stderr, usage = await _run_claude(prompt, study_dir, SETTINGS.repo_dir)
                print(f"@@@ pde-sim purple: pipeline exit code {rc}", flush=True)
                if usage:
                    print(f"@@@ pde-sim purple: usage prompt={usage['prompt_tokens']} "
                          f"completion={usage['completion_tokens']} cached={usage['cached_tokens']} "
                          f"cost_usd={usage['total_cost_usd']} turns={usage['num_turns']} "
                          f"subagents={usage['subagents_spawned']} "
                          f"skill_tool={usage['skill_tool_calls']} "
                          f"skills_read={usage['pdesim_skills_read']}", flush=True)
                # Locate and parse the manifest once; fold code-generation's build/run/fix
                # iteration count (its attempts[] journal length) into usage so the sidecar
                # records a per-stage attempt count for the code-generation stage too.
                manifest_path = _find_manifest(stdout, study_dir, since)
                manifest = json.loads(manifest_path.read_text()) if manifest_path else None
                if usage is not None and isinstance(manifest, dict):
                    usage["code_generation_iterations"] = len(manifest.get("attempts") or [])
            except PdeSimTimeout as e:
                # A killed run reports no cost, but it still consumed tokens; keep
                # what it did report so the spend is not silently lost.
                usage = e.usage
                raise
            finally:
                # Record usage on EVERY path — success, timeout, or failure — so a
                # run's consumption always lands in the sidecar.
                _append_usage_log(usage, ctx, since, rc)

            if manifest_path is None:
                raise RuntimeError(
                    "No results-manifest.json was produced by the pde-sim pipeline "
                    f"(exit={rc}). stderr tail: {stderr[-500:]!r}"
                )
            manifest_dir = manifest_path.parent

            run = _pick_run(manifest)
            if run is None:
                raise RuntimeError(f"results-manifest.json at {manifest_path} has no usable run entry")

            exe, cli_args = _split_command(run["command"])
            nsize = int(run.get("mpi_ranks") or 1)
            nsize = max(1, min(nsize, SETTINGS.max_nsize))

            sources = _collect_sources(manifest_dir, exe)
            if not sources:
                raise RuntimeError(f"No generated source files found next to {manifest_path}")

            print(f"@@@ pde-sim purple: nsize={nsize} cli_args={cli_args!r} "
                  f"files={[n for n, _ in sources]}", flush=True)

            parts_list = [new_text_part(f"Code generation successful ✅\nnsize: {nsize}\n"
                                        f"cli_args: {cli_args}\n")]
            telemetry = _build_telemetry(usage)
            if telemetry:
                # Green reads token/cost/effort from this versioned data Part.
                parts_list.append(new_data_part(telemetry))
            for name, data in sources:
                parts_list.append(new_raw_part(data, filename=name, media_type="text/plain"))
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )
        except Exception as e:  # noqa: BLE001 - report every failure over A2A
            print(f"@@@ pde-sim purple: ❌ {type(e).__name__}: {e}", flush=True)
            parts_list = [new_text_part(f"Code generation failed ❌\nerror: {e}\n")]
            await event_queue.enqueue_event(
                new_agent_parts_message(parts_list, context_id=context.context_id)
            )

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def prepare_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="pdesim_petsc_code_generation",
        name="pde-sim PETSc Code Generation",
        description="Drives the pde-sim multi-agent pipeline to turn a physical-phenomenon "
                    "description into a verified PETSc program, returned as source plus run args.",
        tags=["purple agent", "pde-sim", "PETSc", "multi-agent", "HPC"],
        examples=["Solve the 2-D steady Darcy flow problem on the unit square."],
    )
    return AgentCard(
        name=f"pdesim_purple_c{SETTINGS.config}",
        description="pde-sim binding for petscagent-bench: an A2A Purple agent that generates "
                    f"verified PETSc code (ablation config {SETTINGS.config}).",
        supported_interfaces=[AgentInterface(
            url=url, protocol_binding="JSONRPC", protocol_version="1.0"
        )],
        version="0.1.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain", "application/octet-stream"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


def start(host: str = "localhost", port: int = 9002, card_url: Optional[str] = None) -> None:
    card = prepare_agent_card(card_url or f"http://{host}:{port}")
    request_handler = DefaultRequestHandler(
        agent_executor=PdeSimPurpleExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    app = Starlette(routes=[
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(request_handler, rpc_url="/"),
    ])
    print(f"@@@ pde-sim purple: serving on http://{host}:{port} "
          f"(config={SETTINGS.config}, repo={SETTINGS.repo_dir}, timeout={SETTINGS.timeout_sec:.0f}s)", flush=True)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pde-sim Purple-agent binding for petscagent-bench.")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind")
    parser.add_argument("--card-url", type=str, help="External URL to advertise in the agent card")
    args = parser.parse_args()
    start(host=args.host, port=args.port, card_url=args.card_url)
