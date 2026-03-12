"""Self-modifying toy AI agent."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from difflib import unified_diff
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langchain.tools import tool
from langchain_openai import ChatOpenAI

AGENT_FILE = Path(__file__).resolve()
CONTEXT_FILE = AGENT_FILE.parent / ".agent_context.json"
MARKER_FILE = AGENT_FILE.parent / ".agentity_workdir"

_current_context: dict = {}


def log(message: str) -> None:
    """Print message with timestamp prefix."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


SYSTEM_PROMPT = """\
You are a self-modifying AI agent. You MUST read, edit, and restart your own source code to achieve your goals.

Core tools (must always exist): read_source, edit_source, restart_agent.

Golden rule (execution):
- All goal-directed actions MUST be performed by calling @tool functions.
- NEVER implement a subtask by pasting ad-hoc executable logic into the agent's main runtime / control flow.
- New logic belongs inside small, atomic @tool functions (and tiny shared helpers). The agent's main loop should orchestrate tools, not contain the solution.
- If you cannot complete the user's goal with existing tools, you MUST add or modify @tool functions as needed (using read_source, edit_source, restart_agent) instead of just listing what tools are needed or replying with inability. For example, if asked "what's the weather like now?", you should add and use the appropriate tool(s) to solve the problem, rather than merely stating a tool is needed or responding with a lack of access.

Workflow:
1) read_source, then write a short numbered plan of subtasks.
   - Each subtask has a clear Definition of Done (DoD).
2) For the current subtask, add missing capabilities ONLY by adding ONE atomic @tool at a time (single responsibility).
3) After EVERY edit_source, you MUST call:
   restart_agent(next_step="...")

None of the tools added to the current session will work until the restart is fully completed. 
Therefore, after requesting a restart and receiving a positive response from restart_agent tool, you should immediately terminate your work, without any additional messages, attempts to invoke tools, etc.

Critical: restart chaining via next_step (mandatory):
- Treat next_step as the authoritative continuation instruction after restart.
- After restart, your FIRST action is to follow next_step exactly (no replanning unless blocked).
- next_step MUST be explicit, actionable, and tool-centric, e.g.:
  "SUBTASK 2/5: Call tool fetch_url(url=...). Then call parse_html(...). If DoD met, restart_agent(next_step='SUBTASK 3/5: ...')."
- If you need to modify code again, do it, then restart_agent with an updated next_step.

Autonomy constraints:
- Zero manual user steps. Do not require environment variables, API keys, interactive auth, installs, or external setup.
- Use only Python standard library capabilities available in the environment.
- No mocks/stubs/placeholder outputs. Produce real results or fail with concrete diagnostics (but do not delegate fixes to the user).

Completion:
- You are done ONLY when you have executed the entire pipeline yourself via tool calls and produced the final real outputs.
- Final response must contain the results (not "how to run" instructions). Do NOT restart after the final answer.

Invariants:
- Never remove/alter the system prompt or the user's goal.
- Never remove the three core tools.
- Prefer adding tools over editing existing ones, but edit orchestration code when needed to call tools.\
"""

EXCEPTION_RECOVERY_TEMPLATE = """\
EXCEPTION OCCURRED DURING AGENT INVOCATION:

Exception Type: {exc_type}
Exception Message: {exc_message}

Full Traceback:
{exc_traceback}

YOUR TASK: Analyze this exception and resolve it. You must:
1. Read the source code with read_source tool to understand what caused this exception
2. Identify the root cause of the problem
3. Fix the issue by modifying the source code with edit_source tool appropriately
4. Restart the agent to continue with the original goal with restart_agent

After fixing the exception, continue with the original goal: {goal}
"""


@tool
def read_source() -> str:
    """Read the agent's own source code. Returns the full content of agentity.py."""
    return AGENT_FILE.read_text()


@tool
def edit_source(old_text: str, new_text: str) -> str:
    """Edit the agent's source code by replacing old_text with new_text.

    Creates a backup before the first edit. Pass an empty old_text to append
    new_text before the ``if __name__`` guard (useful for adding new tools).

    Args:
        old_text: Exact substring to find in the current source. Empty string
            means "append before the main guard".
        new_text: Replacement text.
    """
    source = AGENT_FILE.read_text()

    if old_text == "":
        anchor = '\nif __name__ == "__main__":'
        if anchor not in source:
            return "Error: cannot find main guard to insert before."
        source = source.replace(anchor, f"\n\n{new_text}\n{anchor}")
    else:
        if old_text not in source:
            return "Error: old_text not found in source."
        source = source.replace(old_text, new_text, 1)

    AGENT_FILE.write_text(source)
    diff = unified_diff(a=old_text.splitlines(keepends=True), b=new_text.splitlines(keepends=True), lineterm="")
    log("Patch applied:\n" + "".join(diff))
    return f"Source updated ({len(source)} chars)."


def _run_check(cmd: str) -> str | None:
    """Run a validation check command. Returns error message or None."""
    result = subprocess.run(["uv", "run", cmd, "check", str(AGENT_FILE)], capture_output=True, text=True)
    return (
        f"{cmd} check failed:\n{result.stdout}{result.stderr}\nFix errors and retry."
        if result.returncode != 0
        else None
    )


@tool
def restart_agent(next_step: str = "") -> str:
    """Validate the current source with ruff and ty, then trigger an immediate restart.

    Args:
        next_step: Optional instruction for what to do after restart (e.g.,
            "Continue with step 2: implement the calculation logic" or
            "Use the new tool to complete the user's request"). If provided,
            this will replace the goal after restart to guide continuation.

    This function validates the code and immediately restarts the agent process.
    It does not return on success - the process is replaced with a fresh instance.
    Only returns if validation fails.
    """
    log("Restart requested. Validating...")

    issues = ""
    for check in ["ruff", "ty"]:
        if err := _run_check(check):
            issues += err + "\n"

    if issues:
        err_message = "Validation failed."
        log(err_message)
        err_message = f"{err_message} Discovered issues:\n{issues}\nFix them all and retry."
        return err_message

    log("Validation passed. Saving context...")
    global _current_context
    save_context(
        _current_context.get("system_prompt", ""),
        next_step or _current_context.get("current_goal", ""),
        _current_context.get("messages", []),
        _current_context.get("original_goal", ""),
    )

    log("Context saved. Restarting agent...")
    os.execv(sys.executable, [sys.executable, str(AGENT_FILE), "--context"])


def discover_tools() -> list[BaseTool]:
    """Auto-discover all tools decorated with @tool in the current module."""
    tools = [obj for _, obj in inspect.getmembers(sys.modules[__name__]) if isinstance(obj, BaseTool)]
    log(f"Discovered {len(tools)} tools: {', '.join(t.name for t in tools)}")
    return tools


def save_context(system_prompt: str, goal: str, messages: list, original_goal: str | None = None) -> None:
    """Serialize conversation state to JSON.

    Args:
        system_prompt: The system prompt for the agent.
        goal: The current goal/instruction (may be updated after restarts).
        messages: The conversation messages (can be message objects or dicts).
        original_goal: The original user goal (preserved across all restarts).
                      If None, goal is used as the original_goal.
    """
    data = {
        "system_prompt": system_prompt,
        "original_goal": original_goal if original_goal is not None else goal,
        "current_goal": goal,
        "messages": messages,
    }
    CONTEXT_FILE.write_text(json.dumps(data, indent=2))


def load_context() -> tuple[str, str, list, str] | None:
    """Deserialize conversation state from JSON.

    Returns:
        A tuple of (system_prompt, current_goal, messages, original_goal) or None.
    """
    if not CONTEXT_FILE.exists():
        return None
    data = json.loads(CONTEXT_FILE.read_text())
    og = data.get("original_goal") or data.get("current_goal", "")
    cg = data.get("current_goal") or og
    return (data["system_prompt"], cg, data["messages"], og)


def _ensure_workdir() -> None:
    """If not already in a working copy, create a temp dir, copy the agent there, and re-exec."""
    if MARKER_FILE.exists():
        return

    tmp = tempfile.mkdtemp(prefix="agentity-")
    dst = Path(tmp) / AGENT_FILE.name
    shutil.copy2(AGENT_FILE, dst)

    src_dir = AGENT_FILE.parent
    for name in ("pyproject.toml", "uv.lock"):
        src_file = src_dir / name
        if src_file.exists():
            shutil.copy2(src_file, Path(tmp) / name)

    marker = Path(tmp) / MARKER_FILE.name
    marker.write_text(json.dumps({"source_dir": str(AGENT_FILE.parent)}))

    log(f"Working directory: {tmp}")
    os.execv(sys.executable, [sys.executable, str(dst)] + sys.argv[1:])


def main() -> None:
    _ensure_workdir()

    env_path = (
        Path(json.loads(MARKER_FILE.read_text()).get("source_dir", "")) / ".env" if MARKER_FILE.exists() else None
    )
    load_dotenv(env_path)

    parser = argparse.ArgumentParser(description="Agentity: self-modifying AI agent")
    parser.add_argument("goal", nargs="?", default=None, help="The goal for the agent")
    parser.add_argument("--context", action="store_true", help="Resume from saved context")
    args = parser.parse_args()

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        log(
            "Fatal error: OPENAI_API_KEY environment variable is not set.\nPlease set it in your .env file or environment."
        )
        sys.exit(1)

    log(f"Using model: {model} on {base_url}")

    system_prompt = SYSTEM_PROMPT
    goal: str | None = args.goal
    prior_messages: list = []
    original_goal: str | None = None

    if args.context and (ctx := load_context()):
        system_prompt, goal, prior_messages, original_goal = ctx

    goal = goal or input("Enter goal: ")
    log(f"Current goal: {goal}")

    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0.0)  # type: ignore[unknown-argument]

    tools = discover_tools()
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)

    messages = (
        prior_messages + [{"role": "user", "content": goal}]
        if prior_messages and goal != original_goal
        else [{"role": "user", "content": goal}]
    )

    global _current_context
    _current_context = {
        "system_prompt": system_prompt,
        "current_goal": goal,
        "messages": messages,
        "original_goal": original_goal or goal,
    }

    try:
        result = agent.invoke({"messages": messages})

        last = result["messages"][-1]
        content = getattr(last, "content", None) if hasattr(last, "content") else last.get("content", "")
        log(f"\nFinal result:{str(content)}")
    except Exception as e:
        exc_type = type(e).__name__
        exc_message = str(e)
        exc_traceback = "".join(traceback.format_exception(type(e), e, e.__traceback__))

        error_details = EXCEPTION_RECOVERY_TEMPLATE.format(
            exc_type=exc_type,
            exc_message=exc_message,
            exc_traceback=exc_traceback,
            goal=goal or original_goal,
        )

        log("Exception occurred. Restarting agent in auto-healing mode")

        messages.append({"role": "user", "content": error_details})

        _current_context["messages"] = messages
        _current_context["current_goal"] = (
            f"Resolve the exception that occurred and then continue with: {goal or original_goal}"
        )

        save_context(
            system_prompt,
            _current_context["current_goal"],
            messages,
            original_goal or goal,
        )

        log("Context saved with exception details. Restarting agent...")
        os.execv(sys.executable, [sys.executable, str(AGENT_FILE), "--context"])


if __name__ == "__main__":
    main()
