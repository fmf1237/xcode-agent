"""Opt-in OpenAI-compatible smoke test. Run with --live; never run from pytest."""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xcode_agent import Agent, ContextManager, LLMClient, ToolRegistry, TraceRecorder
from xcode_agent.file_tools import register_file_tools


ROOT = Path(__file__).resolve().parents[1]


def trace_summary(trace: TraceRecorder, run_id: str) -> dict:
    events = trace.events(run_id)
    return {"run_id": run_id,
            "events": [event.event_type for event in events],
            "llm_attempts": sum(event.event_type == "llm_start" for event in events),
            "retries": sum(event.event_type == "llm_retry" for event in events),
            "successful_tool_calls": sum(event.event_type == "tool_end" and event.success for event in events)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Explicitly allow real API requests")
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--total-timeout", type=float, default=None)
    args = parser.parse_args()
    if not args.live:
        print(json.dumps({"status": "not_run", "reason": "pass --live to opt in"}))
        return 0

    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL")
    if not key or not model or key == "replace-me":
        print(json.dumps({"status": "not_run", "reason": "OPENAI_API_KEY or OPENAI_MODEL missing"}))
        return 2
    request_timeout = (args.request_timeout if args.request_timeout is not None else
                       float(os.getenv("OPENAI_TIMEOUT", "30")))
    total_timeout = (args.total_timeout if args.total_timeout is not None else
                     max(30.0, 3 * request_timeout))
    if request_timeout <= 0 or total_timeout <= 0:
        print(json.dumps({"status": "invalid_timeout"}))
        return 2

    try:
        llm = LLMClient(key, os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                        model, timeout=request_timeout, retries=0)
        trace = TraceRecorder()

        simple = Agent(llm, ToolRegistry(), ContextManager(), max_rounds=2, trace=trace)
        answer = simple.run("Reply with exactly OK.", total_timeout_s=total_timeout)
        if not answer.strip():
            raise RuntimeError("Empty answer")
        simple_trace = trace_summary(trace, simple.last_run_id)
        if not simple_trace["events"] or simple_trace["events"][0] != "agent_start" or simple_trace["events"][-1] != "agent_end":
            raise RuntimeError("Simple Agent Trace is incomplete")

        tools = ToolRegistry()
        register_file_tools(tools, ROOT / "examples")
        context = ContextManager(system_prompt="For the next question, call read_file before answering.")
        with_tool = Agent(llm, tools, context, max_rounds=4, trace=trace)
        with_tool.run("Read demo.txt with read_file and briefly confirm it was read.",
                      total_timeout_s=total_timeout)
        tool_trace = trace_summary(trace, with_tool.last_run_id)
        if tool_trace["successful_tool_calls"] < 1 or tool_trace["events"][-1] != "agent_end":
            raise RuntimeError("Model did not complete a read_file tool call")

        print(json.dumps({"status": "passed", "request_timeout_s": request_timeout,
                          "total_timeout_s": total_timeout,
                          "simple": simple_trace,
                          "file_tool": tool_trace}, ensure_ascii=False))
        return 0
    except Exception as exc:
        # Never print raw exception messages: providers may include request details.
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__,
                          "error_code": getattr(exc, "error_code", None)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
