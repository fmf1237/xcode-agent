"""Offline simulated I/O benchmark. Run: python examples/phase2_benchmark.py"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xcode_agent import AgentFactory, LLMResponse, TaskNode, ToolRegistry


class SimulatedIOClient:
    def chat(self, messages, tools):
        time.sleep(0.15)  # Fixed local I/O-like wait; no network or real LLM.
        return LLMResponse(content=messages[-1]["content"])


factory = AgentFactory(SimulatedIOClient, ToolRegistry())
root = factory.create()
tasks = [TaskNode(task_id=f"io-{i}", description=f"sample {i}") for i in range(4)]
measurements = {}
for label, workers in [("serial", 1), ("parallel", 4)]:
    started = time.perf_counter()
    results = root.dispatch(tasks, max_workers=workers)
    measurements[label] = {
        "max_workers": workers,
        "total_ms": round((time.perf_counter() - started) * 1000, 2),
        "task_ms": {result.task_id: round(result.duration_ms, 2) for result in results},
        "statuses": [result.status for result in results],
    }
measurements["speedup"] = round(measurements["serial"]["total_ms"] /
                                 measurements["parallel"]["total_ms"], 2)
print(json.dumps(measurements, indent=2))
