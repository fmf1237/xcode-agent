# Phase 3 offline Mock benchmark

These are local simulated delays, not real model or network measurements.

Configuration: `{"task_counts": [2, 4, 8], "repeats": 3, "max_workers": 4, "mock_llm_delay_ms": 15, "mock_tool_delay_ms": 5, "mock_retry_after_ms": 5, "tool_handlers_synchronized": false}`

| Tasks | Topology | Failure mode | Serial median ms | Parallel workers | Parallel median ms | Speedup | Success rate | LLM attempts/run | Retries/run |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | independent | normal | 76.486 | 2 | 38.695 | 1.977 | 100.00% | 4 | 0 |
| 2 | independent | one_mock_429_per_agent | 118.942 | 2 | 59.725 | 1.991 | 100.00% | 6 | 2 |
| 2 | layered_dag | normal | 74.752 | 2 | 73.992 | 1.010 | 100.00% | 4 | 0 |
| 2 | layered_dag | one_mock_429_per_agent | 117.880 | 2 | 117.520 | 1.003 | 100.00% | 6 | 2 |
| 4 | independent | normal | 150.981 | 4 | 38.696 | 3.902 | 100.00% | 8 | 0 |
| 4 | independent | one_mock_429_per_agent | 235.607 | 4 | 60.080 | 3.922 | 100.00% | 12 | 4 |
| 4 | layered_dag | normal | 151.141 | 4 | 76.048 | 1.987 | 100.00% | 8 | 0 |
| 4 | layered_dag | one_mock_429_per_agent | 236.681 | 4 | 118.955 | 1.990 | 100.00% | 12 | 4 |
| 8 | independent | normal | 300.429 | 4 | 77.821 | 3.861 | 100.00% | 16 | 0 |
| 8 | independent | one_mock_429_per_agent | 468.433 | 4 | 120.800 | 3.878 | 100.00% | 24 | 8 |
| 8 | layered_dag | normal | 302.062 | 4 | 78.504 | 3.848 | 100.00% | 16 | 0 |
| 8 | layered_dag | one_mock_429_per_agent | 469.412 | 4 | 119.523 | 3.927 | 100.00% | 24 | 8 |

Speedup = serial median wall time / parallel median wall time.
Success rate = successful tasks / scheduled tasks over all repeats.
Each task makes two successful Mock LLM calls and one Mock Tool call; retry mode adds one failed Mock LLM attempt.
