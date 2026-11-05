# Data behind the tables and figures

| File | What | Source |
|---|---|---|
| `full-37.csv` | Public index, GPU memory, capability density and the five area scores at 1K / 2K / 4K / 8K context, for seele-1.0-3.3B and its parent Jeffan-4B-2010 at Q8_0 | the public suite of the [Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index), version 0.3: 37 benchmarks, 140,178 requests, scored with the suite's own scorer |
| `per-benchmark-8k.csv` | Every benchmark of the public index at 8K context, for seele-1.0-3.3B and Jeffan-4B-2010 Q8_0 (measured) and Jeffan-4B-2010 BF16 and Jev 1.14 (reported by the index, not run by us). `gold` marks the index's 12 gold benchmarks, which weigh 1.2 inside their area | our runs as for `full-37.csv`; the reported columns from the index's leaderboard, snapshot of 2026-10-22 |
| `memory.csv` | llama-server's CUDA buffers per context, in GiB: weights, KV cache, recurrent state and compute, and their sum `gpu_gib` | llama-server on one NVIDIA A16 GPU, 2026-11-05 |
| `speed.csv` | Decisions per second, latency and GPU memory at 8K context on one NVIDIA A16 GPU, plus each run's score on those 500 requests | 500 requests from ANLI and MMLU-Pro at batch 1, warm. Each model ran twice, back to back (A B A B), on 2026-11-05. The last row is the release file itself, served without `--override-kv` |

Columns and definitions:
- **`public_index`** and the area columns are chance-corrected scores × 100: 0 is guessing, 100 is perfect.
- **`requests_too_long`:** a request longer than the context is counted as wrong, never truncated.
- **`gpu_gib`** is the sum of llama-server's CUDA buffers at that context: weights, KV cache, recurrent state and compute. It does not include the CUDA context.
- **`capability_density`** = −log₂(1 − public_index/100) / gpu_gib.
- **`decisions_per_s`** and the latencies are the model's own time per request, as llama-server reports it. HTTP and tokenization are not included.

How the runs were made:
- **Engine:** llama.cpp b11550 with bidirectional attention, one request at a time, reading the option probabilities as [`decide.py`](../decide.py) does.
- **Hardware:** the full-suite runs were split between NVIDIA A16 and L40 GPUs. On 60 requests with 172 questions, both GPUs chose the same option for every question, and the probabilities were identical.
- **The release file:** served as the model card says, it gave the same answer and the same probabilities as the benchmarked file on all 500 speed-test requests.
