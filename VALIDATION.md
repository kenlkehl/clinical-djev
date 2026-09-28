# Validation record

Date: 2026-09-28.

- Python 3.12.3; API dependencies are pinned in `uv.lock`.
- 33 CPU/API/batch tests passed (0.54 s). The suite was run outside the desktop
  sandbox because the sandbox blocked cross-thread asyncio wakeups.
- Ruff checks and Python byte-compilation passed.
- Wheel and source distribution built; the source archive was checked to
  exclude virtual environments, caches, and `.env`.
- The actual pinned DiffusionGemma tokenizer was used to compile
  `examples/clinical-request.json`: 428 prompt tokens, one 32-token canvas,
  verified one-token slots for noul, choice, and score labels.
- A 512-question mixed-type schema with 26-way choices/scores compiled into
  62 canvases. All used the same prompt object. The schema alone used 70,775
  prompt tokens, so this stress case intentionally used a 262,144-token compiler
  limit; it would be rejected by the default 32,768-token deployment limit.
- 64 concurrent real-tokenizer compilations of synthetic clinical input passed
  with 16 CPU workers, each request around 10,400 tokens.
- Host GPUs: two NVIDIA RTX A6000 cards, each 49,140 MiB, compute capability 8.6,
  driver 595.71.05. GPU 1 was selected without stopping other processes.

The pinned GPU runtime is installed locally in `.venv-gpu`; all 199 dependency
versions are recorded in `src/djev/runtime.lock`. CUDA and the required native
structured-read fields were verified on GPU 1.

A temporary **dummy-weight** instance of the real NVFP4 model architecture was
loaded on that A6000. It used Marlin NVFP4 MoE, BF16 KV, and Triton attention;
weight allocation used 17.07 GiB and the default memory budget left 22.44 GiB for
KV. Real GPU/API protocol checks passed for all question types, two denoising
steps, 32 questions split across four groups, long input, and four concurrent
requests. The detailed results are in `outputs/dummy-smoke.json`. Dummy-weight
timings/answers do not establish trained-model performance or semantic accuracy.

These checks identified and fixed two runtime issues: `auto` KV selects the
checkpoint's unsupported FP8 cache on Ampere, so BF16 is forced explicitly;
vLLM diffusion rejects AR seed/temperature fields, so request seeding controls
the initial canvas only. The updated CLI uses `--no-enable-log-requests`.

## Trained checkpoint on GPU 1

The complete 18.82 GB checkpoint is installed in
`/home/klkehl/.cache/huggingface`. Both safetensors shards were verified against
their Hugging Face SHA-256 hashes before loading. The original downloaded copy
also remains under `/data1/ken/models/djev/huggingface`; that SSHFS mount was too
slow for practical cold starts, so the user authorized the local home cache.

Loading the trained model from EXT4 took 53.43 seconds inside the model loader
(total process startup also includes runtime initialization). It used 17.07 GiB
for weights and allocated 22.44 GiB for KV. Observed total GPU 1 memory usage
after the tests was 41,948 MiB. The configuration uses one A6000, Marlin NVFP4,
BF16 activations/KV, Triton attention, and eager execution.

`scripts/smoke.py` passed all semantic assertions on synthetic records. Results
and full probability distributions are in `outputs/smoke.json`:

| Check | Observed wall time |
|---|---:|
| Four questions covering all three types; 428 prompt tokens | 0.159 s |
| 32 choice questions across four canvases; 2,344 prompt tokens | 0.880 s |
| Long record; 10,633 prompt tokens | 1.924 s |
| Four concurrent records, alternating opposite findings | 0.398 s total |
| Concurrent requests with different question schemas | 1.170–1.176 s each |

The example returned fever P(yes) = 0.0000723, adenocarcinoma, expected ECOG
index = 1.000034, and smoking status = unknown, matching its explicit evidence.
The alternate concurrent record returned its fever, squamous histology, ECOG 3,
and current-smoking findings correctly.

Additional live checks, recorded in `outputs/additional-smoke.json`, passed:

- Two denoising steps: 0.262 s.
- Three fixed samples: 0.237 s; three adaptive samples: 0.336 s. Sample counts
  and standard-error fields were checked.
- A 31,033-token synthetic record: 7.659 s, with all expected answers correct.
- A record exceeding the 32,768-token deployment limit: HTTP 413, no truncation.
- Live JSONL processing: three completed, zero failed; resume skipped exactly
  those three rows without reprocessing. See `outputs/smoke-batch-results.jsonl`.

These are functional checks on a tiny synthetic set, with possible prefix-cache
reuse and warmed kernels. They do **not** establish clinical accuracy,
calibration, or sustained corpus throughput. No clinical accuracy claim has
been established. The validation server was stopped after testing to release
GPU 1; the other GPU and its processes were not modified.

## 100k context follow-up

The same trained checkpoint was tested with `--max-model-len 131072`,
`DJEV_MAX_CONCURRENT_READS=2`, and the unchanged 0.85 GPU memory budget on GPU 1.
Model loading used 17.07 GiB and the engine allocated 22.44 GiB for KV.

`scripts/long_context_smoke.py` passed all four semantic checks (noul fever,
choice histology, score ECOG, choice smoking) for each synthetic case. Clinical
facts were distributed across the beginning, middle, and end of the input.
Distinct initial state lines prevented reuse of the long document prefix
between cases; the short common question prefix could still be reused.

| Case | Prompt tokens | Wall time |
|---|---:|---:|
| Single request | 100,412 | 50.123 s |
| Concurrent submission B | 100,419 | 53.261 s |
| Concurrent submission A | 100,414 | 104.021 s |

The two concurrent submissions finished in approximately 104.0 seconds total.
The runtime logs showed one running request and one waiting during long
prefill; this validates accepting and completing the pair, not simultaneous
execution of two full-length prefills.

GPU 1 memory sampled every 0.5 seconds peaked at **41,888 MiB (40.906 GiB)**,
leaving about 6.5 GiB free. This sampling can miss brief peaks, but all requests
completed without an out-of-memory error. The model advertises 262,144 tokens;
this run establishes capacity around 100k, not validation of the full maximum.

Full results: `outputs/long-context-smoke.json`; sampled memory:
`outputs/long-context-gpu-memory.json`. The test server was stopped after the
check. These simple synthetic records do not establish real clinical extraction
accuracy at long context lengths.

## 16K scheduling-budget comparison

On 2026-09-28, the scheduling budget was made configurable through
`DJEV_MAX_NUM_BATCHED_TOKENS` and `--max-num-batched-tokens`; its default was
changed from 4,096 to the requested 16,384. All 33 existing tests and Ruff checks
passed after the change.

The trained checkpoint was tested first with 16,384, then with a fresh 4,096
control. Each used a newly started server on GPU 1, the same pinned runtime and
checkpoint, `--max-model-len 131072`, `DJEV_MAX_CONCURRENT_READS=2`, and the 0.85
GPU memory budget. The existing `scripts/long_context_smoke.py --tokens 100000`
workload and seed 42 were unchanged. Each run submitted one record followed by
two independent records together; distinct initial state lines prevented reuse
of the long document prefix. Startup time is excluded from request timings.

| Measurement | 4,096 budget | 16,384 budget |
|---|---:|---:|
| Single record, 100,412 prompt tokens | 50.566 s | 54.715 s |
| First concurrent completion, 100,414 tokens | 53.073 s | 66.363 s |
| Second concurrent completion, 100,419 tokens | 103.083 s | 112.800 s |
| Concurrent pair total wall time | 103.092 s | 112.809 s |
| Peak sampled GPU memory | 41,888 MiB | 42,174 MiB |
| Engine-reported available KV cache memory | 22.44 GiB | 20.50 GiB |

All four answer checks passed for every record with both budgets, and neither
run encountered an out-of-memory error. GPU memory was sampled at roughly
0.5-second intervals through startup and inference; brief peaks can be missed.
Both logs showed one running and one waiting request during long prefills.
The 16K budget increased measured latency by 8.2% for the single record and
9.4% for the pair. This single-pass comparison found no throughput improvement
for these 100k-token inputs; it does not establish performance on shorter
records, larger question sets, or a sustained production workload.

The requested 16K budget is the new default; use
`--max-num-batched-tokens 4096` or `DJEV_MAX_NUM_BATCHED_TOKENS=4096` to restore
the faster budget for this measured workload. Both test servers were stopped
afterward, releasing GPU 1. GPU 0 and its existing workload were not modified.

Raw results, launch arguments, runtime logs, and memory samples are in
`outputs/batch-budget-16k-run/` and `outputs/batch-budget-4k-control/`; the compact
comparison is `outputs/batch-budget-comparison.json`.
