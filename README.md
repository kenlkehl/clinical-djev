# clinical-djev

Local Jev-style extraction from long clinical text, using native seeded diffusion
reads in vLLM. Python package: `djev`; project/dependency manager: uv.

It exposes `POST /v1/systemone` with **noul**, **choice**, and **score** questions.
It reads label probabilities directly from the model's diffusion canvas; there
is no generated JSON to repair and no fallback to an unrelated language model.
This uses the NVIDIA DiffusionGemma instruction checkpoint in Jev-style mode,
not separate proprietary Jev weights or the hosted Jev service.

## Start

```bash
uv sync --locked
uv run djev install-runtime
uv run djev download-model
uv run djev doctor
uv run djev serve --gpu 1
```

`serve` starts and supervises a single vLLM worker on `127.0.0.1:8010` and the
extraction API on `127.0.0.1:8011`. It checks the installed runtime, waits for
model readiness, verifies a real structured read, and stops its worker on exit.
The Python API environment stays in this folder's `.venv`, the GPU runtime in
`.venv-gpu`, and model weights in `~/.cache/huggingface` on the local home drive.
The separate environments prevent `uv sync` from removing the nightly runtime.
These paths are defaults for this machine; change them using `.env.example`.
Run these commands from the project folder. Keep the GPU runtime and compilation
caches on a local filesystem: network mounts are slow for their many small files.
`install-runtime --directory /path/to/venv` can create the runtime elsewhere;
then set `DJEV_ENGINE_PYTHON=/path/to/venv/bin/python`. Model downloads default
to HTTP; set `HF_HUB_DISABLE_XET=0` to opt into Hugging Face's Xet transfer path.
Keep the 18.82 GB model cache on a local disk too. The machine's `/data1` SSHFS
mount measured only a few MB/s for cold reads, making startup very slow.
`DJEV_STARTUP_TIMEOUT` defaults to 1,800 seconds and can be increased for slow
storage. Do not remove a model cache while its server is running.

Against an existing compatible vLLM server:

```bash
uv run djev api --upstream http://127.0.0.1:8010
```

Its served-model name, model/tokenizer revision, canvas width, and context limit
must match this API's configuration. `djev engine-command` prints the expected
launch command. A newer backend must preserve the same structured-read protocol.

```bash
curl http://127.0.0.1:8011/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/clinical-request.json
```

Interactive API documentation: `http://127.0.0.1:8011/docs`.
`/health` checks the API process; `/ready` also checks the backend.

## Questions and answers

Each request supplies `state` (clinical text or JSON), `questions` (an ordered
map of IDs to questions), and optional shared `instructions`.

```json
{
  "state": "SYNTHETIC: Fever denied. Biopsy: adenocarcinoma. ECOG 1.",
  "questions": {
    "fever": {
      "type": "noul",
      "instructions": "Is current fever explicitly documented as present?"
    },
    "histology": {
      "type": "choice",
      "instructions": "What histology is documented?",
      "criteria": {"adenocarcinoma": null, "squamous": null, "unknown": "Not documented"}
    },
    "ecog": {
      "type": "score",
      "instructions": "Select the explicitly documented ECOG level.",
      "criteria": ["ECOG 0", "ECOG 1", "ECOG 2", "ECOG 3", "ECOG 4"]
    }
  },
  "samples": 1,
  "seed": 42
}
```

| Type | Criteria | Answer |
|---|---|---|
| `noul` | Optional `{"true": "description", "false": "description"}` | `{"type":"noul","noul":P(yes)}` |
| `choice` | 2–26 option names mapped to descriptions or `null` | `choice`, per-option `probabilities`, and winning `confidence` |
| `score` | 2–26 distinct ordered descriptions | Zero-based expected `score`, indexed `legend` and `probabilities`, winning `confidence` |

For score probabilities `[0.1, 0.2, 0.7]`, the score is
`0*0.1 + 1*0.2 + 2*0.7 = 1.6`, not an integer class and not a one-based score.
For a nonuniform numeric scale, compute its expectation from the returned
probabilities yourself. Choice ties resolve by the supplied option order.

`answers` follows these Jev shapes. The response also includes `diagnostics`
with per-question probabilities, normalized entropy, sample count, agreement,
and standard errors. Standard errors are `null` for one sample. For noul, the
value standard error concerns P(yes); for score, it concerns the expected index;
for choice, it concerns the selected option's probability. These measure
variation across noise draws, not uncertainty due to clinical sampling.
`label_mass` is the mass assigned to a question's labels in the backend's
distribution. With constrained reads (the default), that distribution is
normalized over the group's union of labels, not the full vocabulary; it is
not a measure of evidence quality or an abstention probability.

The values are model probabilities **conditional on the supplied labels**.
They are not calibrated clinical probabilities. Include an explicit `unknown`
or `not_documented` choice where appropriate. Noul cannot distinguish false
from missing evidence unless the question defines how to handle it. Use a
choice question for a score that may be missing; appending an "unknown" ordinal
level would distort the numerical expectation.

## Long documents and sampling

Up to 512 questions are supported per request. Choice names can be multiword;
the compiler maps them to single-token labels and verifies each label in its
actual template position. Groups split when the answer template exceeds the
canvas. Every group receives the **whole clinical document and question list**,
using stable internal question IDs and identical prompt token IDs, so prefix
caching can reuse the prefill. The API does not retain patient text in its
bounded schema cache. The backend's GPU KV cache can retain prefixes until
eviction or process exit.

The 32,768-token default limit includes the prompt, all questions, and canvas
space. The API measures the exact tokenized prompt and rejects overflow with
HTTP 413. It never silently truncates or chunks the clinical document. For
longer records, increase `DJEV_MAX_MODEL_LEN` and check memory on representative
inputs; the model's advertised 262,144-token context is not a guarantee that
this deployment can fit that length at the chosen concurrency.

For inputs of 100k tokens or more, the following **128k deployment profile was
tested on this machine's single A6000**:

```bash
DJEV_MAX_CONCURRENT_READS=2 uv run djev serve --gpu 1 --max-model-len 131072
```

Equivalently, set `DJEV_MAX_MODEL_LEN=131072` and
`DJEV_MAX_CONCURRENT_READS=2` in `.env`. The limit includes the clinical text,
question instructions, and canvas. The default launch remains 32,768 tokens.

Both the original 4,096-token scheduling budget and the current 16,384-token
default were tested with this profile on the same synthetic records:

| Scheduling budget | One 100k-token record | Two submitted together, total | Peak sampled GPU use |
|---|---:|---:|---:|
| 4,096 | 50.6 s | 103.1 s | 40.9 GiB |
| 16,384 | 54.7 s | 112.8 s | 41.2 GiB |

All tested answers were correct, with no out-of-memory errors. The larger
budget was slower in this comparison. Use `--max-num-batched-tokens 4096` to
restore the original budget. vLLM's scheduler still largely serialized the
long prefill stages; accepting concurrent requests does not mean both long
prefills execute simultaneously. These are single-pass synthetic capacity
checks, not a clinical accuracy or sustained-throughput benchmark.

Reproduce the check against that running server with:

```bash
uv run python scripts/long_context_smoke.py --tokens 100000
```

The test places evidence at different positions and gives each record a
distinct initial state line to prevent reuse of its long document prefix.

| Request setting | Default | Meaning |
|---|---|---|
| `samples` | `1` | 1–32 independent canvas-noise draws, or `"auto"` |
| `auto_max` | `4` | Total draws if an automatic read is uncertain |
| `auto_threshold` | `0.25` | Reread a group if any first-read normalized label entropy exceeds this |
| `steps` | `1` | 1–8 denoise steps; fixed template positions remain pinned |
| `seed` | Random | Controls canvas noise; returned in metadata |

Each sample gives a probability distribution. The API averages distributions,
not just hard votes. Automatic sampling adds draws for the whole affected
group. Fixed seeds reproduce input canvases; GPU scheduling/kernels can still
cause numerical variation. Start with one sample and measure accuracy on your
own extraction task before spending throughput on additional draws.

## Large-scale JSONL processing

Each input line is a complete request, with an optional top-level `id` used only
by the batch runner. Make a small synthetic input:

```bash
uv run python -c 'import json; r=json.load(open("examples/clinical-request.json")); print(json.dumps({"id":"synthetic-1",**r}))' > cases.jsonl
uv run djev batch cases.jsonl results.jsonl --concurrency 4
uv run djev batch cases.jsonl results.jsonl --concurrency 4 --resume
```

The runner keeps a bounded window of records in memory, writes results in input
order, flushes after each completed record, retries transient HTTP/transport
errors up to three times, and records per-line errors without stopping the
corpus. It exits nonzero if newly processed records fail. Seeds remain stable
across retries. Inputs are not copied into outputs.

Existing output is never overwritten by default. `--resume` verifies the
completed prefix against input-line SHA-256 hashes and line numbers before
appending. It preserves previous error rows too; retry those separately after
correcting the cause. A changed input or partially written final output line
requires explicit repair and is rejected. Use a new output file when changing
models or configuration. Output IDs and extracted answers can themselves be
sensitive; choose their storage accordingly.

The API bounds active extraction requests (default 16) and concurrent upstream
reads (default 4). HTTP 429 indicates admission overload. vLLM handles continuous
GPU batching; no per-record model reload is needed. Run one API process for these
limits to be global, or divide limits across processes if deploying more than one.

## Configuration

Copy `.env.example` to `.env` or export `DJEV_*` environment variables. Useful
settings include `GPU`, `ENGINE_PYTHON`, `HF_HOME`, `MAX_MODEL_LEN`, `CANVAS_LENGTH`,
`MAX_NUM_BATCHED_TOKENS`, `MAX_CONCURRENT_READS`, `MAX_ACTIVE_REQUESTS`,
`GPU_MEMORY_UTILIZATION`, `TIMEOUT`,
`STARTUP_TIMEOUT`, `MODEL`, `REVISION`, `SERVED_MODEL`, `UPSTREAM`, `HOST`, and `PORT`.

The scheduling budget defaults to 16,384 token positions across all requests
per engine step. Set `DJEV_MAX_NUM_BATCHED_TOKENS` or use
`djev serve --max-num-batched-tokens 16384` to change it; `4096` restores the
original budget. This is separate from the per-request context limit and the
concurrent-read limit. Larger budgets can use more working GPU memory and leave
less memory for the KV cache.

The A6000 launch configuration uses one GPU, BF16 activations/KV, Marlin NVFP4
weight-only kernels, Triton attention, chunked prefill, prefix caching, and eager
execution. NVFP4 weights save memory; A6000 does not have native Blackwell FP4
compute. Eager mode is a conservative starting point for this new runtime.
KV dtype is explicitly `bfloat16`: the checkpoint advertises FP8 KV, so `auto`
selects a cache format that Triton attention cannot use on the A6000.
GPU 1 is the default because it was mostly idle during setup; use `--gpu 0`
to select the other device. No existing GPU process is stopped.
Checkpoint files are read eagerly into host RAM before loading, avoiding random
memory-mapped reads. This requires enough host RAM
for the checkpoint in addition to the serving process; this host has 376 GiB.

The API binds to localhost by default and disables access/request-body logs.
Set `DJEV_API_KEY` for bearer authentication before exposing the API, and use
your usual TLS reverse proxy. `DJEV_UPSTREAM_API_KEY` protects the private vLLM
connection. Invalid requests and backend errors are reported without echoing
clinical text. No clinical text is sent to Hugging Face: only model artifacts
are downloaded there. Model inference targets the configured vLLM endpoint.

## Pinned upstream and verification

The GPU runtime is pinned to:

- vLLM `0.30.1rc1.dev274+g5ba2e3090`, commit
  `5ba2e3090b4c6bb6d5b4fd8ed3045107e787309e`.
- Model `nvidia/diffusiongemma-26B-A4B-it-NVFP4`, revision
  `ec4ff3df205028f4e81c954c2227f9312b3ec2ea`.

The exact installed GPU dependency versions are recorded in
`src/djev/runtime.lock` and used by `install-runtime`.

As checked on 2026-09-28, PyPI vLLM 0.30.0 lacks the structured-read fields.
The [merged upstream PR](https://github.com/vllm-project/vllm/pull/57250) and
[structured diffusion example](https://docs.vllm.ai/en/latest/examples/features/structured_diffusion/)
describe the protocol. The server uses raw token-ID completions with seeded
canvases, read-only emission, exact requested label logprobs, pinned template
positions, and constrained unembedding. Missing label logprobs are errors;
they are never fabricated from a probability floor. See also the
[NVFP4 fallback documentation](https://docs.vllm.ai/en/latest/features/quantization/modelopt/)
and [model card](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4).

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run python scripts/smoke.py  # requires a running real GPU server
```

Unit/integration tests cover all answer types, probability math, schema errors,
canvas pinning, exact logprobs, full-context chunking, adaptive sampling,
concurrency limits, API validation/privacy, version gating, and JSONL resume.
The separate GPU smoke script exercises all question types, long input,
multiple question groups, and concurrent requests on synthetic data. It writes
the full responses and timings to `outputs/smoke.json`; it is a functionality
check, not a clinical accuracy evaluation or a production throughput benchmark.
See [VALIDATION.md](VALIDATION.md) for the completed A6000 checks, observed
synthetic timings, and their limits.
