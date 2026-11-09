# molab evaluation notebooks: 4 unified models × 5 vision-centric benchmarks

Twenty self-contained [marimo](https://marimo.io) notebooks for [molab](https://molab.marimo.io). Each one evaluates one model on one benchmark with VLMEvalKit on a single RTX Pro 6000 (96 GB).

| Model | Weights | Paper |
|---|---|---|
| BAGEL-7B-MoT | `ByteDance-Seed/BAGEL-7B-MoT` | arXiv 2505.14683 |
| Bagel-Zebra-CoT | `multimodal-reasoning-lab/Bagel-Zebra-CoT` | arXiv 2507.16746 |
| ThinkMorph-7B | `ThinkMorph/ThinkMorph-7B` | arXiv 2510.27492 |
| DeltaV-2B | `wpj20000/DeltaV-2B` | arXiv 2607.08434 |

| Benchmark | VLMEvalKit name | Size |
|---|---|---|
| SAT | `SAT_circular` | 150 real-image questions, circular evaluation |
| BLINK | `BLINK` | 1,901 (val) |
| V* Bench | `VStarBench` | 191 |
| MMVP | `MMVP` | 300 |
| CV-Bench 3D | `CV-Bench-3D` | 1,200 |

File names are `<model>__<benchmark>.py`, e.g. `thinkmorph__blink.py`.

## Running a notebook on molab

1. Put this folder in a GitHub repo. On molab, use the new-notebook dropdown, paste the GitHub URL of a notebook, and add it to your workspace.
2. Open it and attach the GPU: notebook specs button in the header → RTX Pro 6000 (Blackwell) → save and restart.
3. Fill in the form at the top:
   - **Hugging Face token.** Read access is enough for downloads.
   - **HF dataset repo for checkpoints** (recommended), e.g. `your-name/molab-eval-results`. It is created as a private repo if missing; the token then needs write access. Predictions, partial checkpoints, scores, traces and logs are uploaded every 15 minutes under `<model>__<dataset>/`. All 20 notebooks can share one repo.
   - **OpenAI key** (optional). With a key, the default judge is GPT-5, as in the ThinkMorph paper; without one, VLMEvalKit's exact matching is used.
   - **Smoke test.** Run 20 samples first. It checks the whole pipeline in a few minutes and the progress bar shows the real seconds per sample, which tells you how long the full run will take.
4. Press **Start / resume evaluation** and keep the tab open while it runs.

If the session ends (12-hour cap, 90-minute idle shutdown, or a reclaimed GPU), open the notebook again, attach the GPU, and press Start. It reinstalls, restores the checkpoint from your HF repo and continues from the last saved sample. You lose at most the work since the last upload (15 minutes, plus up to 10 samples, since VLMEvalKit saves every 10).

When it finishes, the last cell shows the accuracy table (`Overall` column) and a download button for predictions, scores and per-sample reasoning traces (`traces_*.jsonl`). "Scoring only" mode re-scores saved predictions without the GPU, for example after you add an OpenAI key.

## What each notebook does

All four models run inside the ThinkMorph authors' VLMEvalKit fork ([hychaochao/VLMEvalKit_Thinkmorph](https://github.com/hychaochao/VLMEvalKit_Thinkmorph), pinned to `cddb6a38`). It is the code behind the ThinkMorph paper's numbers and it contains `SAT_circular`, which upstream VLMEvalKit lacks. Every model therefore gets the same TSV files, the same default multiple-choice prompt and the same scoring code.

| Model | Inference |
|---|---|
| BAGEL | Text-only reasoning with BAGEL's official think-mode system prompt, greedy, up to 4,096 new tokens. CV-Bench 3D uses no-think mode, as in the ThinkMorph paper's BAGEL baseline. |
| Bagel-Zebra-CoT | The loop from the official `infz_bf16.py`: its system prompt, prompt-then-images order, CFG 4.0/2.0, 50 denoising steps. A visual thought follows every reasoning step until the model writes "Final Answer:". |
| ThinkMorph | The fork's own `ThinkMorph` wrapper in interleaved mode (`think=True, understanding_output=False`): up to 3 visual thoughts, 50 denoising steps, up to 4,096 tokens, greedy (the paper's temperature 0). |
| DeltaV | The official `vlmevalkit/vlm/deltav_evalkit.py` with the settings from the repo's `MERGE.md` (incremental encoding, 145 tokens per visual update, up to 16,384 tokens). Decoding inside that wrapper is always greedy. |

Judgment calls you may want to change (all in the `RUN` cell at the top of each notebook, or globally in `generator/make_notebooks.py`):

- **Greedy decoding everywhere.** ThinkMorph's paper uses temperature 0. The fork's example config and Zebra-CoT's script sample at temperature 0.3. Set `do_sample: True` to match those instead.
- **Answer extraction.** When a model marks its answer (`<answer>…</answer>`, `Final Answer:`, `\boxed{}`), only that span goes to VLMEvalKit, so exact matching works without an LLM judge. The full trace is kept in `traces_*.jsonl`. Set `extract_answer: False` to send the full text instead, as the fork does. That only makes sense with the GPT-5 judge.
- **Zebra-CoT is capped at 4 visual thoughts** per question; the official script has no cap.
- **flash-attn.** BAGEL-family code imports `flash_attn`. Prebuilt wheels rarely match a Blackwell GPU (sm_120) with the newest PyTorch, so when flash-attn is missing or broken the notebook substitutes a PyTorch-SDPA version of `flash_attn_varlen_func`. It follows flash-attn's semantics (GQA, bottom-right causal mask), so results are unchanged, but it is slower.
- **DeltaV multi-image budgets.** For multi-image questions (mostly BLINK), DeltaV needs a token budget per extra image. The notebook turns on the TSIM Router (DINOv2 via `torch.hub`) so the budgets are deterministic; without it the official wrapper samples them at random.
- **Denoising steps.** `num_timesteps: 50` dominates ThinkMorph and Zebra-CoT runtime. BRAID evaluates with 20 steps. Lowering it is the biggest speed lever, but it is a protocol change.
- **Resumable output folder.** VLMEvalKit names its output folder after the date, so a run interrupted on one day would restart from zero the next. The driver pins the folder name so the checkpoint is always found.

## Published reference numbers

Each notebook's header shows the numbers below for its pair. They come from different pipelines, prompts and judges, so treat them as a sanity range.

| | SAT | BLINK | V* | MMVP | CV-Bench 3D |
|---|---|---|---|---|---|
| BAGEL (ThinkMorph paper) | 44.67 | 47.66 | 55.49 | 70.33 | 76.03 (2D+3D) |
| BAGEL (BRAID paper) | 44.67 | 54.76 | 57.59 | 75.67 | 83.41 |
| BAGEL (DeltaV paper) | | | 70.1 | 69.3 | |
| ThinkMorph (ThinkMorph paper) | 52.67 | 60.07 | 67.02 | 80.33 | 80.82 (2D+3D) |
| ThinkMorph (DeltaV paper) | | | 64.4 | 8.6 | |
| Bagel-Zebra-CoT (DeltaV paper) | | | 64.9 | 22.0 | |
| DeltaV-2B (DeltaV paper) | | | 75.9 | 51.3 | |

## How long it will take (rough estimate)

These are my estimates for one RTX Pro 6000, not measurements. Use the smoke test's seconds per sample to get real numbers.

| Model | Seconds per sample (est.) | All 5 benchmarks (est.) |
|---|---|---|
| DeltaV-2B | 10–20 | 10–25 GPU-hours |
| BAGEL (text reasoning) | 15–30 | 15–35 GPU-hours |
| ThinkMorph (up to 3 generated images) | 50–90 | 60–110 GPU-hours |
| Bagel-Zebra-CoT (up to 4 generated images) | 90–240 | 100–250 GPU-hours |

BLINK and CV-Bench 3D are about 80% of the work. The image-generating models' BLINK and CV-Bench 3D runs need several 12-hour sessions each, which is why checkpoint sync matters.

## molab limits

What molab documents (as of November 2026):

- **Per notebook:** 4 CPUs, 32 GB RAM, one optional RTX Pro 6000 Blackwell (96 GB).
- **Session length:** up to 12 hours; notebooks idle for more than 90 minutes are shut down.
- **Price:** free "as long as usage is reasonable"; "reasonable" is not defined.
- **Concurrency:** no documented cap on how many notebooks or GPU notebooks one account can run at the same time.
- **Restrictions page:** notebooks "may be terminated without warning" for, among other things, "using notebooks for non-interactive jobs". Long batch evaluations are close to that line. In a GitHub issue about a long training job, a maintainer pointed to this policy.
- **Reliability:** open GitHub issues report GPU sandboxes reclaimed after 15–45 minutes of active use and containers reset during long jobs.

So nothing documented stops you from opening 20 GPU notebooks at once, but it is the kind of usage that could get sessions killed. Run a few at a time, keep checkpoint sync on, and ask the marimo team on Discord before scaling up.

Sources: [molab docs](https://docs.marimo.io/guides/molab/), [molab restrictions](https://molab.marimo.io/pages/molab/restrictions), [molab GPU announcement](https://marimo.io/blog/reintroducing-molab), [issue #11058](https://github.com/marimo-team/marimo/issues/11058), [issue #10428](https://github.com/marimo-team/marimo/issues/10428).

## Regenerating the notebooks

`generator/make_notebooks.py` builds all 20 notebooks from `generator/src/` (model wrappers, VLMEvalKit driver, HF sync script, flash-attn fallback). Edit settings there and run it with `marimo` installed:

```bash
pip install marimo
python generator/make_notebooks.py   # writes to generator/out/molab_eval_notebooks/
```

## What has and has not been tested

Tested: `marimo check` passes on all 20 notebooks and each runs headlessly up to the Start button. The download (via the git fallback, since my sandbox blocks codeload.github.com), install and log-streaming helpers ran against GitHub and PyPI. The checkpoint sync script ran against a mocked Hub API. The main cell ran end to end for all four models with the GPU steps mocked. The wrappers' control flow (prompt order, think prompts, the Zebra-CoT loop and its CFG contexts, answer extraction) ran against a mocked model. The SDPA fallback matches a reference attention implementation to 2e-7. Dependency resolution for both pinned stacks succeeds, and every `transformers` symbol the fork and DeltaV import at load time exists in the pinned versions.

Not tested: a real GPU run. No GPU was available, and the fork's own code was not executed locally. The first smoke test on molab is the real end-to-end check.
