# Colab evaluation notebooks: 4 unified models × 5 vision-centric benchmarks

The Colab version of the molab notebooks in `../molab_eval_notebooks`. Each notebook evaluates one model on one benchmark with the same model wrappers, VLMEvalKit fork, prompts and settings, so molab and Colab scores are directly comparable. See the molab README for the evaluation protocol, judgment calls and published reference numbers.

## Open in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/deltav__vstar.ipynb) (DeltaV × V*, the cheapest first smoke test)

| Model | SAT | BLINK | V* | MMVP | CV-Bench 3D | CV-Bench 2D |
|---|---|---|---|---|---|---|
| BAGEL | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel__sat.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel__blink.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel__vstar.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel__mmvp.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel__cvbench3d.ipynb) | via the CV-Bench 2D add-on cells |
| Bagel-Zebra-CoT | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel_zebracot__sat.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel_zebracot__blink.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel_zebracot__vstar.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel_zebracot__mmvp.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel_zebracot__cvbench3d.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/bagel_zebracot__cvbench2d.ipynb) |
| ThinkMorph | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/thinkmorph__sat.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/thinkmorph__blink.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/thinkmorph__vstar.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/thinkmorph__mmvp.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/thinkmorph__cvbench3d.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/thinkmorph__cvbench2d.ipynb) |
| DeltaV | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/deltav__sat.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/deltav__blink.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/deltav__vstar.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/deltav__mmvp.ipynb) | [open](https://colab.research.google.com/github/mehmetmertdalkilic/molab-umm-eval/blob/main/colab_eval_notebooks/deltav__cvbench3d.ipynb) | – |

## Which GPU (and Colab plan) you need

| Model | Minimum GPU | Colab plan |
|---|---|---|
| BAGEL, Bagel-Zebra-CoT, ThinkMorph | about 40 GB with bf16: **A100**, or **G4** (RTX PRO 6000) / **H100** | A100: Pro. G4 and H100: Pro+ only |
| DeltaV-2B | bf16 support (compute capability 8.0+): **L4** or better | Pro |

- **The free tier won't work for any of them.** Its T4 has 16 GB and no bf16, and every model here generates in bf16. The notebooks check the GPU first and stop with a message if it is too small.
- **G4 is the same RTX PRO 6000 card molab uses**, so G4 runs match the molab setup exactly.
- **A100 40 GB for the BAGEL family is my estimate, not tested.** The weights take about 29 GB, which leaves roughly 10 GB for image generation. If you hit out-of-memory errors, switch to G4 or H100.

Each notebook asks Colab for its usual GPU when it opens (A100 for the BAGEL family, L4 for DeltaV). You can change it under **Runtime → Change runtime type**.

## Running a notebook

1. **Open it** with a link above. Running straight from GitHub is fine. Use **File → Save a copy in Drive** only if you want to keep edits to the settings code.
2. **Choose the GPU**: Runtime → Change runtime type → pick the GPU from the table → Save.
3. **Add secrets**, both optional: open the key icon in the left sidebar, add the secret and switch on **Notebook access**.
   - `HF_TOKEN`: faster model downloads. It is required only if you store checkpoints on Hugging Face, and then it needs write access.
   - `OPENAI_API_KEY`: switches answer matching to the GPT-5 judge the papers used. Without it, VLMEvalKit's exact matching is used.
4. **Settings** (the first form): checkpoints go to **Google Drive** by default, in `MyDrive/umm_eval_results/<model>__<dataset>/`. Set `SMOKE_TEST_N` to 20 for the first run.
5. **Runtime → Run all.** Approve the Google Drive access prompt. Progress, including the seconds per sample, prints in the "Run the evaluation" cell.
6. When it finishes, the last cell shows the accuracy table (`Overall` column) and downloads a zip of predictions, scores and reasoning traces.

## Resuming

Checkpoints are copied to Drive every 10 minutes and at the end. VLMEvalKit saves its progress every 10 samples. If the runtime disconnects or hits its time limit, reconnect, choose the GPU again and **Run all**. The notebook reinstalls, restores the checkpoint from Drive and continues from the last saved sample.

If you set `SMOKE_TEST_N` back to 0, the full run is stored separately from the smoke test.

## Colab limits worth knowing

- **Paid plans:** Pro costs $9.99/month with 100 compute units; Pro+ costs $49.99/month with 600 units. Google doesn't publish how many units each GPU uses per hour, so check the Resources panel after the smoke test to see the real burn rate.
- **Session length:** Pro+ runs for up to 24 hours, and keeps running with the browser closed. Pro only promises "longer runtimes" than free with no fixed number; there, closing the tab stops the run, so keep it open.
- **GPU availability:** GPUs on all plans depend on availability, and Google says Colab prioritizes interactive use.

At my rough estimate of 200–400 GPU-hours for all 20 notebooks (see the molab README), a full sweep on Colab needs a substantial number of compute units. Running the BAGEL-family BLINK and CV-Bench 3D notebooks on molab's free RTX PRO 6000 and the rest on Colab is a reasonable split.

## Regenerating

`../molab_eval_notebooks/generator/make_colab_notebooks.py` builds these notebooks from the same settings and source files as the molab generator:

```bash
pip install marimo   # make_notebooks.py, which this imports, uses marimo for the molab notebooks
python molab_eval_notebooks/generator/make_colab_notebooks.py
```

## What has and has not been tested

Tested: all 20 notebooks pass notebook-format validation and a lint of their combined code. The main function ran end to end with the GPU, downloads and Colab APIs mocked, in three scenarios: ThinkMorph on an A100 40 GB (flash-attn fallback, Drive checkpoints, results table and zip), DeltaV on an L4, and BAGEL on a T4, which is correctly refused.

Not tested: a real Colab run. The first smoke test is the real end-to-end check.
