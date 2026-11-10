"""Generate the 20 Google Colab evaluation notebooks (4 models x 5 benchmarks).

Same models, wrappers, VLMEvalKit fork and settings as the molab notebooks
(make_notebooks.py); only the notebook shell differs: Colab forms instead of
marimo UI, Colab Secrets for tokens, Google Drive (or a HF dataset repo) for
checkpoints.
"""
import json
import pprint
from pathlib import Path

import make_notebooks as mk

OUT = mk.HERE / "out" / "colab_eval_notebooks"
REPO_PATH = "mehmetmertdalkilic/molab-umm-eval"

# Minimum GPU per model family: BAGEL-architecture checkpoints are ~29 GB in bf16;
# DeltaV runs generation under bf16 autocast, so it needs compute capability >= 8.0.
GPU_REQ = {
    "bagel": dict(min_gpu_gib=38, default_gpu="A100", gpus="A100 (Pro), or G4 / H100 (Pro+)"),
    "bagel_zebracot": dict(min_gpu_gib=38, default_gpu="A100", gpus="A100 (Pro), or G4 / H100 (Pro+)"),
    "thinkmorph": dict(min_gpu_gib=38, default_gpu="A100", gpus="A100 (Pro), or G4 / H100 (Pro+)"),
    "deltav": dict(min_gpu_gib=20, default_gpu="L4", gpus="L4 or A100 (Pro), or G4 / H100 (Pro+)"),
}


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code_cell(text, form=False):
    meta = {"cellView": "form"} if form else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta,
            "outputs": [], "source": text.splitlines(True)}


def header_md(run, req, colab_url):
    refs = "\n".join(f"| {r['source']} | {r['value']} |" for r in run["references"]) or "| (none published for this pair) | |"
    return f"""# {run['model_label']} × {run['bench_label']}

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})

{run['about']}

**Benchmark:** `{run['dataset']}` in the ThinkMorph VLMEvalKit fork, {run['bench_size']}.
**Weights:** [`{run['hf_repo']}`](https://huggingface.co/{run['hf_repo']}) ·
**Eval code:** [`{run['fork']['repo']}`](https://github.com/{run['fork']['repo']}) @ `{run['fork']['sha'][:8]}`

### How to run

1. **GPU:** Runtime → Change runtime type → **{req['gpus']}**. At least {req['min_gpu_gib']} GB of GPU memory and bf16 support are needed, so the free T4 will not work.
2. **Secrets** (key icon in the left sidebar, then give this notebook access):
   `HF_TOKEN` is optional (faster downloads; required only for Hugging Face checkpoints).
   `OPENAI_API_KEY` is optional; it switches answer matching to the GPT-5 judge the papers used.
3. **Settings** below: checkpoints go to Google Drive by default, so a run that is cut off
   resumes where it stopped. Use the smoke test (e.g. 20 samples) first.
4. **Runtime → Run all**, approve the Google Drive prompt, and keep the tab open
   (Pro+ background execution keeps running with the tab closed, up to 24 hours).

If the runtime disconnects, reconnect and **Run all** again: it reinstalls, restores the
checkpoint and continues.

| Published reference | Accuracy (%) |
|---|---|
{refs}

Published numbers come from different pipelines and judges; treat them as a sanity range, not a target.
"""


SETTINGS_CELL = '''#@title 1. Settings
#@markdown Where checkpoints (predictions, partial progress, scores, traces) are saved so an interrupted run can resume.
CHECKPOINTS = "Google Drive"  #@param ["Google Drive", "Hugging Face repo", "none"]
DRIVE_FOLDER = "umm_eval_results"  #@param {type:"string"}
HF_RESULTS_REPO = ""  #@param {type:"string"}
#@markdown `auto` uses GPT-5 when `OPENAI_API_KEY` is in Colab Secrets, otherwise exact matching.
JUDGE = "auto"  #@param ["auto", "exact_matching", "gpt-5", "chatgpt-0125"]
#@markdown `eval` re-scores saved predictions without running the model.
MODE = "all"  #@param ["all", "eval"]
#@markdown Smoke test: first N samples only (0 = full benchmark).
SMOKE_TEST_N = 0  #@param {type:"integer"}
SYNC_IMAGES = False  #@param {type:"boolean"}

SETTINGS = dict(CHECKPOINTS=CHECKPOINTS, DRIVE_FOLDER=DRIVE_FOLDER, HF_RESULTS_REPO=HF_RESULTS_REPO,
                JUDGE=JUDGE, MODE=MODE, SMOKE_TEST_N=SMOKE_TEST_N, SYNC_IMAGES=SYNC_IMAGES)
'''

COLAB_EXTRA_HELPERS = '''

RESULT_EXT = (".pkl", ".xlsx", ".csv", ".json", ".jsonl", ".tsv", ".log", ".txt")
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def secret(name):
    """Colab Secrets first, then environment variables."""
    try:
        from google.colab import userdata
        value = userdata.get(name)
        if value:
            return value.strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def gpu_query():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        name, mem, cc = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "mem_gib": float(mem) / 1024, "cc": float(cc)}
    except Exception:
        return None


def copy_tree(src, dst, exts=None, only_missing=False):
    """Copy regular files from src to dst (skips symlinks); returns the number copied."""
    src, dst = Path(src), Path(dst)
    n = 0
    if not src.exists():
        return 0
    for p in src.rglob("*"):
        if p.is_symlink() or not p.is_file():
            continue
        if exts and not p.name.lower().endswith(exts):
            continue
        q = dst / p.relative_to(src)
        if q.exists():
            if only_missing:
                continue
            if q.stat().st_size == p.stat().st_size and int(q.stat().st_mtime) >= int(p.stat().st_mtime):
                continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
    return n
'''

RUN_FN = '''

def run_eval(RUN, S):
    t0 = time.time()
    hf_token = secret("HF_TOKEN")
    openai_key = secret("OPENAI_API_KEY")
    mode = S["MODE"]
    smoke_n = int(S["SMOKE_TEST_N"] or 0)
    judge = S["JUDGE"]
    if judge == "auto":
        judge = "gpt-5" if openai_key else "exact_matching"
    if judge != "exact_matching" and not openai_key:
        raise RuntimeError(f"Judge {judge} needs OPENAI_API_KEY in Colab Secrets.")
    ckpt = S["CHECKPOINTS"]
    hf_repo = S["HF_RESULTS_REPO"].strip() if ckpt == "Hugging Face repo" else ""
    if ckpt == "Hugging Face repo" and not (hf_repo and hf_token):
        raise RuntimeError("Hugging Face checkpoints need HF_RESULTS_REPO and an HF_TOKEN secret with write access.")

    gpu = gpu_query()
    if mode == "all":
        if not gpu:
            raise RuntimeError("No GPU. Runtime > Change runtime type > pick " + RUN["colab_gpus"] + ".")
        if gpu["mem_gib"] < RUN["min_gpu_gib"] or gpu["cc"] < 8.0:
            raise RuntimeError(
                f"{gpu['name']} ({gpu['mem_gib']:.0f} GiB, compute capability {gpu['cc']}) cannot run "
                f"{RUN['model_label']}: it needs at least {RUN['min_gpu_gib']} GiB and bf16 support. "
                "Runtime > Change runtime type > " + RUN["colab_gpus"] + ".")
        log(f"GPU: {gpu['name']}, {gpu['mem_gib']:.0f} GiB")

    root = Path("/content/umm_eval") if Path("/content").is_dir() else pick_root()[0]
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    run_name = RUN["model_name"] + (f"-smoke{smoke_n}" if smoke_n else "")
    sync_prefix = f"{run_name}__{RUN['dataset']}"
    P = {
        "code": root / "code",
        "fork": root / "VLMEvalKit_Thinkmorph",
        "deltav": root / "DeltaV",
        "model": root / "models" / RUN["model_key"],
        "lmu": root / "LMUData",
        "shims": root / "shims",
        "work": root / "results" / sync_prefix,
    }
    for k in ("code", "lmu", "work"):
        P[k].mkdir(parents=True, exist_ok=True)
    if mode == "all" and free < RUN["disk_gb"] * 2**30 and not (P["model"] / ".complete").exists():
        raise RuntimeError(f"Only {free / 2**30:.0f} GiB free under {root}; about {RUN['disk_gb']} GiB are needed.")
    log_file = P["work"] / f"run_{sync_prefix}.log"
    log(f"working under {root} ({free / 2**30:.0f} GiB free); judge={judge}; mode={mode}; smoke={smoke_n or 'off'}; checkpoints={ckpt}")

    drive_dir = None
    if ckpt == "Google Drive":
        from google.colab import drive
        drive.mount("/content/drive")
        drive_dir = Path("/content/drive/MyDrive") / S["DRIVE_FOLDER"].strip("/ ") / sync_prefix
        drive_dir.mkdir(parents=True, exist_ok=True)

    # 1. evaluation code (pinned commits) and helper files
    log("Step 1/6: evaluation code")
    fetch_github(RUN["fork"]["repo"], RUN["fork"]["sha"], P["fork"])
    if RUN["deltav"]:
        fetch_github(RUN["deltav"]["repo"], RUN["deltav"]["sha"], P["deltav"])
    for name, text in SRC.items():
        if name != "flash_attn_shim.py":
            (P["code"] / name).write_text(text)

    # 2. Python dependencies (once per runtime)
    log("Step 2/6: Python dependencies")
    marker = root / f".deps_{RUN['deps_key']}"
    if not marker.exists():
        req = P["code"] / "fork_requirements.txt"
        lines = []
        for line in (P["fork"] / "requirements.txt").read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.split("=")[0].split(">")[0].strip() in ("gradio", "google-genai"):
                continue
            lines.append("opencv-python-headless" if s.startswith("opencv-python") else s)
        req.write_text("\\n".join(lines) + "\\n")
        rc, out = py("import importlib.metadata as m; print(m.version('torch'))")
        tv = next((l.strip() for l in out.splitlines() if l.strip()[:1].isdigit()), None)
        cons = None
        if rc == 0 and tv:
            cons = root / "constraints.txt"  # keep Colab's preinstalled CUDA build of torch
            cons.write_text(f"torch=={tv}\\n")
        args = ["-r", str(req)] + RUN["pip"]
        if not (pip_install(args, cons, log_file) or pip_install(args, None, log_file)):
            raise RuntimeError("Dependency installation failed; see the log above.")
        marker.write_text("ok")

    # 3. CUDA / torch check and flash-attn fallback
    if mode == "all":
        log("Step 3/6: CUDA check")
        cuda_test = (
            "import torch, torch.nn.functional as F; import torchvision; "
            "x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16); (x @ x).sum().item(); "
            "q = torch.randn(1, 2, 16, 64, device='cuda', dtype=torch.bfloat16); F.scaled_dot_product_attention(q, q, q); "
            "print('CUDA' + '_OK', torch.__version__, torch.cuda.get_device_name(0))"
        )
        rc, out = py(cuda_test)
        if rc != 0 or "CUDA_OK" not in out:
            log("torch cannot run on this GPU; installing a CUDA 12.8 build")
            print(out[-2000:])
            pip_install(["--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"], None, log_file)
            rc, out = py(cuda_test)
        if rc != 0 or "CUDA_OK" not in out:
            raise RuntimeError("PyTorch cannot use the GPU:\\n" + out[-1500:])
        log(out.strip().splitlines()[-1])

    use_shim = False
    if RUN["flash_attn_check"]:
        if mode == "all":
            fa = (
                "import torch; from flash_attn import flash_attn_varlen_func as f; "
                "q = torch.randn(8, 4, 64, device='cuda', dtype=torch.bfloat16); "
                "cu = torch.tensor([0, 8], device='cuda', dtype=torch.int32); f(q, q, q, cu, cu, 8, 8, causal=True); print('FA' + '_OK')"
            )
        else:
            fa = "from flash_attn import flash_attn_varlen_func; print('FA' + '_OK')"
        rc, out = py(fa)
        use_shim = rc != 0 or "FA_OK" not in out
        if use_shim:
            (P["shims"] / "flash_attn").mkdir(parents=True, exist_ok=True)
            (P["shims"] / "flash_attn" / "__init__.py").write_text(SRC["flash_attn_shim.py"])
            log("flash-attn not usable here; using the PyTorch SDPA fallback (same math, somewhat slower)")
        else:
            log("flash-attn works on this GPU")

    env = dict(os.environ)
    pp = ([str(P["shims"])] if use_shim else []) + [str(P["code"]), str(P["fork"])]
    if RUN["deltav"]:
        pp.append(str(P["deltav"]))
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env.update({
        "PYTHONPATH": os.pathsep.join(pp),
        "LMUData": str(P["lmu"]),
        "HF_HOME": str(root / "hf_home"),
        "TORCH_HOME": str(root / "torch_home"),
        "MOLAB_TRACE_FILE": str(P["work"] / f"traces_{sync_prefix}.jsonl"),
        "SKIP_ERR": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "USE_TF": "0",
        "USE_FLAX": "0",
        "USE_JAX": "0",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    })
    env.pop("MPLBACKEND", None)  # Colab's inline backend is unavailable in subprocesses
    if hf_token:
        env["HF_TOKEN"] = hf_token
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key

    # 4. model weights
    if mode == "all":
        log(f"Step 4/6: downloading {RUN['hf_repo']}")
        done = P["model"] / ".complete"
        P["model"].mkdir(parents=True, exist_ok=True)
        if not done.exists():
            dl = (
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id={RUN['hf_repo']!r}, local_dir={str(P['model'])!r}, "
                f"allow_patterns={RUN['allow_patterns']!r}, max_workers=8); print('DL_OK')"
            )
            if stream_cmd([sys.executable, "-c", dl], env=env, log_file=log_file, progress_every=30) != 0:
                raise RuntimeError("Model download failed; see the log above.")
            done.write_text("ok")

    # 5. spec + restore checkpoints
    spec = {
        "fork_dir": str(P["fork"]),
        "code_dir": str(P["code"]),
        "deltav_repo": str(P["deltav"]) if RUN["deltav"] else None,
        "stub_thinkmorph": RUN["stub_thinkmorph"],
        "model_class": RUN["model_class"],
        "model_name": run_name,
        "model_kwargs": fill(RUN["model_kwargs"], {
            "MODEL_DIR": str(P["model"]),
            "IMG_DIR": str(P["work"] / "visual_thoughts"),
            "DELTAV_REPO": str(P["deltav"]),
        }),
        "dataset": RUN["dataset"],
        "work_dir": str(P["work"]),
        "judge": judge,
        "mode": mode,
        "smoke_test_n": smoke_n,
        "eval_day": "molab",
        "eval_hash": RUN["fork"]["sha"][:8],
        "hf_repo": hf_repo,
        "sync_prefix": sync_prefix,
        "sync_images": bool(S["SYNC_IMAGES"]),
        "flash_attn_shim": str(P["shims"] / "flash_attn" / "__init__.py") if use_shim else None,
    }
    spec_path = P["code"] / f"spec_{sync_prefix}.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    sync_cmd = [sys.executable, str(P["code"] / "molab_sync.py")]
    exts = RESULT_EXT + (IMAGE_EXT if S["SYNC_IMAGES"] else ())

    def push():
        if drive_dir is not None:
            n = copy_tree(P["work"], drive_dir, exts=exts)
            with open(P["work"] / "sync.log", "a") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} copied {n} file(s) to {drive_dir}\\n")
        elif hf_repo:
            with open(P["work"] / "sync.log", "a") as fh:
                subprocess.run(sync_cmd + ["push", str(spec_path)], env=env, stdout=fh, stderr=subprocess.STDOUT)

    if drive_dir is not None:
        log(f"Step 5/6: restored {copy_tree(drive_dir, P['work'], only_missing=True)} file(s) from {drive_dir}")
    elif hf_repo:
        log(f"Step 5/6: restoring checkpoints from {hf_repo}/{sync_prefix}")
        stream_cmd(sync_cmd + ["pull", spec_path], env=env, log_file=log_file)
    else:
        log("Step 5/6: checkpoints disabled")

    # 6. evaluation, with periodic checkpoint copies
    log(f"Step 6/6: VLMEvalKit {run_name} x {RUN['dataset']}")
    stop = threading.Event()
    every = 10 if drive_dir is not None else RUN["sync_every_min"]

    def sync_loop():
        while not stop.wait(every * 60):
            try:
                push()
            except Exception as e:
                with open(P["work"] / "sync.log", "a") as fh:
                    fh.write(f"sync failed: {e}\\n")

    th = threading.Thread(target=sync_loop, daemon=True)
    if drive_dir is not None or hf_repo:
        th.start()
    try:
        exit_code = stream_cmd([sys.executable, "-u", P["code"] / "molab_driver.py", spec_path], env=env, log_file=log_file)
    finally:
        stop.set()
        if drive_dir is not None or hf_repo:
            log("saving checkpoints")
            push()

    score_files = find_scores(P["work"], run_name, RUN["dataset"])
    minutes = (time.time() - t0) / 60
    if score_files:
        log(f"Finished in {minutes:.0f} min. Scores: {score_files[-1]}")
    else:
        log("No score file was produced. Last log lines:\\n" + tail(log_file))
    return {"exit_code": exit_code, "work_dir": P["work"], "run_name": run_name,
            "score_files": score_files, "log_file": log_file, "minutes": minutes}
'''

RUN_CELL = '''#@title 4. Run the evaluation
RESULT = run_eval(RUN, SETTINGS)
'''

RESULTS_CELL = '''#@title 5. Results (accuracy table and download)
import io
import zipfile
import pandas as pd
from IPython.display import display

for x in Path(RESULT["work_dir"]).rglob(f"{RESULT['run_name']}_{RUN['dataset']}.xlsx"):
    if x.is_symlink():
        continue
    pred = pd.read_excel(x)["prediction"].astype(str)
    bad = pred[pred.str.startswith("Failed to obtain answer")]
    if len(bad):
        print(f"WARNING: {len(bad)} of {len(pred)} answers are model errors, not answers, so the score below is NOT valid.")
        print("Running the notebook again retries them automatically. First error:")
        print(bad.iloc[0][:400])
        print()
if not RESULT["score_files"]:
    print("No scores yet. Last log lines:")
    print(tail(RESULT["log_file"]))
else:
    df = pd.read_csv(RESULT["score_files"][-1])
    print(f"{RUN['model_label']} on {RUN['bench_label']} (accuracy, %; overall = 'Overall' column)")
    display(df)
    zpath = Path("/content") / f"{RESULT['run_name']}__{RUN['dataset']}_results.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in Path(RESULT["work_dir"]).rglob("*"):
            if p.is_file() and not p.is_symlink() and "visual_thoughts" not in p.parts:
                z.write(p, p.relative_to(RESULT["work_dir"]))
    print("Predictions, scores and reasoning traces:", zpath)
    try:
        from google.colab import files
        files.download(str(zpath))
    except Exception:
        pass
'''


def make_colab(model_key, bench_key):
    run = mk.build_run(model_key, bench_key)
    req = GPU_REQ[model_key]
    run["min_gpu_gib"] = req["min_gpu_gib"]
    run["colab_gpus"] = req["gpus"]
    name = f"{model_key}__{bench_key}.ipynb"
    colab_url = f"https://colab.research.google.com/github/{REPO_PATH}/blob/main/colab_eval_notebooks/{name}"
    helpers = ("#@title 3. Helper code (no need to edit)\n"
               "import json\nimport os\nimport shutil\nimport subprocess\nimport sys\nimport threading\n"
               "import time\nimport urllib.request\nimport zipfile\nfrom pathlib import Path\n\n"
               + mk.cell_sources() + "\n\n" + mk.CELL_HELPERS + COLAB_EXTRA_HELPERS + RUN_FN)
    run_src = ("#@title 2. Model and benchmark settings (edit model_kwargs to change decoding / generation)\n"
               "RUN = " + pprint.pformat(run, width=100, sort_dicts=False) + "\n")
    cells = [
        md_cell(header_md(run, req, colab_url)),
        code_cell(SETTINGS_CELL, form=True),
        code_cell(run_src, form=True),
        code_cell(helpers, form=True),
        code_cell(RUN_CELL, form=True),
        code_cell(RESULTS_CELL, form=True),
    ]
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": req["default_gpu"], "machine_shape": "hm",
                      "name": name},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    return name, json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.ipynb"):
        f.unlink()
    for model_key in mk.MODELS:
        for bench_key in mk.BENCHES:
            name, text = make_colab(model_key, bench_key)
            (OUT / name).write_text(text)
            print("wrote", name)


if __name__ == "__main__":
    main()
