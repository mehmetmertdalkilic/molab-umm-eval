"""Generate the 20 molab (marimo) evaluation notebooks: 4 models x 5 benchmarks."""
import copy
import pprint
import sys
import textwrap
from pathlib import Path

from marimo._ast.app_config import _AppConfig
from marimo._ast.cell import CellConfig
from marimo._ast.codegen import generate_filecontents

HERE = Path(__file__).parent
SRC = HERE / "src"
OUT = HERE / "out" / "molab_eval_notebooks"

FORK = {"repo": "hychaochao/VLMEvalKit_Thinkmorph", "sha": "cddb6a38ac5dfaf64020d6aa3964506e26006568"}
DELTAV_REPO = {"repo": "Pengjie-W/DeltaV", "sha": "a104ce7c8f77a2de438dbb04fc393b1ec01c9d26"}

COMMON_PIP = [
    "accelerate>=0.34", "safetensors", "huggingface_hub>=0.34,<1.0", "hf_xet",
    "datasets>=3.0", "regex", "pyyaml", "scipy", "scikit-image", "openpyxl", "jsonlines",
    "rapidfuzz", "einops", "sentencepiece", "timm", "tabulate", "opencv-python-headless",
]

CFG_GEN = {
    "cfg_text_scale": 4.0, "cfg_img_scale": 2.0, "cfg_interval": [0.0, 1.0],
    "timestep_shift": 3.0, "num_timesteps": 50, "cfg_renorm_min": 0.0,
    "cfg_renorm_type": "text_channel",
}

TM_PAPER = "ThinkMorph paper (arXiv 2510.27492, Table 3)"
BRAID_PAPER = "BRAID paper (arXiv 2607.03748, Table 1)"
DV_PAPER = "DeltaV paper (arXiv 2607.08434, Table 5)"

BENCHES = {
    "sat": dict(dataset="SAT_circular", label="SAT (real-image split, circular eval)",
                size="150 questions, scored with VLMEvalKit circular evaluation"),
    "blink": dict(dataset="BLINK", label="BLINK", size="1,901 questions (val split, 14 tasks)"),
    "vstar": dict(dataset="VStarBench", label="V* Bench", size="191 questions"),
    "mmvp": dict(dataset="MMVP", label="MMVP", size="300 questions"),
    "cvbench3d": dict(dataset="CV-Bench-3D", label="CV-Bench 3D", size="1,200 questions (depth + distance)"),
}

REFS = {
    ("bagel", "sat"): [(TM_PAPER, "44.67"), (BRAID_PAPER, "44.67")],
    ("bagel", "blink"): [(TM_PAPER, "47.66"), (BRAID_PAPER, "54.76")],
    ("bagel", "vstar"): [(TM_PAPER, "55.49"), (BRAID_PAPER, "57.59"), (DV_PAPER, "70.1")],
    ("bagel", "mmvp"): [(TM_PAPER, "70.33"), (BRAID_PAPER, "75.67"), (DV_PAPER, "69.3")],
    ("bagel", "cvbench3d"): [(BRAID_PAPER, "83.41"),
                             (TM_PAPER, "76.03 (CV-Bench 2D+3D combined, no-think)")],
    ("thinkmorph", "sat"): [(TM_PAPER, "52.67")],
    ("thinkmorph", "blink"): [(TM_PAPER, "60.07")],
    ("thinkmorph", "vstar"): [(TM_PAPER, "67.02"), (DV_PAPER, "64.4")],
    ("thinkmorph", "mmvp"): [(TM_PAPER, "80.33"), (DV_PAPER, "8.6")],
    ("thinkmorph", "cvbench3d"): [(TM_PAPER, "80.82 (CV-Bench 2D+3D combined)")],
    ("bagel_zebracot", "vstar"): [(DV_PAPER, "64.9")],
    ("bagel_zebracot", "mmvp"): [(DV_PAPER, "22.0")],
    ("deltav", "vstar"): [(DV_PAPER, "75.9")],
    ("deltav", "mmvp"): [(DV_PAPER, "51.3")],
}

MODELS = {
    "bagel": dict(
        label="BAGEL-7B-MoT", model_name="BAGEL-7B-MoT", model_class="BagelThink",
        hf_repo="ByteDance-Seed/BAGEL-7B-MoT",
        allow_patterns=["*.json", "*.txt", "ema.safetensors", "ae.safetensors"],
        disk_gb=45, deps_key="bagel-tf4.49", pip=["transformers==4.49.0"] + COMMON_PIP,
        flash_attn_check=True, stub_thinkmorph=False, deltav=None,
        model_kwargs={"model_path": "{MODEL_DIR}", "ckpt_file": "ema.safetensors", "think": True,
                      "max_new_tokens": 4096, "do_sample": False, "temperature": 0.3,
                      "vit_min_size": 224, "extract_answer": True},
        about=("Base unified model. Text-only reasoning with BAGEL's own think-mode system prompt "
               "(the setting the ThinkMorph paper uses for its BAGEL baseline); CV-Bench 3D runs "
               "in no-think mode, as in that paper. Greedy decoding, up to 4,096 new tokens."),
    ),
    "bagel_zebracot": dict(
        label="Bagel-Zebra-CoT", model_name="Bagel-Zebra-CoT", model_class="ZebraCoT",
        hf_repo="multimodal-reasoning-lab/Bagel-Zebra-CoT",
        allow_patterns=["*.json", "*.txt", "model_bf16.safetensors", "ae.safetensors"],
        disk_gb=45, deps_key="bagel-tf4.49", pip=["transformers==4.49.0"] + COMMON_PIP,
        flash_attn_check=True, stub_thinkmorph=False, deltav=None,
        model_kwargs=dict({"model_path": "{MODEL_DIR}", "ckpt_file": "model_bf16.safetensors",
                           "vit_min_size": 512, "max_think_token_n": 1000, "max_images": 4,
                           "save_dir": "{IMG_DIR}", "do_sample": False, "temperature": 0.3,
                           "extract_answer": True}, **CFG_GEN),
        about=("Interleaved text/image reasoning with the loop from the official "
               "`infz_bf16.py` (system prompt, prompt-then-images order, CFG 4.0/2.0, 50 denoising "
               "steps). Generates a visual thought after every reasoning step until it writes "
               "\"Final Answer:\" (capped at 4 visual thoughts). Greedy text decoding."),
    ),
    "thinkmorph": dict(
        label="ThinkMorph-7B", model_name="ThinkMorph-7B", model_class="ThinkMorphEval",
        hf_repo="ThinkMorph/ThinkMorph-7B",
        allow_patterns=["*.json", "*.txt", "model.safetensors", "ae.safetensors"],
        disk_gb=45, deps_key="bagel-tf4.49", pip=["transformers==4.49.0"] + COMMON_PIP,
        flash_attn_check=True, stub_thinkmorph=False, deltav=None,
        model_kwargs=dict({"model_path": "{MODEL_DIR}", "think": True, "understanding_output": False,
                           "save_dir": "{IMG_DIR}", "temperature": 0.3, "do_sample": False,
                           "max_think_token_n": 4096, "max_rounds": 3, "use_torch_compile": False,
                           "extract_answer": True}, **CFG_GEN),
        about=("Interleaved reasoning through the authors' own VLMEvalKit wrapper "
               "(`think=True, understanding_output=False`). Following the paper, a single pass uses "
               "temperature 0 (greedy) with up to 4,096 tokens; up to 3 visual thoughts, "
               "50 denoising steps."),
    ),
    "deltav": dict(
        label="DeltaV-2B", model_name="DeltaV-2B", model_class="DeltaVEval",
        hf_repo="wpj20000/DeltaV-2B",
        allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"],
        disk_gb=20, deps_key="deltav-tf4.57",
        pip=["transformers==4.57.0", "omegaconf"] + COMMON_PIP,
        flash_attn_check=False, stub_thinkmorph=True, deltav=DELTAV_REPO,
        model_kwargs={"model_path": "{MODEL_DIR}",
                      "extra_cfg_path": "{DELTAV_REPO}/configs/tsim_tok_cfg.json",
                      "tsim_intervals_path": "{DELTAV_REPO}/tools/data_processing/tsim_intervals.json",
                      "budget_key": "budget", "token_alpha": 0.8, "n_base": 144,
                      "exclude_base_in_output": True, "is_incremental_encoding": True,
                      "temperature": 0.7, "max_new_tokens": 16384,
                      "image_token_num_per_image": 145, "decode_and_save_image": False,
                      "vis_save_root": "{IMG_DIR}", "use_tsim_router": True,
                      "extract_answer": True},
        about=("Official DeltaV VLMEvalKit wrapper (`vlmevalkit/vlm/deltav_evalkit.py`) with the "
               "registration settings from the repo's MERGE.md. Decoding is greedy inside the wrapper "
               "(the temperature setting does not change it). The TSIM Router (DINOv2) assigns "
               "visual-update budgets to multi-image inputs so runs are reproducible."),
    ),
}


def build_run(model_key, bench_key):
    m = copy.deepcopy(MODELS[model_key])
    b = BENCHES[bench_key]
    kwargs = m["model_kwargs"]
    if model_key == "bagel" and bench_key == "cvbench3d":
        kwargs["think"] = False
    refs = [{"source": s, "value": v} for s, v in REFS.get((model_key, bench_key), [])]
    return {
        "model_key": model_key,
        "model_label": m["label"],
        "model_name": m["model_name"],
        "model_class": m["model_class"],
        "hf_repo": m["hf_repo"],
        "allow_patterns": m["allow_patterns"],
        "disk_gb": m["disk_gb"],
        "deps_key": m["deps_key"],
        "pip": m["pip"],
        "flash_attn_check": m["flash_attn_check"],
        "stub_thinkmorph": m["stub_thinkmorph"],
        "fork": FORK,
        "deltav": m["deltav"],
        "dataset": b["dataset"],
        "bench_label": b["label"],
        "bench_size": b["size"],
        "about": m["about"],
        "references": refs,
        "sync_every_min": 15,
        "model_kwargs": kwargs,
    }


# ----------------------------------------------------------------------------- cells

CELL_MO = "import marimo as mo"

CELL_IMPORTS = """\
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path"""

CELL_HEADER = '''\
_refs = RUN["references"]
_ref_rows = "\\n".join(f"| {r['source']} | {r['value']} |" for r in _refs) or "| (none published for this pair) | |"
mo.md(
    f"""
# {RUN['model_label']} × {RUN['bench_label']}

{RUN['about']}

**Benchmark:** `{RUN['dataset']}` in the ThinkMorph VLMEvalKit fork, {RUN['bench_size']}.
**Weights:** [`{RUN['hf_repo']}`](https://huggingface.co/{RUN['hf_repo']}) ·
**Eval code:** [`{RUN['fork']['repo']}`](https://github.com/{RUN['fork']['repo']}) @ `{RUN['fork']['sha'][:8]}`

**How to run**

1. Attach the GPU: notebook specs button in the header → **RTX Pro 6000 (Blackwell)** → save and restart.
2. Paste a Hugging Face token. Recommended: also give a private dataset repo for checkpoints
   (e.g. `your-name/molab-eval-results`; it is created if missing, the token needs write access).
   Predictions are uploaded every {RUN['sync_every_min']} minutes, so a session that is shut down
   (12-hour cap, 90-minute idle timeout, reclaimed GPU) resumes where it stopped when you press Start again.
3. Optional: an OpenAI key switches answer matching to the GPT-5 judge the papers used;
   without one, VLMEvalKit's exact matching is used.
4. Press **Start / resume evaluation** and keep the tab open. Use the smoke test (e.g. 20 samples) first.

| Published reference | Accuracy (%) |
|---|---|
{_ref_rows}

Published numbers come from different pipelines and judges; treat them as a sanity range, not a target.
"""
)'''

CELL_UI = '''\
ui_hf_token = mo.ui.text(kind="password", label="Hugging Face token", full_width=True)
ui_repo = mo.ui.text(
    label="HF dataset repo for checkpoints (empty = no sync)",
    placeholder="your-name/molab-eval-results",
    full_width=True,
)
ui_openai = mo.ui.text(kind="password", label="OpenAI API key (optional, enables GPT-5 judge)", full_width=True)
ui_judge = mo.ui.dropdown(
    options=["auto", "exact_matching", "gpt-5", "chatgpt-0125"],
    value="auto",
    label="Answer judge (auto = gpt-5 with a key, else exact_matching)",
)
ui_mode = mo.ui.dropdown(
    options={"Inference + scoring": "all", "Scoring only (re-score saved predictions)": "eval"},
    value="Inference + scoring",
    label="Mode",
)
ui_smoke = mo.ui.number(start=0, stop=100000, step=1, value=0, label="Smoke test: first N samples only (0 = full benchmark)")
ui_sync_images = mo.ui.checkbox(value=False, label="Also upload generated visual thoughts (images) to the HF repo")
ui_start = mo.ui.run_button(label="Start / resume evaluation", kind="success")
mo.vstack([ui_hf_token, ui_repo, ui_openai, mo.hstack([ui_judge, ui_mode], justify="start"), ui_smoke, ui_sync_images, ui_start])'''

CELL_ENV = '''\
def _probe():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


gpu_info = _probe()
_disk = []
for _p in sorted({"/tmp", str(Path.home()), os.getcwd()}):
    try:
        _disk.append(f"`{_p}`: {shutil.disk_usage(_p).free / 2**30:.0f} GiB free")
    except Exception:
        pass
mo.callout(
    mo.md(
        (f"**GPU:** {gpu_info}" if gpu_info else "**No GPU visible.** Attach the RTX Pro 6000 from the notebook specs button, then restart.")
        + f"  \\n**Python:** {sys.version.split()[0]}  \\n**Disk:** " + " · ".join(_disk)
    ),
    kind="success" if gpu_info else "warn",
)'''


def cell_sources():
    parts = ["# Helper files written to disk at run time (model wrappers, VLMEvalKit driver, HF sync, flash-attn fallback)."]
    parts.append("SRC = {}")
    for name in ["molab_models.py", "molab_driver.py", "molab_sync.py", "flash_attn_shim.py"]:
        text = (SRC / name).read_text()
        assert "'''" not in text, name
        assert not text.endswith("\\"), name
        parts.append(f"SRC[{name!r}] = r'''{text}'''")
    return "\n\n".join(parts)


CELL_HELPERS = '''\
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stream_cmd(cmd, env=None, cwd=None, log_file=None, progress_every=60):
    """Run a command, echo its output (progress bars throttled), return the exit code."""
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd)[:400])
    fh = open(log_file, "a", encoding="utf-8") if log_file else None
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, env=env, cwd=cwd, errors="replace",
    )
    last = 0.0
    try:
        for line in proc.stdout:
            if fh:
                fh.write(line)
                fh.flush()
            s = line.rstrip()
            if not s:
                continue
            if progress_every and ("it/s" in s or "s/it" in s or "%|" in s):
                if time.time() - last < progress_every:
                    continue
                last = time.time()
            print(s[-1500:], flush=True)
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
        if fh:
            fh.close()
    return proc.returncode


def py(code, env=None, timeout=900):
    """Run a Python snippet in a fresh interpreter; return (exit code, combined output)."""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=timeout)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def pip_install(args, constraints=None, log_file=None):
    attempts = []
    if shutil.which("uv"):
        attempts.append([shutil.which("uv"), "pip", "install", "--python", sys.executable])
    attempts.append([sys.executable, "-m", "pip", "install", "--disable-pip-version-check"])
    attempts.append([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--break-system-packages"])
    for base in attempts:
        cmd = base + (["-c", str(constraints)] if constraints else []) + list(args)
        if stream_cmd(cmd, log_file=log_file, progress_every=0) == 0:
            return True
    return False


def fetch_github(repo, sha, dest):
    """Download a pinned commit of a GitHub repo (no git needed)."""
    dest = Path(dest)
    marker = dest / ".molab_sha"
    if marker.exists() and marker.read_text().strip() == sha:
        return dest
    tmp = dest.parent / f".dl_{dest.name}"
    src = None
    for url in (f"https://codeload.github.com/{repo}/zip/{sha}", f"https://github.com/{repo}/archive/{sha}.zip"):
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        try:
            log(f"downloading {url}")
            zpath = tmp / "src.zip"
            urllib.request.urlretrieve(url, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(tmp)
            src = [p for p in tmp.iterdir() if p.is_dir()][0]
            break
        except Exception as e:
            log(f"zip download failed ({e})")
    if src is None and shutil.which("git"):
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        log(f"falling back to git for {repo}@{sha[:8]}")
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", f"https://github.com/{repo}.git"],
                    ["git", "fetch", "-q", "--depth", "1", "origin", sha], ["git", "checkout", "-q", "FETCH_HEAD"]):
            subprocess.run(cmd, cwd=tmp, check=True)
        shutil.rmtree(tmp / ".git", ignore_errors=True)
        src = tmp
    if src is None:
        raise RuntimeError(f"could not download {repo}@{sha}")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(src), str(dest))
    shutil.rmtree(tmp, ignore_errors=True)
    marker.write_text(sha)
    return dest


def pick_root():
    """Largest writable disk among the usual candidates (override with MOLAB_EVAL_ROOT)."""
    cands = []
    for p in [os.environ.get("MOLAB_EVAL_ROOT"), "/tmp", str(Path.home()), os.getcwd(), "/workspace", "/data", "/mnt"]:
        if p and os.path.isdir(p) and os.access(p, os.W_OK):
            cands.append((shutil.disk_usage(p).free, p))
    free, base = max(cands)
    root = Path(base) / "molab_eval"
    root.mkdir(parents=True, exist_ok=True)
    return root, free


def fill(obj, mapping):
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace("{" + k + "}", v)
        return obj
    if isinstance(obj, list):
        return [fill(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: fill(v, mapping) for k, v in obj.items()}
    return obj


def find_scores(work_dir, model_name, dataset):
    return sorted(Path(work_dir).glob(f"{model_name}/*/{model_name}_{dataset}_acc.csv"))


def tail(path, n=40):
    try:
        return "".join(Path(path).read_text(errors="replace").splitlines(True)[-n:])
    except Exception:
        return ""'''

CELL_RUN = '''\
mo.stop(
    not ui_start.value,
    mo.callout(mo.md("Fill in the fields above, then press **Start / resume evaluation**."), kind="info"),
)

_t0 = time.time()
hf_token = (ui_hf_token.value or os.environ.get("HF_TOKEN", "")).strip()
openai_key = (ui_openai.value or os.environ.get("OPENAI_API_KEY", "")).strip()
hf_repo = (ui_repo.value or os.environ.get("MOLAB_RESULTS_REPO", "")).strip()
mode = ui_mode.value
smoke_n = int(ui_smoke.value or 0)
judge = ui_judge.value
if judge == "auto":
    judge = "gpt-5" if openai_key else "exact_matching"
mo.stop(
    judge != "exact_matching" and not openai_key,
    mo.callout(mo.md(f"Judge `{judge}` needs an OpenAI API key."), kind="danger"),
)
mo.stop(
    mode == "all" and not gpu_info,
    mo.callout(mo.md("No GPU visible. Attach the RTX Pro 6000 from the notebook specs button and restart."), kind="danger"),
)
mo.stop(
    bool(hf_repo) and not hf_token,
    mo.callout(mo.md("Checkpoint sync needs a Hugging Face token with write access."), kind="danger"),
)

root, _free = pick_root()
mo.stop(
    mode == "all" and _free < RUN["disk_gb"] * 2**30 and not (root / "models" / RUN["model_key"] / ".complete").exists(),
    mo.callout(mo.md(f"Only {_free / 2**30:.0f} GiB free under `{root}`; about {RUN['disk_gb']} GiB are needed."), kind="danger"),
)
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
for _k in ("code", "lmu", "work"):
    P[_k].mkdir(parents=True, exist_ok=True)
log_file = P["work"] / f"run_{sync_prefix}.log"
log(f"working under {root} ({_free / 2**30:.0f} GiB free); judge={judge}; mode={mode}; smoke={smoke_n or 'off'}")

# 1. evaluation code (pinned commits) and helper files
log("Step 1/6: evaluation code")
fetch_github(RUN["fork"]["repo"], RUN["fork"]["sha"], P["fork"])
if RUN["deltav"]:
    fetch_github(RUN["deltav"]["repo"], RUN["deltav"]["sha"], P["deltav"])
for _name, _text in SRC.items():
    if _name != "flash_attn_shim.py":
        (P["code"] / _name).write_text(_text)

# 2. Python dependencies (once per session)
log("Step 2/6: Python dependencies")
_marker = root / f".deps_{RUN['deps_key']}"
if not _marker.exists():
    _req = P["code"] / "fork_requirements.txt"
    _lines = []
    for _l in (P["fork"] / "requirements.txt").read_text().splitlines():
        _s = _l.strip()
        if not _s or _s.startswith("#") or _s.split("=")[0].split(">")[0].strip() in ("gradio", "google-genai"):
            continue
        _lines.append("opencv-python-headless" if _s.startswith("opencv-python") else _s)
    _req.write_text("\\n".join(_lines) + "\\n")
    _rc, _out = py("import importlib.metadata as m; print(m.version('torch'))")
    _cons = None
    _tv = next((_l.strip() for _l in _out.splitlines() if _l.strip()[:1].isdigit()), None)
    if _rc == 0 and _tv:
        # keep molab's preinstalled CUDA build of torch
        _cons = root / "constraints.txt"
        _cons.write_text(f"torch=={_tv}\\n")
    _args = ["-r", str(_req)] + RUN["pip"]
    _ok = pip_install(_args, _cons, log_file) or pip_install(_args, None, log_file)
    mo.stop(not _ok, mo.callout(mo.md("Dependency installation failed; see the log above."), kind="danger"))
    _marker.write_text("ok")

# 3. CUDA / torch sanity check (Blackwell needs a CUDA 12.8+ torch build)
if mode == "all":
    log("Step 3/6: CUDA check")
    _cuda_test = (
        "import torch, torch.nn.functional as F; import torchvision; "
        "x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16); (x @ x).sum().item(); "
        "q = torch.randn(1, 2, 16, 64, device='cuda', dtype=torch.bfloat16); F.scaled_dot_product_attention(q, q, q); "
        "print('CUDA' + '_OK', torch.__version__, torch.cuda.get_device_name(0))"
    )
    _rc, _out = py(_cuda_test)
    if _rc != 0 or "CUDA_OK" not in _out:
        log("torch cannot run on this GPU; installing a CUDA 12.8 build")
        print(_out[-2000:])
        pip_install(["--upgrade", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128"], None, log_file)
        _rc, _out = py(_cuda_test)
    log(_out.strip().splitlines()[-1] if _out.strip() else "no output")
    mo.stop(_rc != 0 or "CUDA_OK" not in _out, mo.callout(mo.md("PyTorch cannot use the GPU:\\n```\\n" + _out[-1500:] + "\\n```"), kind="danger"))

# BAGEL-family code imports flash-attn; prebuilt wheels rarely match sm_120 + the newest torch,
# so fall back to a PyTorch SDPA implementation of flash_attn_varlen_func when it is unusable.
_use_shim = False
if RUN["flash_attn_check"]:
    if mode == "all":
        _fa = (
            "import torch; from flash_attn import flash_attn_varlen_func as f; "
            "q = torch.randn(8, 4, 64, device='cuda', dtype=torch.bfloat16); "
            "cu = torch.tensor([0, 8], device='cuda', dtype=torch.int32); f(q, q, q, cu, cu, 8, 8, causal=True); print('FA' + '_OK')"
        )
    else:
        _fa = "from flash_attn import flash_attn_varlen_func; print('FA' + '_OK')"
    _rc, _out = py(_fa)
    _use_shim = _rc != 0 or "FA_OK" not in _out
    if _use_shim:
        (P["shims"] / "flash_attn").mkdir(parents=True, exist_ok=True)
        (P["shims"] / "flash_attn" / "__init__.py").write_text(SRC["flash_attn_shim.py"])
        log("flash-attn not usable here; using the PyTorch SDPA fallback (same math, somewhat slower)")
    else:
        log("flash-attn works on this GPU")

# environment shared by all subprocesses
env = dict(os.environ)
_pp = ([str(P["shims"])] if _use_shim else []) + [str(P["code"]), str(P["fork"])]
if RUN["deltav"]:
    _pp.append(str(P["deltav"]))
if env.get("PYTHONPATH"):
    _pp.append(env["PYTHONPATH"])
env.update({
    "PYTHONPATH": os.pathsep.join(_pp),
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
if hf_token:
    env["HF_TOKEN"] = hf_token
if openai_key:
    env["OPENAI_API_KEY"] = openai_key

# 4. model weights
if mode == "all":
    log(f"Step 4/6: downloading {RUN['hf_repo']}")
    _done = P["model"] / ".complete"
    P["model"].mkdir(parents=True, exist_ok=True)
    if not _done.exists():
        _dl = (
            "from huggingface_hub import snapshot_download; "
            f"snapshot_download(repo_id={RUN['hf_repo']!r}, local_dir={str(P['model'])!r}, "
            f"allow_patterns={RUN['allow_patterns']!r}, max_workers=8); print('DL_OK')"
        )
        _rc = stream_cmd([sys.executable, "-c", _dl], env=env, log_file=log_file, progress_every=30)
        mo.stop(_rc != 0, mo.callout(mo.md("Model download failed; see the log above."), kind="danger"))
        _done.write_text("ok")

# 5. spec + restore checkpoints
_spec = {
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
    "sync_images": bool(ui_sync_images.value),
    "flash_attn_shim": str(P["shims"] / "flash_attn" / "__init__.py") if _use_shim else None,
}
spec_path = P["code"] / f"spec_{sync_prefix}.json"
spec_path.write_text(json.dumps(_spec, indent=2))
_sync_cmd = [sys.executable, str(P["code"] / "molab_sync.py")]
if hf_repo:
    log(f"Step 5/6: restoring checkpoints from {hf_repo}/{sync_prefix}")
    stream_cmd(_sync_cmd + ["pull", spec_path], env=env, log_file=log_file)
else:
    log("Step 5/6: checkpoint sync disabled (no HF dataset repo given)")

# 6. evaluation (VLMEvalKit run.py through the driver), with periodic checkpoint upload
log(f"Step 6/6: VLMEvalKit {run_name} x {RUN['dataset']}")
_stop = threading.Event()


def _sync_loop():
    while not _stop.wait(RUN["sync_every_min"] * 60):
        with open(P["work"] / "sync.log", "a") as _fh:
            subprocess.run(_sync_cmd + ["push", str(spec_path)], env=env, stdout=_fh, stderr=subprocess.STDOUT)


_th = threading.Thread(target=_sync_loop, daemon=True)
if hf_repo:
    _th.start()
try:
    exit_code = stream_cmd([sys.executable, "-u", P["code"] / "molab_driver.py", spec_path], env=env, log_file=log_file)
finally:
    _stop.set()
    if hf_repo:
        log("uploading checkpoints")
        stream_cmd(_sync_cmd + ["push", spec_path], env=env, log_file=log_file)

score_files = find_scores(P["work"], run_name, RUN["dataset"])
run_result = {
    "exit_code": exit_code,
    "work_dir": P["work"],
    "run_name": run_name,
    "score_files": score_files,
    "log_file": log_file,
    "minutes": (time.time() - _t0) / 60,
}
mo.callout(
    mo.md(
        f"Finished in {run_result['minutes']:.0f} min. Scores: `{score_files[-1]}`" if score_files
        else "No score file was produced. The last log lines:\\n```\\n" + tail(log_file) + "\\n```"
    ),
    kind="success" if score_files else "danger",
)'''

CELL_RESULTS = '''\
import io
import pandas as pd

_files = run_result["score_files"]
if not _files:
    _view = mo.md("No scores yet.")
else:
    _df = pd.read_csv(_files[-1])
    _buf = io.BytesIO()
    with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _z:
        for _p in Path(run_result["work_dir"]).rglob("*"):
            if _p.is_file() and not _p.is_symlink() and "visual_thoughts" not in _p.parts:
                _z.write(_p, _p.relative_to(run_result["work_dir"]))
    _view = mo.vstack([
        mo.md(f"### {RUN['model_label']} on {RUN['bench_label']} (accuracy, %)"),
        mo.ui.table(_df, selection=None),
        mo.md("Overall accuracy is the `Overall` column. Full reasoning traces are in the `traces_*.jsonl` file inside the zip."),
        mo.download(data=_buf.getvalue(), filename=f"{run_result['run_name']}__{RUN['dataset']}_results.zip", label="Download predictions, scores and traces"),
    ])
_view'''


def make_notebook(model_key, bench_key):
    run = build_run(model_key, bench_key)
    run_src = "# Edit model_kwargs to change decoding / generation settings.\nRUN = " + pprint.pformat(run, width=100, sort_dicts=False)
    codes = [CELL_MO, CELL_IMPORTS, run_src, CELL_HEADER, CELL_UI, CELL_ENV, cell_sources(),
             CELL_HELPERS, CELL_RUN, CELL_RESULTS]
    names = ["_"] * len(codes)
    configs = [CellConfig() for _ in codes]
    for i in (3,):
        configs[i] = CellConfig(hide_code=True)
    configs[6] = CellConfig(hide_code=True)
    header = (f"# {run['model_label']} x {run['bench_label']} - molab evaluation notebook "
              "(VLMEvalKit, ThinkMorph fork)\n")
    text = generate_filecontents(
        codes, names, configs,
        config=_AppConfig(width="medium", app_title=f"{run['model_label']} x {run['bench_label']}"),
        header_comments=header,
    )
    return text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.py"):
        f.unlink()
    for mk in MODELS:
        for bk in BENCHES:
            path = OUT / f"{mk}__{bk}.py"
            path.write_text(make_notebook(mk, bk))
            print("wrote", path.name)


if __name__ == "__main__":
    main()
