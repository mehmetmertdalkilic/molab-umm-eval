"""Generate the Qwen2.5-VL-7B and SenseNova-U1 evaluation notebooks (molab + Colab, 4 in total).

Each notebook runs SAT, V* and MMVP one after another with the ThinkMorph VLMEvalKit fork
(exact matching), then repairs the answers exact matching cannot read with a Gemini judge
(src/molab_judge.py: the same procedure and prompt as the llm_judge_repair notebook).
"""
import copy
import json
import pprint

import make_colab_notebooks as mc
import make_notebooks as mk

COLAB_OUT = mc.OUT
MOLAB_OUT = mk.OUT
BENCH_KEYS = ["sat", "vstar", "mmvp"]
U1_REPO = {"repo": "OpenSenseNova/SenseNova-U1", "sha": "1793b3bcb196da2a0a86b282744392ff0ad0f7a3",
           "dir": "SenseNova-U1", "pythonpath": "src"}

MODELS = {
    "qwen25vl": dict(
        label="Qwen2.5-VL-7B-Instruct", model_name="Qwen2.5-VL-7B-Instruct", model_class="Qwen25VLEval",
        hf_repo="Qwen/Qwen2.5-VL-7B-Instruct",
        allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"],
        disk_gb=25, deps_key="qwen25vl-tf4.49",
        pip=["transformers==4.49.0", "qwen-vl-utils==0.0.11"] + mk.COMMON_PIP,
        # the fork's ThinkMorph package imports flash_attn at import time; Qwen does not need it
        stub_thinkmorph=True, extra_repos=[],
        min_gpu_gib=22, default_gpu="L4", gpus="L4 or A100 (Pro), or G4 / H100 (Pro+)",
        model_kwargs={"model_path": "{MODEL_DIR}", "min_pixels": 1280 * 28 * 28, "max_pixels": 16384 * 28 * 28,
                      "use_custom_prompt": False, "max_new_tokens": 2048, "attn_implementation": "auto"},
        about=("Qwen2.5-VL-7B-Instruct through the fork's own `Qwen2VLChat` wrapper with the settings of its "
               "`Qwen2.5-VL-7B-Instruct` entry: images resized to between 1,280 and 16,384 visual tokens "
               "(min/max_pixels 1280·28·28 and 16384·28·28), the benchmark's standard prompt (no custom "
               "prompt), greedy decoding (top_k 1), up to 2,048 new tokens. Flash-attention 2 when it is "
               "installed, otherwise PyTorch SDPA attention (the only change to the wrapper)."),
        runtime="about 30–60 minutes for all three benchmarks on an A100 or L4, plus the 16 GB download",
        thinking_option=False,
    ),
    "sensenova_u1": dict(
        label="SenseNova-U1-8B-MoT", model_name="SenseNova-U1-8B-MoT", model_class="SenseNovaU1Eval",
        hf_repo="sensenova/SenseNova-U1-8B-MoT",
        allow_patterns=["*.json", "*.txt", "*.jinja", "*.safetensors"],
        disk_gb=45, deps_key="sensenova-u1-tf4.57.1",
        pip=["transformers==4.57.1", "accelerate>=1.1"] + mk.COMMON_PIP,
        stub_thinkmorph=True, extra_repos=[U1_REPO],
        min_gpu_gib=39, default_gpu="A100", gpus="G4 or H100 (Pro+), or an 80 GB A100; a 40 GB A100 is very tight",
        model_kwargs={"model_path": "{MODEL_DIR}", "think": True, "max_new_tokens": 8192, "do_sample": False,
                      "min_pixels": 65536, "max_pixels": 4194304, "attn_backend": "auto", "extract_answer": True},
        about=("SenseNova-U1-8B-MoT (NEO-unify, no separate vision encoder) through the official "
               "`sensenova_u1` package (OpenSenseNova/SenseNova-U1 @ 1793b3bc), with the settings of "
               "SenseNova's own VLMEvalKit evaluation (`evaluation/easi/config/sensenova_models.py`): "
               "thinking on, greedy decoding (temperature 0), up to 8,192 new tokens. The prompt is exactly "
               "what the checkpoint's chat template renders (`<image>` per image, then the benchmark's "
               "standard question); images are preprocessed as in the repo's VQA example (at most "
               "2048×2048 pixels). The answer is the text after `</think>`; an answer still reasoning at "
               "the token limit has no answer, as in SenseNova's own server (empty prediction)."),
        runtime=("hard to predict because it depends on how long the model thinks: roughly 30–90 s per "
                 "question, so about 6–20 hours for the 791 answers on a G4 / H100. Run the smoke test "
                 "first: the progress bar shows seconds per question"),
        thinking_option=True,
    ),
}


def build_run(model_key, bench_key):
    m = copy.deepcopy(MODELS[model_key])
    b = mk.BENCHES[bench_key]
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
        "flash_attn_check": False,
        "stub_thinkmorph": m["stub_thinkmorph"],
        "fork": mk.FORK,
        "deltav": None,
        "extra_repos": m["extra_repos"],
        "dataset": b["dataset"],
        "bench_label": b["label"],
        "bench_size": b["size"],
        "min_gpu_gib": m["min_gpu_gib"],
        "colab_gpus": m["gpus"],
        "sync_every_min": 15,
        "model_kwargs": m["model_kwargs"],
    }


def runs_source(model_key):
    runs = {bk: build_run(model_key, bk) for bk in BENCH_KEYS}
    return "RUNS = " + pprint.pformat(runs, width=100, sort_dicts=False)


def sources():
    parts = ["# Helper files written to disk at run time (model wrappers, VLMEvalKit driver, HF sync, Gemini judge)."]
    parts.append("SRC = {}")
    for name in ["molab_models.py", "molab_driver.py", "molab_sync.py", "molab_judge.py",
                 "flash_attn_shim.py", "msda_shim.py"]:
        text = (mk.SRC / name).read_text()
        assert "'''" not in text, name
        assert not text.endswith("\\"), name
        parts.append(f"SRC[{name!r}] = r'''{text}'''")
    return "\n\n".join(parts)


# One benchmark, end to end. Shared by the molab and Colab notebooks (secrets come from Colab
# Secrets or from environment variables, which the molab notebook fills from its form).
RUN_FN = '''

def selected_runs(RUNS, S):
    """The benchmarks ticked in the settings, with the thinking setting applied."""
    out = []
    for key, flag in (("sat", "RUN_SAT"), ("vstar", "RUN_VSTAR"), ("mmvp", "RUN_MMVP")):
        if S.get(flag, True):
            run = json.loads(json.dumps(RUNS[key]))
            if "think" in run["model_kwargs"] and not S.get("THINKING", True):
                run["model_kwargs"]["think"] = False
                run["model_name"] += "-nothink"
                run["model_label"] += " (thinking off)"
            out.append((key, run))
    return out


def run_eval(RUN, S):
    t0 = time.time()
    hf_token = secret("HF_TOKEN")
    gemini_key = secret("GEMINI_API_KEY") or secret("GOOGLE_API_KEY")
    mode = S["MODE"]
    smoke_n = int(S["SMOKE_TEST_N"] or 0)
    ckpt = S["CHECKPOINTS"]
    hf_repo = S["HF_RESULTS_REPO"].strip() if ckpt == "Hugging Face repo" else ""
    if ckpt == "Hugging Face repo" and not (hf_repo and hf_token):
        raise RuntimeError("Hugging Face checkpoints need a results repo and a Hugging Face token with write access.")

    gpu = gpu_query()
    if mode == "all":
        if not gpu:
            raise RuntimeError("No GPU. Attach one: " + RUN["colab_gpus"] + ".")
        if gpu["mem_gib"] < RUN["min_gpu_gib"] or gpu["cc"] < 8.0:
            raise RuntimeError(
                f"{gpu['name']} ({gpu['mem_gib']:.0f} GiB, compute capability {gpu['cc']}) cannot run "
                f"{RUN['model_label']}: it needs at least {RUN['min_gpu_gib']} GiB and bf16 support. "
                "Use " + RUN["colab_gpus"] + ".")
        log(f"GPU: {gpu['name']}, {gpu['mem_gib']:.0f} GiB")

    root = Path("/content/umm_eval") if Path("/content").is_dir() else pick_root()[0]
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    run_name = RUN["model_name"] + (f"-smoke{smoke_n}" if smoke_n else "")
    sync_prefix = f"{run_name}__{RUN['dataset']}"
    P = {
        "code": root / "code",
        "fork": root / "VLMEvalKit_Thinkmorph",
        "model": root / "models" / RUN["model_key"],
        "lmu": root / "LMUData",
        "work": root / "results" / sync_prefix,
        "judge_pkgs": root / "judge_pkgs",
    }
    for k in ("code", "lmu", "work"):
        P[k].mkdir(parents=True, exist_ok=True)
    if mode == "all" and free < RUN["disk_gb"] * 2**30 and not (P["model"] / ".complete").exists():
        raise RuntimeError(f"Only {free / 2**30:.0f} GiB free under {root}; about {RUN['disk_gb']} GiB are needed.")
    log_file = P["work"] / f"run_{sync_prefix}.log"
    log(f"{RUN['model_label']} x {RUN['bench_label']}: working under {root} ({free / 2**30:.0f} GiB free); "
        f"mode={mode}; smoke={smoke_n or 'off'}; checkpoints={ckpt}")

    drive_dir = None
    if ckpt == "Google Drive":
        from google.colab import drive
        drive.mount("/content/drive")
        drive_dir = Path("/content/drive/MyDrive") / S["DRIVE_FOLDER"].strip("/ ") / sync_prefix
        drive_dir.mkdir(parents=True, exist_ok=True)

    # 1. evaluation code (pinned commits) and helper files
    log("Step 1/7: evaluation code")
    fetch_github(RUN["fork"]["repo"], RUN["fork"]["sha"], P["fork"])
    extra_paths = []
    for r in RUN.get("extra_repos", []):
        d = fetch_github(r["repo"], r["sha"], root / r["dir"])
        extra_paths.append(str(Path(d) / r.get("pythonpath", "")))
    for name, text in SRC.items():
        if name != "flash_attn_shim.py":
            (P["code"] / name).write_text(text)

    # 2. Python dependencies (once per runtime)
    log("Step 2/7: Python dependencies")
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
            cons = root / "constraints.txt"  # keep the preinstalled CUDA build of torch
            cons.write_text(f"torch=={tv}\\n")
        args = ["-r", str(req)] + RUN["pip"]
        if not (pip_install(args, cons, log_file) or pip_install(args, None, log_file)):
            raise RuntimeError("Dependency installation failed; see the log above.")
        for other in root.glob(".deps_*"):
            other.unlink()  # another model's packages were replaced
        marker.write_text("ok")

    # 3. CUDA / torch check
    if mode == "all":
        log("Step 3/7: CUDA check")
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

    env = dict(os.environ)
    pp = [str(P["code"]), str(P["fork"])] + extra_paths
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
    env.pop("GEMINI_API_KEY", None)  # only the judge step gets the key
    env.pop("GOOGLE_API_KEY", None)
    if hf_token:
        env["HF_TOKEN"] = hf_token

    # 4. model weights (once per runtime)
    if mode == "all":
        log(f"Step 4/7: downloading {RUN['hf_repo']}")
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
        "deltav_repo": None,
        "extra_paths": extra_paths,
        "stub_thinkmorph": RUN["stub_thinkmorph"],
        "model_class": RUN["model_class"],
        "model_name": run_name,
        "model_kwargs": fill(RUN["model_kwargs"], {"MODEL_DIR": str(P["model"])}),
        "dataset": RUN["dataset"],
        "work_dir": str(P["work"]),
        "judge": "exact_matching",
        "mode": mode,
        "smoke_test_n": smoke_n,
        "eval_day": "molab",
        "eval_hash": RUN["fork"]["sha"][:8],
        "hf_repo": hf_repo,
        "sync_prefix": sync_prefix,
        "sync_images": False,
        "flash_attn_shim": None,
    }
    spec_path = P["code"] / f"spec_{sync_prefix}.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    sync_cmd = [sys.executable, str(P["code"] / "molab_sync.py")]

    def push():
        if drive_dir is not None:
            n = copy_tree(P["work"], drive_dir, exts=RESULT_EXT)
            with open(P["work"] / "sync.log", "a") as fh:
                fh.write(f"{time.strftime('%H:%M:%S')} copied {n} file(s) to {drive_dir}\\n")
        elif hf_repo:
            with open(P["work"] / "sync.log", "a") as fh:
                subprocess.run(sync_cmd + ["push", str(spec_path)], env=env, stdout=fh, stderr=subprocess.STDOUT)

    if drive_dir is not None:
        log(f"Step 5/7: restored {copy_tree(drive_dir, P['work'], only_missing=True)} file(s) from {drive_dir}")
    elif hf_repo:
        log(f"Step 5/7: restoring checkpoints from {hf_repo}/{sync_prefix}")
        if stream_cmd(sync_cmd + ["pull", spec_path], env=env, log_file=log_file) != 0:
            raise RuntimeError("Could not restore the saved checkpoints from Hugging Face (see the log above). "
                               "Stopped so a fresh run cannot overwrite your saved progress; "
                               "wait a few minutes and start again.")
    else:
        log("Step 5/7: checkpoints disabled")

    # 6. VLMEvalKit inference + exact-matching scores, with periodic checkpoint copies
    log(f"Step 6/7: VLMEvalKit {run_name} x {RUN['dataset']}")
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

    # 7. Gemini judge for the answers exact matching could not read
    judge_csv = P["work"] / "gemini_judge" / f"{run_name}_{RUN['dataset']}_judge_scores.csv"
    if not score_files:
        log("Step 7/7: no exact-matching score yet, so the Gemini judge is skipped. Last log lines:\\n" + tail(log_file))
    elif not gemini_key:
        log("Step 7/7: no GEMINI_API_KEY, so the Gemini judge is skipped (exact-matching scores only).")
    else:
        log(f"Step 7/7: Gemini judge ({S['GEMINI_MODEL']}, thinking level {S['THINKING_LEVEL']})")
        if not (P["judge_pkgs"] / ".ok").exists():
            # installed into its own folder so it cannot change the evaluation's packages
            if not pip_install(["--target", str(P["judge_pkgs"]), "google-genai"], None, log_file):
                raise RuntimeError("Could not install google-genai for the judge; see the log above.")
            (P["judge_pkgs"] / ".ok").write_text("ok")
        jspec = P["code"] / f"judge_{sync_prefix}.json"
        jspec.write_text(json.dumps({"work_dir": str(P["work"]), "model_name": run_name, "dataset": RUN["dataset"],
                                     "gemini_model": S["GEMINI_MODEL"], "thinking_level": S["THINKING_LEVEL"],
                                     "parallel": 4}))
        jenv = dict(env, GEMINI_API_KEY=gemini_key,
                    PYTHONPATH=os.pathsep.join([str(P["judge_pkgs"]), str(P["code"])]))
        if stream_cmd([sys.executable, "-u", P["code"] / "molab_judge.py", jspec], env=jenv, log_file=log_file) != 0:
            log("The Gemini judge failed (see above); the exact-matching score is still valid. Run again to retry.")
        if drive_dir is not None or hf_repo:
            push()
    minutes = (time.time() - t0) / 60
    log(f"{RUN['bench_label']} finished in {minutes:.0f} min.")
    return {"exit_code": exit_code, "work_dir": P["work"], "run_name": run_name, "dataset": RUN["dataset"],
            "bench_label": RUN["bench_label"], "model_label": RUN["model_label"], "score_files": score_files,
            "judge_csv": judge_csv if judge_csv.exists() else None, "log_file": log_file, "minutes": minutes}


def results_table(results):
    """One row per benchmark: exact matching, with the Gemini judge, and what the judge did."""
    import pandas as pd
    rows, warnings = [], []
    for key, res in results.items():
        row = {"benchmark": res["bench_label"]}
        for x in Path(res["work_dir"]).rglob(f"{res['run_name']}_{res['dataset']}.xlsx"):
            if x.is_symlink() or "gemini_judge" in x.parts:
                continue
            pred = pd.read_excel(x)["prediction"].astype(str)
            bad = pred[pred.str.startswith("Failed to obtain answer")]
            if len(bad):
                warnings.append(f"{res['bench_label']}: {len(bad)} of {len(pred)} answers are model errors, not answers, "
                                f"so its score is NOT valid. Running again retries them. First error: {bad.iloc[0][:300]}")
        if res["score_files"]:
            v = float(pd.read_csv(res["score_files"][-1])["Overall"].iloc[0])
            row["exact matching"] = round(v * 100 if v <= 1 else v, 2)
        if res.get("judge_csv"):
            j = pd.read_csv(res["judge_csv"]).iloc[0]
            row["with Gemini judge"] = j["with Gemini judge"]
            for c in ("unreadable answers", "sent to judge", "judge found an option", "of which right",
                      "empty or cut off (not judged)", "crashed answers"):
                row[c] = int(j[c])
        rows.append(row)
    return pd.DataFrame(rows), warnings


def zip_results(results, zpath):
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for res in results.values():
            base = Path(res["work_dir"])
            for p in base.rglob("*"):
                if p.is_file() and not p.is_symlink():
                    z.write(p, Path(base.name) / p.relative_to(base))
    return zpath
'''


# ------------------------------------------------------------------------------------- Colab


def colab_header(m, colab_url):
    think = ("\n**Thinking:** on by default (`THINKING` in the settings). Thinking off runs as "
             f"`{m['model_name']}-nothink`, so both can be kept side by side.\n" if m["thinking_option"] else "")
    return f"""# {m['label']} × SAT, V\\* and MMVP

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url})

{m['about']}
{think}
**Benchmarks:** `SAT_circular` (150 questions, circular evaluation), `VStarBench` (191) and `MMVP` (300), one after
another, in the ThinkMorph VLMEvalKit fork, the same benchmark files and prompts as the other models.
**Weights:** [`{m['hf_repo']}`](https://huggingface.co/{m['hf_repo']}) ·
**Eval code:** [`{mk.FORK['repo']}`](https://github.com/{mk.FORK['repo']}) @ `{mk.FORK['sha'][:8]}`

**Scoring.** VLMEvalKit's exact matching first (its `acc.csv`), then **Gemini 3 Flash** repairs only the answers exact
matching cannot read, with VLMEvalKit's own judge prompt: the same procedure as the `llm_judge_repair` notebook.
Empty answers and answers cut off by the token limit are not sent to the judge and count as wrong.

### How to run

1. **GPU:** Runtime → Change runtime type → **{m['gpus']}**.
2. **Secrets** (key icon in the left sidebar, then give this notebook access): `GEMINI_API_KEY` for the judge
   (without it you get exact-matching scores only); `HF_TOKEN` is optional (faster downloads; needed for Hugging Face checkpoints).
3. **Settings** below: checkpoints go to Google Drive by default (`MyDrive/umm_eval_results/<model>__<benchmark>`, the same layout
   as the other notebooks), so a run that is cut off resumes where it stopped. Use the smoke test (e.g. 10 samples) first.
4. **Runtime → Run all**, approve the Google Drive prompt, and keep the tab open.

**Expected time:** {m['runtime']}. If the runtime disconnects, reconnect and **Run all** again: finished
benchmarks are re-scored from their checkpoints in seconds and the rest continue where they stopped.
"""


def colab_settings(m):
    think = ('#@markdown Thinking (SenseNova\'s own evaluation uses it). Off runs as a separate model name ("-nothink").\n'
             'THINKING = True  #@param {type:"boolean"}\n') if m["thinking_option"] else "THINKING = True\n"
    return f'''#@title 1. Settings
#@markdown Where checkpoints (predictions, partial progress, scores, traces) are saved so an interrupted run can resume.
CHECKPOINTS = "Google Drive"  #@param ["Google Drive", "Hugging Face repo", "none"]
DRIVE_FOLDER = "umm_eval_results"  #@param {{type:"string"}}
HF_RESULTS_REPO = ""  #@param {{type:"string"}}
#@markdown Benchmarks to run (in this order).
RUN_SAT = True  #@param {{type:"boolean"}}
RUN_VSTAR = True  #@param {{type:"boolean"}}
RUN_MMVP = True  #@param {{type:"boolean"}}
{think}#@markdown Gemini judge (needs the `GEMINI_API_KEY` secret).
GEMINI_MODEL = "gemini-3-flash-preview"  #@param {{type:"string"}}
THINKING_LEVEL = "low"  #@param ["minimal", "low", "medium", "high"]
#@markdown `eval` re-scores saved predictions (and re-runs the judge) without running the model.
MODE = "all"  #@param ["all", "eval"]
#@markdown Smoke test: first N samples of each benchmark only (0 = full benchmarks).
SMOKE_TEST_N = 0  #@param {{type:"integer"}}

SETTINGS = dict(CHECKPOINTS=CHECKPOINTS, DRIVE_FOLDER=DRIVE_FOLDER, HF_RESULTS_REPO=HF_RESULTS_REPO,
                RUN_SAT=RUN_SAT, RUN_VSTAR=RUN_VSTAR, RUN_MMVP=RUN_MMVP, THINKING=THINKING,
                GEMINI_MODEL=GEMINI_MODEL, THINKING_LEVEL=THINKING_LEVEL, MODE=MODE, SMOKE_TEST_N=SMOKE_TEST_N)
'''


COLAB_RUN = '''#@title 4. Run the evaluation (benchmarks one after another)
RESULTS = {}
for _key, _run in selected_runs(RUNS, SETTINGS):
    print("=" * 100)
    RESULTS[_key] = run_eval(_run, SETTINGS)
'''

COLAB_RESULTS = '''#@title 5. Results (accuracy table and download)
from IPython.display import display

_table, _warnings = results_table(RESULTS)
for _w in _warnings:
    print("WARNING:", _w)
if RESULTS:
    _first = next(iter(RESULTS.values()))
    print(f"{_first['model_label']}: accuracy (%). SAT counts a question only if every option order is right.")
    display(_table)
    _zpath = zip_results(RESULTS, Path("/content") / f"{_first['run_name']}__sat_vstar_mmvp_results.zip")
    print("Predictions, scores, judge replies and reasoning traces:", _zpath)
    try:
        from google.colab import files
        files.download(str(_zpath))
    except Exception:
        pass
'''


def make_colab(model_key):
    m = MODELS[model_key]
    name = f"{model_key}__sat_vstar_mmvp.ipynb"
    colab_url = f"https://colab.research.google.com/github/{mc.REPO_PATH}/blob/main/colab_eval_notebooks/{name}"
    helpers = ("#@title 3. Helper code (no need to edit)\n"
               "import json\nimport os\nimport shutil\nimport subprocess\nimport sys\nimport threading\n"
               "import time\nimport urllib.request\nimport zipfile\nfrom pathlib import Path\n\n"
               + sources() + "\n\n" + mk.CELL_HELPERS + mc.COLAB_EXTRA_HELPERS + RUN_FN)
    runs = ("#@title 2. Model and benchmark settings (edit model_kwargs to change decoding / generation)\n"
            + runs_source(model_key) + "\n")
    cells = [
        mc.md_cell(colab_header(m, colab_url)),
        mc.code_cell(colab_settings(m), form=True),
        mc.code_cell(runs, form=True),
        mc.code_cell(helpers, form=True),
        mc.code_cell(COLAB_RUN, form=True),
        mc.code_cell(COLAB_RESULTS, form=True),
    ]
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": m["default_gpu"], "machine_shape": "hm", "name": name},
            "accelerator": "GPU",
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    return name, json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


# ------------------------------------------------------------------------------------- molab


def molab_header(m):
    think = ("\n**Thinking:** on by default (checkbox below). Thinking off runs as "
             f"`{m['model_name']}-nothink`, so both can be kept side by side.\n" if m["thinking_option"] else "")
    text = f"""# {m['label']} × SAT, V\\* and MMVP

{m['about']}
{think}
**Benchmarks:** `SAT_circular` (150 questions, circular evaluation), `VStarBench` (191) and `MMVP` (300), one after
another, in the ThinkMorph VLMEvalKit fork, the same benchmark files and prompts as the other models.
**Weights:** [`{m['hf_repo']}`](https://huggingface.co/{m['hf_repo']}) ·
**Eval code:** [`{mk.FORK['repo']}`](https://github.com/{mk.FORK['repo']}) @ `{mk.FORK['sha'][:8]}`

**Scoring.** VLMEvalKit's exact matching first (its `acc.csv`), then **Gemini 3 Flash** repairs only the answers exact
matching cannot read, with VLMEvalKit's own judge prompt: the same procedure as the `llm_judge_repair` notebook.
Empty answers and answers cut off by the token limit are not sent to the judge and count as wrong.

**How to run**

1. Attach the GPU: notebook specs button in the header → **RTX Pro 6000 (Blackwell)** → save and restart.
2. Paste a Hugging Face token and your results repo (e.g. `your-name/molab-eval-results`; the token needs write access).
   Predictions are uploaded every 15 minutes to `<model>__<benchmark>`, the same layout as the other notebooks, so a
   session that is shut down (12-hour cap, 90-minute idle timeout) resumes where it stopped when you press Start again.
3. Paste a Gemini API key for the judge (without one you get exact-matching scores only).
4. Press **Start / resume evaluation** and keep the tab open. Use the smoke test (e.g. 10 samples) first.

**Expected time:** {m['runtime']}.
"""
    assert '"""' not in text
    return 'mo.md(r"""' + text + '""")'


def molab_ui(m):
    think = ('ui_thinking = mo.ui.checkbox(value=True, label="Thinking (SenseNova\'s own evaluation uses it; off runs as \'-nothink\')")\n'
             if m["thinking_option"] else 'ui_thinking = mo.ui.checkbox(value=True, label="Thinking (not used by this model)", disabled=True)\n')
    return f'''ui_hf_token = mo.ui.text(kind="password", label="Hugging Face token", full_width=True)
ui_repo = mo.ui.text(
    label="HF dataset repo for checkpoints (empty = no sync)",
    placeholder="your-name/molab-eval-results",
    full_width=True,
)
ui_gemini = mo.ui.text(kind="password", label="Gemini API key (for the judge; empty = exact matching only)", full_width=True)
ui_gemini_model = mo.ui.text(value="gemini-3-flash-preview", label="Gemini model")
ui_thinking_level = mo.ui.dropdown(options=["minimal", "low", "medium", "high"], value="low", label="Judge thinking level")
ui_sat = mo.ui.checkbox(value=True, label="SAT")
ui_vstar = mo.ui.checkbox(value=True, label="V*")
ui_mmvp = mo.ui.checkbox(value=True, label="MMVP")
{think}ui_mode = mo.ui.dropdown(
    options={{"Inference + scoring": "all", "Scoring only (re-score saved predictions, re-run the judge)": "eval"}},
    value="Inference + scoring",
    label="Mode",
)
ui_smoke = mo.ui.number(start=0, stop=100000, step=1, value=0, label="Smoke test: first N samples of each benchmark (0 = full)")
ui_start = mo.ui.run_button(label="Start / resume evaluation", kind="success")
mo.vstack([ui_hf_token, ui_repo, ui_gemini, mo.hstack([ui_gemini_model, ui_thinking_level], justify="start"),
           mo.hstack([mo.md("**Benchmarks:**"), ui_sat, ui_vstar, ui_mmvp], justify="start"),
           ui_thinking, ui_mode, ui_smoke, ui_start])'''


MOLAB_RUN = '''\
mo.stop(
    not ui_start.value,
    mo.callout(mo.md("Fill in the fields above, then press **Start / resume evaluation**."), kind="info"),
)
for _name, _value in (("HF_TOKEN", ui_hf_token.value), ("GEMINI_API_KEY", ui_gemini.value)):
    if (_value or "").strip():
        os.environ[_name] = _value.strip()
_repo = (ui_repo.value or os.environ.get("MOLAB_RESULTS_REPO", "")).strip()
settings = dict(
    CHECKPOINTS="Hugging Face repo" if _repo else "none", DRIVE_FOLDER="", HF_RESULTS_REPO=_repo,
    RUN_SAT=ui_sat.value, RUN_VSTAR=ui_vstar.value, RUN_MMVP=ui_mmvp.value, THINKING=ui_thinking.value,
    GEMINI_MODEL=ui_gemini_model.value.strip() or "gemini-3-flash-preview", THINKING_LEVEL=ui_thinking_level.value,
    MODE=ui_mode.value, SMOKE_TEST_N=int(ui_smoke.value or 0),
)
eval_results, _error = {}, None
for _key, _run in selected_runs(RUNS, settings):
    print("=" * 100)
    try:
        eval_results[_key] = run_eval(_run, settings)
    except RuntimeError as _e:
        _error = f"{_run['bench_label']}: {_e}"
        break
mo.callout(
    mo.md(f"**Stopped.** {_error}" if _error else
          f"Finished {len(eval_results)} benchmark(s) in {sum(r['minutes'] for r in eval_results.values()):.0f} min."),
    kind="danger" if _error else "success",
)'''

MOLAB_RESULTS = '''\
_table, _warnings = results_table(eval_results)
if not eval_results:
    _view = mo.md("No results yet.")
else:
    _first = next(iter(eval_results.values()))
    _zip = zip_results(eval_results, Path(_first["work_dir"]).parent / f"{_first['run_name']}__sat_vstar_mmvp_results.zip")
    _view = mo.vstack([mo.callout(mo.md(_w), kind="danger") for _w in _warnings] + [
        mo.md(f"### {_first['model_label']}: accuracy (%)\\nSAT counts a question only if every option order is right."),
        mo.ui.table(_table, selection=None),
        mo.download(data=_zip.read_bytes(), filename=_zip.name, label="Download predictions, scores, judge replies and traces"),
    ])
_view'''


def make_molab(model_key):
    from marimo._ast.app_config import _AppConfig
    from marimo._ast.cell import CellConfig
    from marimo._ast.codegen import generate_filecontents

    m = MODELS[model_key]
    codes = [mk.CELL_MO, mk.CELL_IMPORTS, "# Edit model_kwargs to change decoding / generation settings.\n" + runs_source(model_key),
             molab_header(m), molab_ui(m), mk.CELL_ENV, sources(),
             mk.CELL_HELPERS + mc.COLAB_EXTRA_HELPERS + RUN_FN, MOLAB_RUN, MOLAB_RESULTS]
    names = ["_"] * len(codes)
    configs = [CellConfig() for _ in codes]
    for i in (3, 6, 7):
        configs[i] = CellConfig(hide_code=True)
    header = f"# {m['label']} x SAT, V* and MMVP - molab evaluation notebook (VLMEvalKit, ThinkMorph fork; Gemini judge)\n"
    return generate_filecontents(
        codes, names, configs,
        config=_AppConfig(width="medium", app_title=f"{m['label']} x SAT, V* and MMVP"),
        header_comments=header,
    )


def main():
    for model_key in MODELS:
        name, text = make_colab(model_key)
        (COLAB_OUT / name).write_text(text)
        print("wrote", COLAB_OUT / name)
        path = MOLAB_OUT / f"{model_key}__sat_vstar_mmvp.py"
        path.write_text(make_molab(model_key))
        print("wrote", path)


if __name__ == "__main__":
    main()
