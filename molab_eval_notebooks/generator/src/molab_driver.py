"""Launch one VLMEvalKit (ThinkMorph fork) run from a JSON spec.

Usage: python molab_driver.py spec.json

Registers the requested model wrapper in `vlmeval.config.supported_VLM`, pins
VLMEvalKit's output folder name so that a run interrupted on one day resumes on
another, optionally truncates the benchmark for a smoke test, then executes the
fork's own run.py.
"""
import json
import os
import pickle
import runpy
import sys
import types
from functools import partial

# With SKIP_ERR=1, VLMEvalKit stores an exception as the answer text
# ("Failed to obtain answer: <class 'RuntimeError'> ..."). On --reuse it only drops
# "Failed to obtain answer via API.", so crashed answers would be kept and re-scored.
CRASH_PREFIX = "Failed to obtain answer"


def _is_crash(v):
    return isinstance(v, str) and v.startswith(CRASH_PREFIX)


def scrub_crashed_answers(work_dir, model_name, dataset):
    """Remove crashed answers from saved predictions so the run retries them, and delete
    this run's score files so they are recomputed. Returns the number of crashed answers."""
    import pandas as pd

    root = os.path.join(work_dir, model_name)
    if not os.path.isdir(root):
        return 0
    crashed = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            if os.path.islink(p) or dataset not in fn or "_result" in fn:
                continue
            if fn == f"{model_name}_{dataset}.xlsx":
                df = pd.read_excel(p)
                if "prediction" in df:
                    bad = df["prediction"].astype(str).str.startswith(CRASH_PREFIX)
                    if bad.any():
                        crashed = max(crashed, int(bad.sum()))
                        df[~bad].to_excel(p, index=False)
            elif fn.endswith(".pkl"):
                try:
                    with open(p, "rb") as f:
                        obj = pickle.load(f)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    keep = {k: v for k, v in obj.items() if not _is_crash(v)}
                    if len(keep) != len(obj):
                        crashed = max(crashed, len(obj) - len(keep))
                        with open(p, "wb") as f:
                            pickle.dump(keep, f)
    if crashed:
        prefix = f"{model_name}_{dataset}_"
        for dirpath, _, files in os.walk(root):
            for fn in files:
                if fn.startswith(prefix) and (fn.endswith("acc.csv") or fn.endswith("score.csv")
                                              or "_result." in fn):
                    os.remove(os.path.join(dirpath, fn))
        print(f"[molab] removed {crashed} crashed answer(s) from saved predictions; "
              "they will be generated again and the run re-scored", flush=True)
    return crashed


def report_crashed_answers(work_dir, model_name, dataset):
    import pandas as pd

    for dirpath, _, files in os.walk(os.path.join(work_dir, model_name)):
        p = os.path.join(dirpath, f"{model_name}_{dataset}.xlsx")
        if f"{model_name}_{dataset}.xlsx" in files and not os.path.islink(p):
            pred = pd.read_excel(p)["prediction"].astype(str)
            bad = pred[pred.str.startswith(CRASH_PREFIX)]
            if len(bad):
                print(f"[molab] WARNING: {len(bad)} of {len(pred)} answers are model errors, not answers; "
                      f"the score is not valid. First error: {bad.iloc[0][:300]}", flush=True)


def main(spec_path):
    with open(spec_path) as f:
        spec = json.load(f)

    fork = spec["fork_dir"]
    for p in [spec["code_dir"], fork] + ([spec["deltav_repo"]] if spec.get("deltav_repo") else []):
        if p not in sys.path:
            sys.path.insert(0, p)

    shim = spec.get("flash_attn_shim")
    if shim:
        # Force the PyTorch-SDPA stand-in, even if a broken or partial flash_attn
        # (e.g. a namespace-only directory) sits in site-packages.
        import importlib.util

        for k in [k for k in sys.modules if k == "flash_attn" or k.startswith("flash_attn.")]:
            del sys.modules[k]
        shim_root = os.path.dirname(os.path.dirname(shim))
        if shim_root not in sys.path:
            sys.path.insert(0, shim_root)
        fa_spec = importlib.util.spec_from_file_location(
            "flash_attn", shim, submodule_search_locations=[os.path.dirname(shim)])
        fa_mod = importlib.util.module_from_spec(fa_spec)
        sys.modules["flash_attn"] = fa_mod
        fa_spec.loader.exec_module(fa_mod)
        # transformers must not route anything through its own flash-attn-2 code path
        import transformers.utils as tf_utils
        import transformers.utils.import_utils as tf_import_utils

        def _no_fa2(*args, **kwargs):
            return False

        tf_import_utils.is_flash_attn_2_available = _no_fa2
        tf_utils.is_flash_attn_2_available = _no_fa2
        print(f"[molab] flash_attn -> PyTorch SDPA fallback ({fa_mod.__version__})", flush=True)

    if spec.get("stub_thinkmorph"):
        # DeltaV runs on transformers 4.57; the fork's BAGEL/ThinkMorph modeling code
        # targets 4.49, so keep it from being imported at all.
        stub = types.ModuleType("vlmeval.vlm.ThinkMorph")

        class ThinkMorph:  # placeholder; vlmeval.config wraps it in functools.partial
            def __init__(self, *a, **k):
                raise RuntimeError("ThinkMorph is not available in the DeltaV environment")

        stub.ThinkMorph = ThinkMorph
        sys.modules["vlmeval.vlm.ThinkMorph"] = stub

    import vlmeval.smp as smp
    import vlmeval.smp.misc as smp_misc

    original_timestr = smp_misc.timestr
    eval_day = spec.get("eval_day", "molab")

    def pinned_timestr(granularity="second"):
        if granularity == "day":
            return eval_day
        return original_timestr(granularity)

    def pinned_githash(fallback="unknown", digits=8):
        return spec.get("eval_hash", "pinned")

    # run.py names its output folder T<day>_G<githash>; pinning both keeps the
    # partial-prediction checkpoint in the same folder across sessions and days.
    smp_misc.timestr = pinned_timestr
    smp.timestr = pinned_timestr
    smp_misc.githash = pinned_githash
    smp.githash = pinned_githash

    smoke_n = int(spec.get("smoke_test_n") or 0)
    if smoke_n > 0:
        import vlmeval.dataset as vds

        original_build = vds.build_dataset

        def build_truncated(name, **kw):
            ds = original_build(name, **kw)
            if ds is not None:
                ds.data = ds.data.iloc[:smoke_n].reset_index(drop=True)
                print(f"[molab] smoke test: {name} truncated to {len(ds.data)} rows", flush=True)
            return ds

        vds.build_dataset = build_truncated

    import vlmeval.config as cfg
    import molab_models

    cls = molab_models.get_model_class(spec["model_class"], spec)
    cfg.supported_VLM[spec["model_name"]] = partial(cls, **spec["model_kwargs"])
    print(f"[molab] registered {spec['model_name']} -> {cls.__name__}", flush=True)

    argv = [
        "run.py",
        "--data", spec["dataset"],
        "--model", spec["model_name"],
        "--work-dir", spec["work_dir"],
        "--judge", spec["judge"],
        "--api-nproc", str(spec.get("api_nproc", 4)),
        "--mode", spec.get("mode", "all"),
        "--reuse",
    ]
    if spec.get("verbose"):
        argv.append("--verbose")
    scrub_crashed_answers(spec["work_dir"], spec["model_name"], spec["dataset"])
    print("[molab] " + " ".join(argv), flush=True)
    sys.argv = argv
    os.chdir(fork)
    try:
        runpy.run_path(os.path.join(fork, "run.py"), run_name="__main__")
    finally:
        report_crashed_answers(spec["work_dir"], spec["model_name"], spec["dataset"])


if __name__ == "__main__":
    main(sys.argv[1])
