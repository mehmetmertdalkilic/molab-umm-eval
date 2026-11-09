"""Launch one VLMEvalKit (ThinkMorph fork) run from a JSON spec.

Usage: python molab_driver.py spec.json

Registers the requested model wrapper in `vlmeval.config.supported_VLM`, pins
VLMEvalKit's output folder name so that a run interrupted on one day resumes on
another, optionally truncates the benchmark for a smoke test, then executes the
fork's own run.py.
"""
import json
import os
import runpy
import sys
import types
from functools import partial


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
    print("[molab] " + " ".join(argv), flush=True)
    sys.argv = argv
    os.chdir(fork)
    runpy.run_path(os.path.join(fork, "run.py"), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1])
