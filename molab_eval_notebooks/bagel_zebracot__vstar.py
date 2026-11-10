# Bagel-Zebra-CoT x V* Bench - molab evaluation notebook (VLMEvalKit, ThinkMorph fork)

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Bagel-Zebra-CoT x V* Bench")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import json
    import os
    import shutil
    import subprocess
    import sys
    import threading
    import time
    import urllib.request
    import zipfile
    from pathlib import Path

    return (
        Path,
        json,
        os,
        shutil,
        subprocess,
        sys,
        threading,
        time,
        urllib,
        zipfile,
    )


@app.cell
def _():
    # Edit model_kwargs to change decoding / generation settings.
    RUN = {'model_key': 'bagel_zebracot',
     'model_label': 'Bagel-Zebra-CoT',
     'model_name': 'Bagel-Zebra-CoT',
     'model_class': 'ZebraCoT',
     'hf_repo': 'multimodal-reasoning-lab/Bagel-Zebra-CoT',
     'allow_patterns': ['*.json', '*.txt', 'model_bf16.safetensors', 'ae.safetensors'],
     'disk_gb': 45,
     'deps_key': 'bagel-tf4.49',
     'pip': ['transformers==4.49.0',
             'accelerate>=0.34',
             'safetensors',
             'huggingface_hub>=0.34,<1.0',
             'hf_xet',
             'datasets>=3.0',
             'regex',
             'pyyaml',
             'scipy',
             'scikit-image',
             'openpyxl',
             'jsonlines',
             'rapidfuzz',
             'einops',
             'sentencepiece',
             'timm',
             'tabulate',
             'opencv-python-headless'],
     'flash_attn_check': True,
     'stub_thinkmorph': False,
     'fork': {'repo': 'hychaochao/VLMEvalKit_Thinkmorph',
              'sha': 'cddb6a38ac5dfaf64020d6aa3964506e26006568'},
     'deltav': None,
     'dataset': 'VStarBench',
     'bench_label': 'V* Bench',
     'bench_size': '191 questions',
     'about': 'Interleaved text/image reasoning with the loop from the official `infz_bf16.py` (system '
              'prompt, prompt-then-images order, CFG 4.0/2.0, 50 denoising steps). Generates a visual '
              'thought after every reasoning step until it writes "Final Answer:" (capped at 4 visual '
              'thoughts). Text sampled at temperature 0.3 (seed 42), as in the official script.',
     'references': [{'source': 'DeltaV paper (arXiv 2607.08434, Table 5)', 'value': '64.9'}],
     'sync_every_min': 15,
     'model_kwargs': {'model_path': '{MODEL_DIR}',
                      'ckpt_file': 'model_bf16.safetensors',
                      'vit_min_size': 512,
                      'max_think_token_n': 1000,
                      'max_images': 4,
                      'save_dir': '{IMG_DIR}',
                      'do_sample': True,
                      'temperature': 0.3,
                      'seed': 42,
                      'extract_answer': True,
                      'cfg_text_scale': 4.0,
                      'cfg_img_scale': 2.0,
                      'cfg_interval': [0.0, 1.0],
                      'timestep_shift': 3.0,
                      'num_timesteps': 50,
                      'cfg_renorm_min': 0.0,
                      'cfg_renorm_type': 'text_channel'}}
    return (RUN,)


@app.cell(hide_code=True)
def _(RUN, mo):
    _refs = RUN["references"]
    _ref_rows = "\n".join(f"| {r['source']} | {r['value']} |" for r in _refs) or "| (none published for this pair) | |"
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
    )
    return


@app.cell
def _(mo):
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
    mo.vstack([ui_hf_token, ui_repo, ui_openai, mo.hstack([ui_judge, ui_mode], justify="start"), ui_smoke, ui_sync_images, ui_start])
    return (
        ui_hf_token,
        ui_judge,
        ui_mode,
        ui_openai,
        ui_repo,
        ui_smoke,
        ui_start,
        ui_sync_images,
    )


@app.cell
def _(Path, mo, os, shutil, subprocess, sys):
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
            + f"  \n**Python:** {sys.version.split()[0]}  \n**Disk:** " + " · ".join(_disk)
        ),
        kind="success" if gpu_info else "warn",
    )
    return (gpu_info,)


@app.cell(hide_code=True)
def _():
    # Helper files written to disk at run time (model wrappers, VLMEvalKit driver, HF sync, flash-attn fallback).

    SRC = {}

    SRC['molab_models.py'] = r'''"""Model wrappers used by the molab evaluation notebooks.

    All four models run inside the official ThinkMorph VLMEvalKit fork
    (github.com/hychaochao/VLMEvalKit_Thinkmorph) so that every model sees the same
    benchmark files, prompts and scoring code.

      * BagelThink      - base BAGEL-7B-MoT, text-only reasoning with BAGEL's own
                          think-mode system prompt (or no-think).
      * ZebraCoT        - Bagel-Zebra-CoT, the interleaved text/image loop from
                          the official `infz_bf16.py` script.
      * ThinkMorphEval  - the fork's own ThinkMorph wrapper (interleaved mode),
                          plus answer extraction and trace logging.
      * DeltaVEval      - the official DeltaV VLMEvalKit wrapper
                          (DeltaV repo, vlmevalkit/vlm/deltav_evalkit.py).

    Every wrapper writes the full reasoning trace of each sample to the JSONL file
    named by $MOLAB_TRACE_FILE and returns only the final answer span to
    VLMEvalKit when one is marked (<answer>..</answer>, "Final Answer:", \\boxed{}).
    """
    import json
    import os
    import random
    import re
    import time
    import uuid
    from copy import deepcopy

    import numpy as np
    import torch
    from PIL import Image

    from vlmeval.vlm.base import BaseModel

    # --------------------------------------------------------------------------- utils

    BAGEL_THINK_SYSTEM_PROMPT = (
        "You should first think about the reasoning process in the mind and then provide "
        "the user with the answer. \nThe reasoning process is enclosed within <think> </think> "
        "tags, i.e. <think> reasoning process here </think> answer here"
    )

    ZEBRA_SYSTEM_PROMPT = (
        "You are an AI reasoning assistant capable of step-by-step interleaved text and visual "
        "chain of thought. Think step by step and use visual aids to enhance your problem-solving. "
        'Provide your final conclusion clearly in the format of "Final Answer: <answer here>"'
    )

    _ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
    _BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
    _FINAL = re.compile(r"final answer\s*[:\uff1a]", re.I)


    def extract_answer(text):
        """Return the final-answer span the model marked, or None."""
        if not text:
            return None
        tags = _ANSWER_TAG.findall(text)
        tags = [t for t in tags if t.strip()]
        if tags:
            return tags[-1].strip()
        hits = list(_FINAL.finditer(text))
        if hits:
            tail = text[hits[-1].end():].strip()
            tail = tail.split("<|im_end|>")[0].strip()
            first_line = tail.splitlines()[0].strip() if tail else ""
            if first_line:
                return first_line
        boxed = _BOXED.findall(text)
        if boxed:
            return boxed[-1].strip()
        return None


    def finalize_prediction(text, extract=True):
        """What VLMEvalKit receives as the prediction."""
        text = text or ""
        if not extract:
            return text
        ans = extract_answer(text)
        if ans:
            return ans
        if "</think>" in text:
            tail = text.rsplit("</think>", 1)[1]
            tail = re.sub(r"<\|?image_(start|end)\|?>", " ", tail).strip()
            if tail:
                return tail
        return text


    def log_trace(record):
        path = os.environ.get("MOLAB_TRACE_FILE")
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:  # never break evaluation because of logging
            print(f"[molab] trace write failed: {e}", flush=True)


    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


    def split_message(message):
        """VLMEvalKit message -> (list of PIL images, list of text pieces, ordered items)."""
        images, texts, ordered = [], [], []
        for m in message:
            if m["type"] == "image":
                v = m["value"]
                img = Image.open(v).convert("RGB") if isinstance(v, str) else v.convert("RGB")
                images.append(img)
                ordered.append(img)
            elif m["type"] == "text":
                texts.append(m["value"])
                ordered.append(m["value"])
        return images, texts, ordered


    def message_paths(message):
        return [m["value"] for m in message if m["type"] == "image" and isinstance(m["value"], str)]


    # ------------------------------------------------------------------- BAGEL family


    def load_bagel_inferencer(model_path, ckpt_file, vit_min_size=224):
        """Load a BAGEL-architecture checkpoint with the fork's (ThinkMorph) code."""
        from vlmeval.vlm.thinkmorph import (
            Qwen2Config, SiglipVisionConfig, Qwen2Tokenizer, load_ae, add_special_tokens,
            ImageTransform, InterleaveInferencer,
        )
        from vlmeval.vlm.ThinkMorph import _load_bagel_model

        llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        ckpt = os.path.join(model_path, ckpt_file)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        t0 = time.time()
        model = _load_bagel_model(llm_config, vit_config, vae_config, ckpt)
        vae_model = vae_model.to("cuda").to(torch.bfloat16).eval()
        print(f"[molab] loaded {ckpt} in {time.time() - t0:.0f}s; "
              f"GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

        return InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(980, vit_min_size, 14),
            new_token_ids=new_token_ids,
        )


    class _BagelBase(BaseModel):
        INSTALL_REQ = False
        INTERLEAVE = True

        def __init__(self, model_path, ckpt_file, vit_min_size=224, do_sample=False,
                     temperature=0.3, extract_answer=True, seed=42, **kwargs):
            super().__init__()
            seed_everything(seed)
            self.do_sample = bool(do_sample)
            self.temperature = float(temperature)
            self.extract = bool(extract_answer)
            self.inferencer = load_bagel_inferencer(model_path, ckpt_file, vit_min_size)
            from vlmeval.vlm.thinkmorph.data.data_utils import pil_img2rgb
            self._pil_img2rgb = pil_img2rgb

        # context helpers (mirror InterleaveInferencer.interleave_inference)
        def _resize(self, img):
            return self.inferencer.vae_transform.resize_transform(self._pil_img2rgb(img))

        def _understanding_context(self, system_prompt, items):
            inf = self.inferencer
            ctx = inf.init_gen_context()
            if system_prompt:
                ctx = inf.update_context_text(system_prompt, ctx)
            for it in items:
                if isinstance(it, str):
                    ctx = inf.update_context_text(it, ctx)
                else:
                    ctx = inf.update_context_image(self._resize(it), ctx, vae=False)
            return ctx

        def _gen_text(self, ctx, max_len):
            return self.inferencer.gen_text(
                ctx, max_length=max_len, do_sample=self.do_sample, temperature=self.temperature)


    class BagelThink(_BagelBase):
        """Base BAGEL-7B-MoT, understanding-only output.

        Input layout follows the fork's ThinkMorph wrapper (images first, then the
        question with <img><|image_k|></img> placeholders for multi-image items).
        """

        def __init__(self, model_path, ckpt_file="ema.safetensors", think=True,
                     max_new_tokens=4096, **kwargs):
            super().__init__(model_path, ckpt_file, **kwargs)
            self.think = bool(think)
            self.max_new_tokens = int(max_new_tokens)

        def generate_inner(self, message, dataset=None):
            images, texts, _ = split_message(message)
            parts = []
            k = 0
            for m in message:
                if m["type"] == "image":
                    k += 1
                    parts.append(f"<img><|image_{k}|></img>")
                else:
                    parts.append(m["value"])
            prompt = " ".join(parts)
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                ctx = self._understanding_context(
                    BAGEL_THINK_SYSTEM_PROMPT if self.think else None, images + [prompt])
                out = self._gen_text(ctx, self.max_new_tokens)
            pred = finalize_prediction(out, self.extract)
            log_trace({"dataset": dataset, "images": message_paths(message), "prompt": prompt,
                       "output": out, "prediction": pred})
            return pred


    class ZebraCoT(_BagelBase):
        """Bagel-Zebra-CoT with the interleaved loop of the official infz_bf16.py.

        Loop: understanding step -> stop if it contains "Final Answer:" / <answer>,
        otherwise generate a visual thought conditioned on everything so far,
        append it, and continue (capped at `max_images` visual thoughts).
        """

        def __init__(self, model_path, ckpt_file="model_bf16.safetensors", vit_min_size=512,
                     max_think_token_n=1000, max_images=4, save_dir=None,
                     cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=(0.0, 1.0),
                     timestep_shift=3.0, num_timesteps=50, cfg_renorm_min=0.0,
                     cfg_renorm_type="text_channel", **kwargs):
            super().__init__(model_path, ckpt_file, vit_min_size=vit_min_size, **kwargs)
            self.max_think_token_n = int(max_think_token_n)
            self.max_images = int(max_images)
            self.save_dir = save_dir
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            self.gen_kwargs = dict(
                cfg_text_scale=cfg_text_scale, cfg_img_scale=cfg_img_scale,
                cfg_interval=list(cfg_interval), timestep_shift=timestep_shift,
                num_timesteps=num_timesteps, cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
            )

        def _generate_image(self, items):
            inf = self.inferencer
            gen_ctx = inf.init_gen_context()
            cfg_text_ctx = deepcopy(gen_ctx)
            cfg_img_ctx = deepcopy(gen_ctx)
            gen_ctx = inf.update_context_text(ZEBRA_SYSTEM_PROMPT, gen_ctx)
            cfg_img_ctx = inf.update_context_text(ZEBRA_SYSTEM_PROMPT, cfg_img_ctx)
            image_shapes = (1024, 1024)
            for it in items:
                if isinstance(it, str):
                    cfg_text_ctx = deepcopy(gen_ctx)
                    gen_ctx = inf.update_context_text(it, gen_ctx)
                    cfg_img_ctx = inf.update_context_text(it, cfg_img_ctx)
                else:
                    img = self._resize(it)
                    gen_ctx = inf.update_context_image(img, gen_ctx, vae=True)
                    image_shapes = img.size[::-1]
                    cfg_text_ctx = deepcopy(gen_ctx)
            return inf.gen_image(
                image_shapes, gen_ctx,
                cfg_text_precontext=cfg_text_ctx, cfg_img_precontext=cfg_img_ctx,
                **self.gen_kwargs,
            )

        def generate_inner(self, message, dataset=None):
            images, texts, _ = split_message(message)
            prompt = "\n".join(texts)
            current = [prompt] + images  # infz_bf16.py: prompt first, then the images
            steps, saved = [], []
            n_img = 0
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                while True:
                    ctx = self._understanding_context(ZEBRA_SYSTEM_PROMPT, current)
                    out = self._gen_text(ctx, self.max_think_token_n)
                    steps.append(out)
                    if "Final Answer:" in out or "<answer>" in out:
                        break
                    if n_img >= self.max_images:
                        break
                    current = current + [out]
                    img = self._generate_image(current)
                    n_img += 1
                    if self.save_dir:
                        p = os.path.join(self.save_dir, f"zebra_{uuid.uuid4().hex[:10]}_{n_img}.jpg")
                        img.save(p)
                        saved.append(p)
                    current = current + [img]
            full = "\n".join(steps)
            pred = finalize_prediction(full, self.extract)
            log_trace({"dataset": dataset, "images": message_paths(message), "prompt": prompt,
                       "steps": steps, "generated_images": saved, "prediction": pred})
            return pred


    def build_thinkmorph_class():
        from vlmeval.vlm.ThinkMorph import ThinkMorph

        class ThinkMorphEval(ThinkMorph):
            """Fork's ThinkMorph wrapper; returns the <answer> span, logs full traces."""

            def __init__(self, extract_answer=True, **kwargs):
                self._extract = bool(extract_answer)
                super().__init__(**kwargs)
                # The fork's single-sample InterleaveInferencer.gen_text dropped the
                # `gen_context = deepcopy(gen_context)` that BAGEL's official gen_text starts with,
                # so generating text appends its tokens to the context's KV cache while kv_lens stays
                # unchanged. In interleaved mode (understanding_output=False) the next context update
                # then fails with "shape mismatch: value tensor of shape [N, 4, 128] ...". The fork's
                # batch path (used by its batch_size=8 config) copies the cache, as BAGEL does.
                inf = self.inferencer
                _orig_gen_text = inf.gen_text

                def _gen_text_on_copy(gen_context, *args, **kw):
                    return _orig_gen_text(deepcopy(gen_context), *args, **kw)

                inf.gen_text = _gen_text_on_copy

            def generate_inner(self, message, dataset=None):
                input_list = self.build_thinkmorph_input(message)
                output_list = self._run_single_inference(
                    input_list, understanding_output=self.understanding_output)
                if self.understanding_output:
                    full = output_list[0] if output_list else ""
                else:
                    full = self._format_generation_outputs(output_list)
                texts = [o for o in output_list if isinstance(o, str)]
                pred = finalize_prediction("\n".join(texts), self._extract) if self._extract else full
                log_trace({"dataset": dataset, "images": message_paths(message),
                           "prompt": input_list[-1] if input_list else "",
                           "trace": full, "prediction": pred,
                           "n_visual_thoughts": sum(1 for o in output_list if not isinstance(o, str))})
                return pred

        return ThinkMorphEval


    # ------------------------------------------------------------------------ DeltaV


    def build_deltav_class(deltav_repo):
        import importlib.util
        import sys

        if deltav_repo not in sys.path:
            sys.path.insert(0, deltav_repo)
        # DeltaV targets Python 3.10. On 3.11+ dataclasses reject an instance as a field default
        # ("mutable default ... use default_factory"); TSIMTokExtraCfg has four. Give each field a
        # default_factory instead: TSIMTokExtraCfg.load() passes all four explicitly anyway.
        import re
        _cfg = os.path.join(deltav_repo, "deltav", "tsim_tok", "modeling_tsim_tok.py")
        if os.path.exists(_cfg):
            _txt = open(_cfg).read()
            _new = re.sub(r"^(\s+\w+): (\w+Cfg) = \2\(\)\s*$", r"\1: \2 = field(default_factory=\2)",
                          _txt, flags=re.M)
            if _new != _txt:
                open(_cfg, "w").write(_new)
                print("[molab] patched DeltaV config dataclasses for Python 3.11+", flush=True)
        # DeltaV imports a compiled CUDA op for its deformable attention; use the pure-PyTorch
        # version when it is not built (it never is on molab/Colab: no matching nvcc).
        if "MultiScaleDeformableAttention" not in sys.modules and \
                importlib.util.find_spec("MultiScaleDeformableAttention") is None:
            _s = importlib.util.spec_from_file_location(
                "MultiScaleDeformableAttention", os.path.join(os.path.dirname(os.path.abspath(__file__)), "msda_shim.py"))
            _m = importlib.util.module_from_spec(_s)
            _s.loader.exec_module(_m)
            sys.modules["MultiScaleDeformableAttention"] = _m
            print("[molab] MultiScaleDeformableAttention -> PyTorch fallback", flush=True)
        path = os.path.join(deltav_repo, "vlmevalkit", "vlm", "deltav_evalkit.py")
        spec = importlib.util.spec_from_file_location("deltav_evalkit", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class DeltaVEval(mod.DeltaV):
            def __init__(self, model_path, extra_cfg_path, extract_answer=True,
                         use_tsim_router=True, seed=42, **kwargs):
                seed_everything(seed)
                self._extract = bool(extract_answer)
                if use_tsim_router:
                    # The TSIM Router allocates visual-update budgets for multi-image inputs
                    # (DINOv2 ViT-B/14 via torch.hub). Without it the official wrapper samples
                    # budgets at random, so results would not be reproducible.
                    try:
                        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", trust_repo=True)
                        kwargs.setdefault("visual_extractor_ckpt_path", "")
                        kwargs.setdefault("visual_extractor_repo_path", "")
                    except Exception as e:
                        print(f"[molab] DINOv2 unavailable ({e}); TSIM Router disabled, "
                              "multi-image budgets will be sampled with a fixed seed.", flush=True)
                        kwargs.pop("visual_extractor_ckpt_path", None)
                else:
                    kwargs.pop("visual_extractor_ckpt_path", None)
                super().__init__(model_path, extra_cfg_path, **kwargs)
                print(f"[molab] DeltaV loaded; TSIM Router "
                      f"{'on' if self.token_sampler is not None else 'off'}", flush=True)

            def generate_inner(self, message, dataset=None):
                seed_everything(1234)
                out = super().generate_inner(message, dataset=dataset)
                pred = finalize_prediction(out, self._extract)
                log_trace({"dataset": dataset, "images": message_paths(message),
                           "prompt": "".join(m["value"] for m in message if m["type"] == "text"),
                           "output": out, "prediction": pred})
                return pred

        return DeltaVEval


    # ---------------------------------------------------------------------- registry


    def get_model_class(name, spec):
        if name == "BagelThink":
            return BagelThink
        if name == "ZebraCoT":
            return ZebraCoT
        if name == "ThinkMorphEval":
            return build_thinkmorph_class()
        if name == "DeltaVEval":
            return build_deltav_class(spec["deltav_repo"])
        raise ValueError(f"unknown model class {name}")
    '''

    SRC['molab_driver.py'] = r'''"""Launch one VLMEvalKit (ThinkMorph fork) run from a JSON spec.

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
    '''

    SRC['molab_sync.py'] = r'''"""Checkpoint VLMEvalKit outputs to / from a private Hugging Face dataset repo.

    Usage: python molab_sync.py {push,pull} spec.json      (HF_TOKEN must be set)

    push: uploads new or changed result files (predictions, partial .pkl
          checkpoints, scores, traces, logs) under <repo>/<sync_prefix>/.
    pull: copies files from <repo>/<sync_prefix>/ into the local work dir when the
          local copy is missing, so an interrupted run resumes where it stopped.
    """
    import json
    import os
    import shutil
    import sys
    import tempfile
    import time
    from pathlib import Path

    RESULT_EXT = (".pkl", ".xlsx", ".csv", ".json", ".jsonl", ".tsv", ".log", ".txt")
    IMAGE_EXT = (".jpg", ".jpeg", ".png")
    STATE = ".molab_sync_state.json"


    def main(action, spec_path):
        from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download

        spec = json.loads(Path(spec_path).read_text())
        repo = spec["hf_repo"]
        prefix = spec["sync_prefix"]
        local = Path(spec["work_dir"])
        local.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN") or None
        api = HfApi(token=token)
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)

        if action == "pull":
            tmp = Path(tempfile.mkdtemp(prefix="molab_pull_"))
            try:
                snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{prefix}/**"],
                                  local_dir=str(tmp), token=token)
                src = tmp / prefix
                copied = 0
                if src.exists():
                    for p in src.rglob("*"):
                        if p.is_file():
                            dst = local / p.relative_to(src)
                            if not dst.exists():
                                dst.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(p, dst)
                                copied += 1
                print(f"[sync] restored {copied} file(s) from {repo}/{prefix}", flush=True)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            return

        exts = RESULT_EXT + (IMAGE_EXT if spec.get("sync_images") else ())
        state_path = local / STATE
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        ops, new_state = [], dict(state)
        for p in local.rglob("*"):
            if p.is_symlink() or not p.is_file() or p.name == STATE:
                continue
            if not p.name.lower().endswith(exts):
                continue
            rel = p.relative_to(local).as_posix()
            st = p.stat()
            sig = [int(st.st_mtime), st.st_size]
            if state.get(rel) == sig:
                continue
            ops.append(CommitOperationAdd(path_in_repo=f"{prefix}/{rel}", path_or_fileobj=str(p)))
            new_state[rel] = sig
        if not ops:
            print("[sync] nothing new to upload", flush=True)
            return
        for attempt in range(3):
            try:
                api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                                  commit_message=f"{prefix}: {len(ops)} file(s) @ {time.strftime('%Y-%m-%d %H:%M:%S')}")
                state_path.write_text(json.dumps(new_state))
                print(f"[sync] uploaded {len(ops)} file(s) to {repo}/{prefix}", flush=True)
                return
            except Exception as e:  # rate limits / transient network errors
                print(f"[sync] upload attempt {attempt + 1} failed: {e}", flush=True)
                time.sleep(30 * (attempt + 1))


    if __name__ == "__main__":
        main(sys.argv[1], sys.argv[2])
    '''

    SRC['flash_attn_shim.py'] = r'''"""Pure-PyTorch stand-in for `flash_attn.flash_attn_varlen_func`.

    Only used when the real flash-attn package cannot be imported (prebuilt
    flash-attn wheels rarely match the RTX PRO 6000 / sm_120 + newest torch combo
    that molab ships). Semantics follow flash-attn >= 2.1:
      * q: (total_q, n_heads, d), k/v: (total_k, n_kv_heads, d)  (GQA supported)
      * causal masks are bottom-right aligned (query i of a length-Lq segment may
        attend to keys j <= i + (Lk - Lq)), which is what KV-cache decoding needs.
    """
    import torch
    import torch.nn.functional as F

    __version__ = "2.7.0-sdpa-shim"


    def flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q=None, max_seqlen_k=None,
        dropout_p=0.0, softmax_scale=None, causal=False,
        window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
        deterministic=False, return_attn_probs=False, block_table=None, **kwargs,
    ):
        n_heads, n_kv = q.shape[1], k.shape[1]
        if n_heads != n_kv:
            rep = n_heads // n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        cq = cu_seqlens_q.tolist()
        ck = cu_seqlens_k.tolist()
        out = torch.empty_like(q)
        for i in range(len(cq) - 1):
            qs, qe, ks, ke = cq[i], cq[i + 1], ck[i], ck[i + 1]
            lq, lk = qe - qs, ke - ks
            if lq == 0:
                continue
            qi = q[qs:qe].transpose(0, 1).unsqueeze(0)  # 1,H,Lq,D
            ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
            vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
            mask, is_causal = None, False
            if causal and lq > 1:
                if lq == lk:
                    is_causal = True
                else:
                    rows = torch.arange(lq, device=q.device).unsqueeze(1)
                    cols = torch.arange(lk, device=q.device).unsqueeze(0)
                    mask = cols <= rows + (lk - lq)
            oi = F.scaled_dot_product_attention(
                qi, ki, vi, attn_mask=mask, dropout_p=0.0,
                is_causal=is_causal, scale=softmax_scale,
            )
            out[qs:qe] = oi.squeeze(0).transpose(0, 1).to(out.dtype)
        return out


    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        """(B, L, H, D) layout wrapper, provided for completeness."""
        b, lq, lk = q.shape[0], q.shape[1], k.shape[1]
        cu_q = torch.arange(0, (b + 1) * lq, lq, device=q.device, dtype=torch.int32)
        cu_k = torch.arange(0, (b + 1) * lk, lk, device=q.device, dtype=torch.int32)
        o = flash_attn_varlen_func(
            q.reshape(b * lq, *q.shape[2:]), k.reshape(b * lk, *k.shape[2:]),
            v.reshape(b * lk, *v.shape[2:]), cu_q, cu_k, lq, lk,
            softmax_scale=softmax_scale, causal=causal,
        )
        return o.reshape(q.shape)
    '''

    SRC['msda_shim.py'] = r'''"""Pure-PyTorch stand-in for DeltaV's compiled `MultiScaleDeformableAttention` CUDA op.

    DeltaV's INSTALL.md calls the compiled op optional ("a pure-PyTorch path is used
    otherwise"), but ms_deform_attn_func.py imports it unconditionally. This module provides
    the forward pass with the same maths as the CUDA kernel (bilinear sampling at
    loc * size - 0.5, zero outside the map, attention-weighted sum), computed in float32
    like the kernel, which only supports float/double. Inference only: no backward.
    """
    import torch
    import torch.nn.functional as F

    __version__ = "pytorch-fallback"


    def ms_deform_attn_forward(value, spatial_shapes, level_start_index, sampling_locations,
                               attention_weights, im2col_step=64):
        """value (N, S, M, D); sampling_locations (N, Lq, M, L, P, 2) as (x, y) in [0, 1];
        attention_weights (N, Lq, M, L, P). Returns (N, Lq, M * D)."""
        out_dtype = value.dtype
        value = value.float()
        sampling_locations = sampling_locations.float()
        attention_weights = attention_weights.float()
        n, _, m, d = value.shape
        _, lq, _, n_levels, n_points, _ = sampling_locations.shape
        shapes = [(int(h), int(w)) for h, w in spatial_shapes.tolist()]
        starts = [int(s) for s in level_start_index.tolist()]
        grids = 2 * sampling_locations - 1  # grid_sample(align_corners=False): pixel = loc * size - 0.5
        sampled = []
        for lid in range(n_levels):
            h, w = shapes[lid]
            v = value[:, starts[lid]:starts[lid] + h * w]                      # N, HW, M, D
            v = v.flatten(2).transpose(1, 2).reshape(n * m, d, h, w)          # N*M, D, H, W
            g = grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)             # N*M, Lq, P, 2
            sampled.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
        weights = attention_weights.transpose(1, 2).reshape(n * m, 1, lq, n_levels * n_points)
        out = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1).view(n, m * d, lq)
        return out.transpose(1, 2).contiguous().to(out_dtype)


    def ms_deform_attn_backward(*args, **kwargs):
        raise NotImplementedError("PyTorch fallback for MultiScaleDeformableAttention is inference-only")
    '''
    return (SRC,)


@app.cell
def _(Path, os, shutil, subprocess, sys, time, urllib, zipfile):
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
            return ""

    return (
        fetch_github,
        fill,
        find_scores,
        log,
        pick_root,
        pip_install,
        py,
        stream_cmd,
        tail,
    )


@app.cell
def _(
    RUN,
    SRC,
    fetch_github,
    fill,
    find_scores,
    gpu_info,
    json,
    log,
    mo,
    os,
    pick_root,
    pip_install,
    py,
    stream_cmd,
    subprocess,
    sys,
    tail,
    threading,
    time,
    ui_hf_token,
    ui_judge,
    ui_mode,
    ui_openai,
    ui_repo,
    ui_smoke,
    ui_start,
    ui_sync_images,
):
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
        _req.write_text("\n".join(_lines) + "\n")
        _rc, _out = py("import importlib.metadata as m; print(m.version('torch'))")
        _cons = None
        _tv = next((_l.strip() for _l in _out.splitlines() if _l.strip()[:1].isdigit()), None)
        if _rc == 0 and _tv:
            # keep molab's preinstalled CUDA build of torch
            _cons = root / "constraints.txt"
            _cons.write_text(f"torch=={_tv}\n")
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
        mo.stop(_rc != 0 or "CUDA_OK" not in _out, mo.callout(mo.md("PyTorch cannot use the GPU:\n```\n" + _out[-1500:] + "\n```"), kind="danger"))

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
            else "No score file was produced. The last log lines:\n```\n" + tail(log_file) + "\n```"
        ),
        kind="success" if score_files else "danger",
    )
    return (run_result,)


@app.cell
def _(Path, RUN, mo, run_result, zipfile):
    import io
    import pandas as pd

    _files = run_result["score_files"]
    _crash = None
    for _x in Path(run_result["work_dir"]).rglob(f"{run_result['run_name']}_{RUN['dataset']}.xlsx"):
        if _x.is_symlink():
            continue
        _pred = pd.read_excel(_x)["prediction"].astype(str)
        _bad = _pred[_pred.str.startswith("Failed to obtain answer")]
        if len(_bad):
            _crash = mo.callout(mo.md(
                f"**{len(_bad)} of {len(_pred)} answers are model errors, not answers, so this score is not valid.** "
                "Running the notebook again retries them automatically.\n\n"
                f"First error:\n```\n{_bad.iloc[0][:400]}\n```"), kind="danger")
    if not _files:
        _view = _crash or mo.md("No scores yet.")
    else:
        _df = pd.read_csv(_files[-1])
        _buf = io.BytesIO()
        with zipfile.ZipFile(_buf, "w", zipfile.ZIP_DEFLATED) as _z:
            for _p in Path(run_result["work_dir"]).rglob("*"):
                if _p.is_file() and not _p.is_symlink() and "visual_thoughts" not in _p.parts:
                    _z.write(_p, _p.relative_to(run_result["work_dir"]))
        _view = mo.vstack(([_crash] if _crash else []) + [
            mo.md(f"### {RUN['model_label']} on {RUN['bench_label']} (accuracy, %)"),
            mo.ui.table(_df, selection=None),
            mo.md("Overall accuracy is the `Overall` column. Full reasoning traces are in the `traces_*.jsonl` file inside the zip."),
            mo.download(data=_buf.getvalue(), filename=f"{run_result['run_name']}__{RUN['dataset']}_results.zip", label="Download predictions, scores and traces"),
        ])
    _view
    return


if __name__ == "__main__":
    app.run()
