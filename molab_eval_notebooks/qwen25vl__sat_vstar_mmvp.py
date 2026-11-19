# Qwen2.5-VL-7B-Instruct x SAT, V* and MMVP - molab evaluation notebook (VLMEvalKit, ThinkMorph fork; Gemini judge)

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Qwen2.5-VL-7B-Instruct x SAT, V* and MMVP",
)


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
    RUNS = {'sat': {'model_key': 'qwen25vl',
             'model_label': 'Qwen2.5-VL-7B-Instruct',
             'model_name': 'Qwen2.5-VL-7B-Instruct',
             'model_class': 'Qwen25VLEval',
             'hf_repo': 'Qwen/Qwen2.5-VL-7B-Instruct',
             'allow_patterns': ['*.json', '*.txt', '*.jinja', '*.safetensors'],
             'disk_gb': 25,
             'deps_key': 'qwen25vl-tf4.49',
             'pip': ['transformers==4.49.0',
                     'qwen-vl-utils==0.0.11',
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
             'flash_attn_check': False,
             'stub_thinkmorph': True,
             'fork': {'repo': 'hychaochao/VLMEvalKit_Thinkmorph',
                      'sha': 'cddb6a38ac5dfaf64020d6aa3964506e26006568'},
             'deltav': None,
             'extra_repos': [],
             'dataset': 'SAT_circular',
             'bench_label': 'SAT (real-image split, circular eval)',
             'bench_size': '150 questions, scored with VLMEvalKit circular evaluation',
             'min_gpu_gib': 22,
             'colab_gpus': 'L4 or A100 (Pro), or G4 / H100 (Pro+)',
             'sync_every_min': 15,
             'model_kwargs': {'model_path': '{MODEL_DIR}',
                              'min_pixels': 1003520,
                              'max_pixels': 12845056,
                              'use_custom_prompt': False,
                              'max_new_tokens': 2048,
                              'attn_implementation': 'auto'}},
     'vstar': {'model_key': 'qwen25vl',
               'model_label': 'Qwen2.5-VL-7B-Instruct',
               'model_name': 'Qwen2.5-VL-7B-Instruct',
               'model_class': 'Qwen25VLEval',
               'hf_repo': 'Qwen/Qwen2.5-VL-7B-Instruct',
               'allow_patterns': ['*.json', '*.txt', '*.jinja', '*.safetensors'],
               'disk_gb': 25,
               'deps_key': 'qwen25vl-tf4.49',
               'pip': ['transformers==4.49.0',
                       'qwen-vl-utils==0.0.11',
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
               'flash_attn_check': False,
               'stub_thinkmorph': True,
               'fork': {'repo': 'hychaochao/VLMEvalKit_Thinkmorph',
                        'sha': 'cddb6a38ac5dfaf64020d6aa3964506e26006568'},
               'deltav': None,
               'extra_repos': [],
               'dataset': 'VStarBench',
               'bench_label': 'V* Bench',
               'bench_size': '191 questions',
               'min_gpu_gib': 22,
               'colab_gpus': 'L4 or A100 (Pro), or G4 / H100 (Pro+)',
               'sync_every_min': 15,
               'model_kwargs': {'model_path': '{MODEL_DIR}',
                                'min_pixels': 1003520,
                                'max_pixels': 12845056,
                                'use_custom_prompt': False,
                                'max_new_tokens': 2048,
                                'attn_implementation': 'auto'}},
     'mmvp': {'model_key': 'qwen25vl',
              'model_label': 'Qwen2.5-VL-7B-Instruct',
              'model_name': 'Qwen2.5-VL-7B-Instruct',
              'model_class': 'Qwen25VLEval',
              'hf_repo': 'Qwen/Qwen2.5-VL-7B-Instruct',
              'allow_patterns': ['*.json', '*.txt', '*.jinja', '*.safetensors'],
              'disk_gb': 25,
              'deps_key': 'qwen25vl-tf4.49',
              'pip': ['transformers==4.49.0',
                      'qwen-vl-utils==0.0.11',
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
              'flash_attn_check': False,
              'stub_thinkmorph': True,
              'fork': {'repo': 'hychaochao/VLMEvalKit_Thinkmorph',
                       'sha': 'cddb6a38ac5dfaf64020d6aa3964506e26006568'},
              'deltav': None,
              'extra_repos': [],
              'dataset': 'MMVP',
              'bench_label': 'MMVP',
              'bench_size': '300 questions',
              'min_gpu_gib': 22,
              'colab_gpus': 'L4 or A100 (Pro), or G4 / H100 (Pro+)',
              'sync_every_min': 15,
              'model_kwargs': {'model_path': '{MODEL_DIR}',
                               'min_pixels': 1003520,
                               'max_pixels': 12845056,
                               'use_custom_prompt': False,
                               'max_new_tokens': 2048,
                               'attn_implementation': 'auto'}}}
    return (RUNS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Qwen2.5-VL-7B-Instruct × SAT, V\* and MMVP

    Qwen2.5-VL-7B-Instruct through the fork's own `Qwen2VLChat` wrapper with the settings of its `Qwen2.5-VL-7B-Instruct` entry: images resized to between 1,280 and 16,384 visual tokens (min/max_pixels 1280·28·28 and 16384·28·28), the benchmark's standard prompt (no custom prompt), greedy decoding (top_k 1), up to 2,048 new tokens. Flash-attention 2 when it is installed, otherwise PyTorch SDPA attention (the only change to the wrapper).

    **Benchmarks:** `SAT_circular` (150 questions, circular evaluation), `VStarBench` (191) and `MMVP` (300), one after
    another, in the ThinkMorph VLMEvalKit fork, the same benchmark files and prompts as the other models.
    **Weights:** [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) ·
    **Eval code:** [`hychaochao/VLMEvalKit_Thinkmorph`](https://github.com/hychaochao/VLMEvalKit_Thinkmorph) @ `cddb6a38`

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

    **Expected time:** about 30–60 minutes for all three benchmarks on an A100 or L4, plus the 16 GB download.
    """)
    return


@app.cell
def _(mo):
    ui_hf_token = mo.ui.text(kind="password", label="Hugging Face token", full_width=True)
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
    ui_thinking = mo.ui.checkbox(value=True, label="Thinking (not used by this model)", disabled=True)
    ui_mode = mo.ui.dropdown(
        options={"Inference + scoring": "all", "Scoring only (re-score saved predictions, re-run the judge)": "eval"},
        value="Inference + scoring",
        label="Mode",
    )
    ui_smoke = mo.ui.number(start=0, stop=100000, step=1, value=0, label="Smoke test: first N samples of each benchmark (0 = full)")
    ui_start = mo.ui.run_button(label="Start / resume evaluation", kind="success")
    mo.vstack([ui_hf_token, ui_repo, ui_gemini, mo.hstack([ui_gemini_model, ui_thinking_level], justify="start"),
               mo.hstack([mo.md("**Benchmarks:**"), ui_sat, ui_vstar, ui_mmvp], justify="start"),
               ui_thinking, ui_mode, ui_smoke, ui_start])
    return (
        ui_gemini,
        ui_gemini_model,
        ui_hf_token,
        ui_mmvp,
        ui_mode,
        ui_repo,
        ui_sat,
        ui_smoke,
        ui_start,
        ui_thinking,
        ui_thinking_level,
        ui_vstar,
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
    return


@app.cell(hide_code=True)
def _():
    # Helper files written to disk at run time (model wrappers, VLMEvalKit driver, HF sync, Gemini judge).

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
      * Qwen25VLEval    - the fork's Qwen2VLChat (Qwen2.5-VL-7B-Instruct entry).
      * SenseNovaU1Eval - SenseNova-U1 through the official sensenova_u1 package.

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


    # ------------------------------------------------------------------- Qwen2.5-VL


    def _flash_attn_2_usable():
        try:
            import flash_attn  # noqa: F401
            from transformers.utils import is_flash_attn_2_available
            return bool(is_flash_attn_2_available())
        except Exception:
            return False


    def patch_qwen25_vision_sdpa():
        """Memory fix for Qwen2.5-VL's vision encoder under SDPA attention (transformers 4.49).

        transformers' SDPA vision attention passes 3-D tensors (heads, patches, dim) plus a dense
        block-diagonal mask, so PyTorch can only use its "math" kernel, which materialises a float32
        heads x patches x patches matrix: about 29 GB per copy for two 1080p frames. It runs out of GPU
        memory on large or two-image inputs and is slow. VLMEvalKit's default (flash-attention 2) never
        builds that matrix. This keeps exactly the same attention but lets PyTorch use its fused kernels:
        each image (or window) block attends only to itself, computed per block when there are few
        blocks (full-attention layers) and with the same mask in 4-D form otherwise (window layers).
        """
        import torch.nn.functional as F
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as M

        cls = getattr(M, "Qwen2_5_VLVisionSdpaAttention", None)
        if cls is None or getattr(cls, "_molab_patched", False):
            return False

        def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None):
            seq_length = hidden_states.shape[0]
            q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
            if position_embeddings is None:
                emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
                cos, sin = emb.cos().float(), emb.sin().float()
            else:
                cos, sin = position_embeddings
            q, k = M.apply_rotary_pos_emb_vision(q, k, cos, sin)
            q, k, v = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))  # (1, heads, patches, dim)
            bounds = [int(b) for b in cu_seqlens.tolist()]
            if len(bounds) - 1 <= 8:
                out = torch.cat([F.scaled_dot_product_attention(q[:, :, s:e], k[:, :, s:e], v[:, :, s:e])
                                 for s, e in zip(bounds[:-1], bounds[1:])], dim=2)
            else:
                mask = torch.zeros([1, 1, seq_length, seq_length], device=q.device, dtype=torch.bool)
                for s, e in zip(bounds[:-1], bounds[1:]):
                    mask[..., s:e, s:e] = True
                out = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=0.0)
            out = out.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
            return self.proj(out)

        cls.forward = forward
        cls._molab_patched = True
        return True


    def build_qwen25vl_class():
        from vlmeval.vlm.qwen2_vl import Qwen2VLChat

        class Qwen25VLEval(Qwen2VLChat):
            """The fork's Qwen2VLChat with the settings of its "Qwen2.5-VL-7B-Instruct" entry.

            Qwen2VLChat always asks transformers for flash-attention-2, which is rarely installed on
            molab / Colab, so PyTorch SDPA attention is used when it is missing, with the vision
            encoder's SDPA attention made memory-safe (patch_qwen25_vision_sdpa).
            """

            def __init__(self, model_path, attn_implementation="auto", seed=42, **kwargs):
                from transformers import Qwen2_5_VLForConditionalGeneration as _Cls

                seed_everything(seed)
                attn = attn_implementation
                if attn == "auto":
                    attn = "flash_attention_2" if _flash_attn_2_usable() else "sdpa"
                if attn == "sdpa" and patch_qwen25_vision_sdpa():
                    print("[molab] Qwen2.5-VL vision attention: per-image SDPA (no dense attention matrix)", flush=True)
                _own = _Cls.__dict__.get("from_pretrained")  # None: inherited from PreTrainedModel
                _orig = _Cls.from_pretrained

                def _from_pretrained(*a, **k):
                    k["attn_implementation"] = attn
                    return _orig(*a, **k)

                _Cls.from_pretrained = _from_pretrained
                try:
                    super().__init__(model_path=model_path, **kwargs)
                finally:
                    if _own is None:
                        del _Cls.from_pretrained
                    else:
                        _Cls.from_pretrained = _own
                print(f"[molab] Qwen2.5-VL loaded with {attn} attention; "
                      f"GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

            def generate_inner(self, message, dataset=None):
                out = super().generate_inner(message, dataset=dataset)
                log_trace({"dataset": dataset, "images": message_paths(message),
                           "prompt": "\n".join(m["value"] for m in message if m["type"] == "text"),
                           "output": out, "prediction": out})
                return out

        return Qwen25VLEval


    # ----------------------------------------------------------------- SenseNova-U1


    def _mem_available_gib():
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
        except Exception:
            pass
        return 1e9


    class SenseNovaU1Eval(BaseModel):
        """SenseNova-U1 (NEO-unify) through the official `sensenova_u1` package.

        Image preprocessing and generation follow the repo's VQA example
        (examples/vqa/inference.py: load_image_native, model.generate). The prompt is what the
        checkpoint's chat_template.jinja renders for one user turn, which is what SenseNova's own
        VLMEvalKit evaluation sends through its server: no system block, "<image>\\n" per image,
        and, with thinking off, an empty "<think>\\n\\n</think>\\n\\n" after the assistant header.

        With thinking on, the answer is the text after </think>. An answer cut off by the token
        limit before </think> has no answer, as in SenseNova's server (its reasoning parser then
        returns empty content), so VLMEvalKit receives an empty prediction for it.
        """

        INSTALL_REQ = False
        INTERLEAVE = True
        IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"

        def __init__(self, model_path, think=True, max_new_tokens=8192, do_sample=False,
                     temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=None,
                     min_pixels=65536, max_pixels=4194304, attn_backend="auto",
                     extract_answer=True, seed=42, **kwargs):
            super().__init__()
            seed_everything(seed)
            import sensenova_u1
            from sensenova_u1.models.neo_unify.conversation import get_conv_template
            from sensenova_u1.models.neo_unify.utils import load_image_native
            from sensenova_u1.utils import load_model_and_tokenizer

            sensenova_u1.set_attn_backend(attn_backend)
            t0 = time.time()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda" and _mem_available_gib() < 48:
                # the repo's default path builds the 35 GB model in CPU memory first; with less RAM than
                # that, stream the weights straight to the GPU (the loader's device_map option)
                self.model, self.tokenizer = load_model_and_tokenizer(
                    model_path, dtype=torch.bfloat16, device=None, device_map="cuda")
            else:
                self.model, self.tokenizer = load_model_and_tokenizer(model_path, dtype=torch.bfloat16, device=self.device)
            self.model.eval()
            self._load_image = load_image_native
            tpl = get_conv_template(self.model.template)
            self.user_role, self.assistant_role = tpl.roles
            self.sep = tpl.sep
            tok = self.tokenizer
            self.model.img_context_token_id = tok.convert_tokens_to_ids(self.IMG_CONTEXT)
            self.model.img_start_token_id = tok.convert_tokens_to_ids(self.IMG_START)
            self.eos_token_id = tok.convert_tokens_to_ids(self.sep.strip())
            self.think = bool(think)
            self.max_new_tokens = int(max_new_tokens)
            self.min_pixels, self.max_pixels = int(min_pixels), int(max_pixels)
            self.extract = bool(extract_answer)
            self.gen_cfg = dict(max_new_tokens=self.max_new_tokens, do_sample=bool(do_sample),
                                eos_token_id=self.eos_token_id)
            if do_sample:
                self.gen_cfg.update(temperature=float(temperature), top_p=float(top_p))
                if top_k is not None:
                    self.gen_cfg["top_k"] = int(top_k)
            if repetition_penalty is not None:
                self.gen_cfg["repetition_penalty"] = float(repetition_penalty)
            print(f"[molab] SenseNova-U1 loaded in {time.time() - t0:.0f}s; attention "
                  f"{sensenova_u1.effective_attn_backend()}; thinking {'on' if self.think else 'off'}; "
                  f"GPU mem {torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.1f} GiB", flush=True)

        def build_query(self, message):
            """Rendered prompt (with one <image> placeholder per image) and the images, in order."""
            content, images = "", []
            for m in message:
                if m["type"] == "image":
                    images.append(m["value"])
                    content += "<image>\n"
                elif m["type"] == "text":
                    content += m["value"]
            query = self.user_role + content + self.sep + self.assistant_role
            if not self.think:
                query += "<think>\n\n</think>\n\n"
            return query, images

        @torch.inference_mode()
        def generate_inner(self, message, dataset=None):
            query, images = self.build_query(message)
            prompt = query
            model, tok = self.model, self.tokenizer
            pixel_values = grid_hw = None
            if images:
                pv, gh = [], []
                for img in images:
                    img = Image.open(img) if isinstance(img, str) else img
                    p, g = self._load_image(img, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
                    pv.append(p)
                    gh.append(g)
                pixel_values = torch.cat(pv).to(self.device, dtype=model.dtype)
                grid_hw = torch.cat(gh).to(self.device)
                for i in range(grid_hw.shape[0]):
                    n = int(grid_hw[i, 0] * grid_hw[i, 1] * model.downsample_ratio ** 2)
                    query = query.replace("<image>", self.IMG_START + self.IMG_CONTEXT * n + self.IMG_END, 1)
            enc = tok(query, return_tensors="pt")
            out_ids = model.generate(pixel_values=pixel_values, input_ids=enc["input_ids"].to(self.device),
                                     grid_hw=grid_hw, attention_mask=enc["attention_mask"].to(self.device),
                                     **dict(self.gen_cfg))
            n_new = int(out_ids.shape[-1])
            # keep <think> / </think> even if the tokenizer marks them special; cut at <|im_end|>
            raw = tok.decode(out_ids[0], skip_special_tokens=False).split(self.sep.strip())[0]
            for s in tok.all_special_tokens:
                if s not in ("<think>", "</think>"):
                    raw = raw.replace(s, "")
            raw = raw.strip()
            cut_off = False
            if "</think>" in raw:
                answer = raw.rsplit("</think>", 1)[1].strip()
            elif self.think and ("<think>" in raw or n_new >= self.max_new_tokens):
                answer, cut_off = "", True  # still reasoning when the token limit was reached
            else:
                answer = raw
            pred = finalize_prediction(answer, self.extract) if answer else ""
            log_trace({"dataset": dataset, "images": message_paths(message),
                       "prompt": prompt, "output": raw, "prediction": pred, "new_tokens": n_new, "cut_off": cut_off})
            return pred


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
        if name == "Qwen25VLEval":
            return build_qwen25vl_class()
        if name == "SenseNovaU1Eval":
            return SenseNovaU1Eval
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
        extra = ([spec["deltav_repo"]] if spec.get("deltav_repo") else []) + list(spec.get("extra_paths") or [])
        for p in [spec["code_dir"], fork] + extra:
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
                # Only predictions, partial checkpoints and scores: never the generated images.
                # A Zebra-CoT folder can hold over a thousand images, one request each, which
                # exceeds Hugging Face's limit of 1,000 requests per 5 minutes.
                patterns = [f"{prefix}/*{ext}" for ext in RESULT_EXT]
                for attempt in range(6):
                    try:
                        snapshot_download(repo, repo_type="dataset", allow_patterns=patterns,
                                          local_dir=str(tmp), token=token, max_workers=2)
                        break
                    except Exception as e:
                        if attempt == 5 or "429" not in str(e):
                            raise
                        wait = 60 * (attempt + 1)
                        print(f"[sync] Hugging Face rate limit; retrying in {wait} s", flush=True)
                        time.sleep(wait)
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

    SRC['molab_judge.py'] = r'''"""Gemini-judge repair for one finished VLMEvalKit multiple-choice run (SAT, V*, MMVP).

    Usage: python molab_judge.py judge_spec.json   (GEMINI_API_KEY in the environment)

    Same procedure as the llm_judge_repair notebook:
      1. read every saved answer with VLMEvalKit's exact-matching rules (copied verbatim from the fork),
      2. send only the answers those rules cannot read to Gemini, with VLMEvalKit's own judge prompt
         (up to 3 tries for a reply that names an option or Z),
      3. re-score: an answer read by the rules keeps that letter, otherwise the judge's letter counts;
         Z, no reply, or no answer counts as wrong (VLMEvalKit would draw a random option instead).

    Not sent to the judge: crashed answers, empty answers, answers whose reasoning was cut off by the
    token limit (<think> without a closing </think>: the model never answered), and, on circular
    benchmarks, questions already lost because another option order was read as a wrong answer.

    Writes to <work_dir>/gemini_judge/: <model>_<dataset>_judged.xlsx (every answer with the rules'
    letter, the judge's reply and the final letter), <model>_<dataset>_judge_scores.csv, and
    judge_cache.jsonl (replies are reused when the notebook runs again). With "include_cut_off": true
    in the spec, cut-off answers are judged too (from their reasoning) and the files end in _incl_cut_off.
    """
    import copy as cp
    import hashlib
    import json
    import os
    import random
    import re
    import string
    import sys
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import pandas as pd

    CRASH = "Failed to obtain answer"


    # ---- VLMEvalKit's answer reading, copied from the ThinkMorph fork (vlmeval/utils/matching_util.py,
    # ---- vlmeval/dataset/utils/multiple_choice.py, vlmeval/smp/vlm.py) so exact scores match acc.csv
    def can_infer_option(answer, choices):
        if 'Failed to obtain answer via API' in answer:
            return False
        for err in ["Sorry, I can't help with images of people yet.", "I can't process this file.",
                    "I'm sorry, but without the image provided", 'Cannot determine the answer']:
            if err in answer:
                return 'Z'

        def count_choice(splits, choices, prefix='', suffix=''):
            return sum(1 for c in choices if prefix + c + suffix in splits)
        answer_mod = cp.copy(answer)
        for c in '.()[],:;!*#{}':
            answer_mod = answer_mod.replace(c, ' ')
        splits = [x.strip() for x in answer_mod.split()]
        count = count_choice(splits, choices)
        if count == 1:
            for ch in choices:
                if ch in splits and splits.index(ch) > (len(splits) - 5):
                    return ch
        elif count == 0 and count_choice(splits, {'Z', ''}) == 1:
            return 'Z'
        return False


    def can_infer_text(answer, choices):
        answer = answer.lower()
        if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
            return False
        choices = {k: str(v).lower() for k, v in choices.items()}
        cands = [k for k in choices if choices[k] in answer]
        return cands[0] if len(cands) == 1 else False


    def can_infer(answer, choices):
        answer = str(answer)
        copt = can_infer_option(answer, choices)
        return copt if copt else can_infer_text(answer, choices)


    def build_option_str(option_dict):
        s = 'There are several options: \n'
        for c, content in option_dict.items():
            if not pd.isna(content):
                s += f'{c}. {content}\n'
        return s


    def build_prompt(question, options, prediction):
        tmpl = (
            'You are an AI assistant who will help me to match '
            'an answer with several options of a single-choice question. '
            'You are provided with a question, several options, and an answer, '
            'and you need to find which option is most similar to the answer. '
            'If the meaning of all options are significantly different from the answer, output Z. '
            'Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n'
            'Example 1: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
            'Answer: a cute teddy bear\nYour output: A\n'
            'Example 2: \n'
            'Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n'
            'Answer: Spider\nYour output: Z\n'
            'Example 3: \n'
            'Question: {}?\nOptions: {}\nAnswer: {}\nYour output: '
        )
        return tmpl.format(question, options, prediction)


    def choices_of(row):
        return {c: row[c] for c in string.ascii_uppercase if c in row and not pd.isna(row[c])}


    def read_by_rules(pred, choices):
        """VLMEvalKit's rule-based reading: the answer text, then an <answer>...</answer> or \\boxed{} span."""
        pred = str(pred)
        ret = can_infer(pred, choices)
        if ret:
            return ret
        span = None
        if "<answer>" in pred and "</answer>" in pred:
            span = pred.split("<answer>")[1].split("</answer>")[0]
        elif "boxed" in pred:
            m = re.search(r'\\boxed\{([^}]*)\}', pred)
            span = m.group(1).strip() if m else None
        return (can_infer(span, choices) or None) if span else None


    def cut_off(pred):
        """True when the reasoning was still open at the token limit (no answer was given)."""
        i = pred.rfind("<think>")
        return i >= 0 and "</think>" not in pred[i:]


    def cut_off_reasoning(work, model, ds):
        """index -> reasoning text of answers cut off by the token limit, from the run's trace file.

        Models whose cut-off answers are saved as empty predictions (SenseNova-U1) keep the reasoning
        in traces_<model>__<dataset>.jsonl. Images are saved as <index>.jpg / <index>_<k>.png, which
        gives each record's row; the last record of a row wins (a later finished retry clears it).
        """
        found = {}
        files = [work / f"traces_{model}__{ds}.jsonl"]
        files = [f for f in files if f.exists()] or sorted(work.rglob(f"traces_{model}__{ds}.jsonl"))
        for f in files:
            for line in f.open(encoding="utf-8"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("dataset") not in (None, ds) or not r.get("images"):
                    continue
                stem = Path(str(r["images"][0])).stem.split("_")[0]
                if not stem.isdigit():
                    continue
                text = str(r.get("output") or "")
                if r.get("cut_off") or cut_off(text):
                    found[int(stem)] = text
                else:
                    found.pop(int(stem), None)
        return found


    # ---- Gemini
    LOCK = threading.Lock()
    ERRORS = []  # last error of every call that kept failing


    def retryable(e):
        """Retry rate limits, server errors, timeouts and dropped connections; stop on a bad key or request."""
        if "PerDay" in str(e):  # daily quota used up: waiting a minute will not help
            return False
        code = getattr(e, "code", None)  # google.genai.errors.APIError carries the HTTP status
        if isinstance(code, int):
            return code in (408, 409, 429) or code >= 500
        return True


    def make_caller(model, thinking_level):
        from google import genai

        # preview models (gemini-3-flash-preview) are served on v1beta; google-genai >= 3.6 defaults to v1
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options={"api_version": "v1beta"})
        state = {"config": {"thinking_config": {"thinking_level": thinking_level}} if thinking_level else None}

        def call(prompt):
            """The reply text, or None when Gemini keeps failing (rate limits, server or network errors)."""
            last = None
            for attempt in range(8):
                try:
                    r = client.models.generate_content(model=model, contents=prompt, config=state["config"])
                    return (r.text or "").strip()
                except Exception as e:
                    if not retryable(e):
                        raise
                    last = e
                    time.sleep(min(60, 2 ** attempt) + random.random())
            with LOCK:
                ERRORS.append(f"{type(last).__name__}: {str(last)[:300]}")
            return None

        try:  # check the key, the model name and the thinking setting before starting
            if call("Reply with the single letter A.") is None:
                raise SystemExit(f"[judge] Gemini is not answering right now ({ERRORS[-1]}). The exact-matching "
                                 "score is saved; run the notebook again later (MODE eval) to judge.")
        except SystemExit:
            raise
        except Exception as e:
            if "thinking" in str(e).lower() and state["config"]:
                print(f"[judge] {model} does not accept thinking_level={thinking_level!r}; continuing without it.", flush=True)
                state["config"] = None
                call("Reply with the single letter A.")
            else:
                try:
                    names = [m.name.split("/")[-1] for m in client.models.list() if "flash" in m.name]
                except Exception:
                    names = []
                raise SystemExit(f"[judge] Gemini call failed: {str(e)[:300]}\n"
                                 f"Flash models available to this key: {names}")
        return call


    def score(df, col, circular):
        right = df[col] == df["answer"]
        if circular:
            return 100 * right.groupby(df["group"]).all().mean()
        return 100 * right.mean()


    def main(spec_path):
        spec = json.loads(Path(spec_path).read_text())
        work, model, ds = Path(spec["work_dir"]), spec["model_name"], spec["dataset"]
        gm, level = spec.get("gemini_model", "gemini-3-flash-preview"), spec.get("thinking_level", "low")
        xlsx = [p for p in work.rglob(f"{model}_{ds}.xlsx") if not p.is_symlink() and "gemini_judge" not in p.parts]
        if not xlsx:
            raise SystemExit(f"[judge] no saved answers ({model}_{ds}.xlsx) under {work}")
        src = max(xlsx, key=lambda p: p.stat().st_mtime)
        acc = src.with_name(f"{model}_{ds}_acc.csv")
        df = pd.read_excel(src)
        circular = "circular" in ds.lower()
        out = work / "gemini_judge"
        out.mkdir(parents=True, exist_ok=True)

        df["choices"] = [choices_of(r) for _, r in df.iterrows()]
        pred = df["prediction"].fillna("").astype(str)
        df["crashed"] = pred.str.startswith(CRASH)
        # include_cut_off: also send answers cut off by the token limit (their unfinished reasoning) to the
        # judge, as the llm_judge_repair notebook does; the judge then infers a leaning from the reasoning
        include_cut_off = bool(spec.get("include_cut_off"))
        suffix = "_incl_cut_off" if include_cut_off else ""
        text = pred.copy()
        if include_cut_off:
            recovered = cut_off_reasoning(work, model, ds)
            for i in df.index[pred.str.strip() == ""]:
                if recovered.get(int(df.at[i, "index"])):
                    text.at[i] = recovered[int(df.at[i, "index"])]
            df["no_answer"] = ~df["crashed"] & (text.str.strip() == "")
        else:
            df["no_answer"] = ~df["crashed"] & pred.map(lambda p: not p.strip() or cut_off(p))
        df["judge_text"] = text
        df["rules"] = [None if c else read_by_rules(p, ch) for p, ch, c in zip(pred, df["choices"], df["crashed"])]
        df["group"] = (df["index"] % 1_000_000) if circular else df["index"]
        rule_wrong = df["rules"].notna() & (df["rules"] != df["answer"])
        lost = df["group"].isin(df.loc[rule_wrong, "group"]) if circular else pd.Series(False, index=df.index)
        df["needs_judge"] = df["rules"].isna() & ~df["crashed"] & ~df["no_answer"] & ~lost
        df["judge"], df["judge_raw"] = None, None

        cache_file = out / "judge_cache.jsonl"
        cache = {}
        if cache_file.exists():
            for line in cache_file.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
        lock = threading.Lock()
        jobs = [(i, build_prompt(df.at[i, "question"], build_option_str(df.at[i, "choices"]), df.at[i, "judge_text"]),
                 df.at[i, "choices"]) for i in df.index[df["needs_judge"]]]
        key = lambda prompt: hashlib.sha1(f"{gm}|{prompt}".encode()).hexdigest()  # noqa: E731
        todo = [j for j in jobs if key(j[1]) not in cache]
        print(f"[judge] {model} x {ds}: {len(df)} answers, {int(df['rules'].isna().sum())} unreadable by the rules, "
              f"{len(jobs)} sent to {gm} ({len(jobs) - len(todo)} already cached), "
              f"{int(df['no_answer'].sum())} " + ("empty (not judged)" if include_cut_off else "empty or cut off (not judged)")
              + (f"; {int((df['needs_judge'] & df['judge_text'].map(cut_off)).sum())} of those sent are cut-off reasoning"
                 if include_cut_off else "") + ".", flush=True)
        call = make_caller(gm, level) if todo else None

        def judge(job):
            i, prompt, choices = job
            k = key(prompt)
            if k in cache:
                return i, cache[k]["letter"], cache[k]["raw"]
            letter, raw = None, None
            for _ in range(3):
                try:
                    raw = call(prompt)
                except Exception as e:  # one bad request must not stop the others
                    with LOCK:
                        ERRORS.append(f"{type(e).__name__}: {str(e)[:300]}")
                    raw = None
                if raw is None:
                    break
                letter = can_infer(raw, choices) or None
                if letter:
                    break
            if letter:  # only definite outcomes are cached; failures are retried next time
                with lock:
                    cache[k] = {"key": k, "letter": letter, "raw": raw}
                    with cache_file.open("a") as f:
                        f.write(json.dumps(cache[k]) + "\n")
            return i, letter, raw

        with ThreadPoolExecutor(max_workers=max(1, int(spec.get("parallel", 4)))) as pool:
            results = list(pool.map(judge, jobs))
        for i, letter, raw in results:
            df.at[i, "judge"], df.at[i, "judge_raw"] = letter, raw

        df["final"] = [r if isinstance(r, str) else (j if isinstance(j, str) else None) for r, j in zip(df["rules"], df["judge"])]
        exact, judged = score(df, "rules", circular), score(df, "final", circular)
        nj = df["needs_judge"]
        official = None
        if acc.exists():
            v = float(pd.read_csv(acc)["Overall"].iloc[0])
            official = v * 100 if v <= 1 else v
        row = {
            "model": model, "benchmark": ds,
            "exact matching": round(exact, 2),
            "official (acc.csv)": None if official is None else round(official, 2),
            "with Gemini judge": round(judged, 2), "change": round(judged - exact, 2),
            "answers": len(df),
            "unreadable answers": int(df["rules"].isna().sum()),
            "sent to judge": int(nj.sum()),
            "judge found an option": int((nj & df["judge"].isin(list(string.ascii_uppercase[:8]))).sum()),
            "of which right": int((nj & (df["judge"] == df["answer"])).sum()),
            "judge: no answer (Z)": int((nj & (df["judge"] == "Z")).sum()),
            "judge failed": int((nj & df["judge"].isna()).sum()),
            "empty or cut off (not judged)": int(df["no_answer"].sum()),
            "cut-off reasoning judged": int((nj & df["judge_text"].map(cut_off)).sum()),
            "of which right (cut off)": int((nj & df["judge_text"].map(cut_off) & (df["judge"] == df["answer"])).sum()),
            "includes cut-off answers": include_cut_off,
            "crashed answers": int(df["crashed"].sum()),
            "judge model": gm, "thinking level": level,
        }
        pd.DataFrame([row]).to_csv(out / f"{model}_{ds}_judge_scores{suffix}.csv", index=False)
        keep = [c for c in ["index", "category", "question", "A", "B", "C", "D", "E", "F", "answer", "prediction"] if c in df]
        cols = keep + (["judge_text"] if include_cut_off else []) + ["rules", "judge", "judge_raw", "final"]
        df[cols].to_excel(out / f"{model}_{ds}_judged{suffix}.xlsx", index=False)
        if official is not None and abs(official - exact) > 0.01:
            print(f"[judge] WARNING: recomputed exact-matching score {exact:.2f} differs from acc.csv ({official:.2f})", flush=True)
        if ERRORS:
            print(f"[judge] WARNING: {len(ERRORS)} Gemini call(s) kept failing and count as wrong; running the "
                  f"notebook again retries only those. First error: {ERRORS[0]}", flush=True)
        print(f"[judge] {model} x {ds}: exact matching {exact:.2f} -> with Gemini judge {judged:.2f} "
              f"({row['judge found an option']} of {row['sent to judge']} judged answers mapped to an option, "
              f"{row['of which right']} right). Saved to {out}", flush=True)


    if __name__ == "__main__":
        main(sys.argv[1])
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


@app.cell(hide_code=True)
def _(
    Path,
    SRC,
    json,
    os,
    shutil,
    subprocess,
    sys,
    threading,
    time,
    urllib,
    zipfile,
):
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
            req.write_text("\n".join(lines) + "\n")
            rc, out = py("import importlib.metadata as m; print(m.version('torch'))")
            tv = next((l.strip() for l in out.splitlines() if l.strip()[:1].isdigit()), None)
            cons = None
            if rc == 0 and tv:
                cons = root / "constraints.txt"  # keep the preinstalled CUDA build of torch
                cons.write_text(f"torch=={tv}\n")
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
                raise RuntimeError("PyTorch cannot use the GPU:\n" + out[-1500:])
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
                    fh.write(f"{time.strftime('%H:%M:%S')} copied {n} file(s) to {drive_dir}\n")
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
                        fh.write(f"sync failed: {e}\n")

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
            judge_status = "not run: no exact-matching score yet"
            log("Step 7/7: no exact-matching score yet, so the Gemini judge is skipped. Last log lines:\n" + tail(log_file))
        elif not gemini_key:
            judge_status = "not run: no GEMINI_API_KEY" + (" (scores from an earlier run)" if judge_csv.exists() else "")
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
            judge_csv.unlink(missing_ok=True)  # never show an earlier run's judge scores as this run's
            if stream_cmd([sys.executable, "-u", P["code"] / "molab_judge.py", jspec], env=jenv, log_file=log_file) == 0:
                judge_status = "done"
            else:
                judge_status = "failed: see the [judge] lines in the output above"
                log("The Gemini judge failed (see above); the exact-matching score is still valid. "
                    "Run again (MODE eval is enough) to retry.")
            if drive_dir is not None or hf_repo:
                push()
        minutes = (time.time() - t0) / 60
        log(f"{RUN['bench_label']} finished in {minutes:.0f} min.")
        return {"exit_code": exit_code, "work_dir": P["work"], "run_name": run_name, "dataset": RUN["dataset"],
                "bench_label": RUN["bench_label"], "model_label": RUN["model_label"], "score_files": score_files,
                "judge_csv": judge_csv if judge_csv.exists() else None, "judge_status": judge_status,
                "log_file": log_file, "minutes": minutes}


    def results_table(results):
        """One row per benchmark: exact matching, with the Gemini judge, and what the judge did."""
        import pandas as pd
        rows, warnings = [], []
        counts = ("unreadable answers", "sent to judge", "judge found an option", "of which right",
                  "empty or cut off (not judged)", "crashed answers")
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
                for c in counts:
                    row[c] = int(j[c])
            row["judge"] = res.get("judge_status", "")
            if row["judge"].startswith("failed"):
                warnings.append(f"{res['bench_label']}: the Gemini judge failed, so there is no judged score. "
                                "Its [judge] lines in the output above say why; run again (MODE eval) to retry.")
            rows.append(row)
        table = pd.DataFrame(rows)
        for c in counts:
            if c in table:
                table[c] = table[c].astype("Int64")
        return table, warnings


    def zip_results(results, zpath):
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for res in results.values():
                base = Path(res["work_dir"])
                for p in base.rglob("*"):
                    if p.is_file() and not p.is_symlink():
                        z.write(p, Path(base.name) / p.relative_to(base))
        return zpath


    return results_table, run_eval, selected_runs, zip_results


@app.cell
def _(
    RUNS,
    mo,
    os,
    run_eval,
    selected_runs,
    ui_gemini,
    ui_gemini_model,
    ui_hf_token,
    ui_mmvp,
    ui_mode,
    ui_repo,
    ui_sat,
    ui_smoke,
    ui_start,
    ui_thinking,
    ui_thinking_level,
    ui_vstar,
):
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
    )
    return (eval_results,)


@app.cell
def _(Path, eval_results, mo, results_table, zip_results):
    _table, _warnings = results_table(eval_results)
    if not eval_results:
        _view = mo.md("No results yet.")
    else:
        _first = next(iter(eval_results.values()))
        _zip = zip_results(eval_results, Path(_first["work_dir"]).parent / f"{_first['run_name']}__sat_vstar_mmvp_results.zip")
        _view = mo.vstack([mo.callout(mo.md(_w), kind="danger") for _w in _warnings] + [
            mo.md(f"### {_first['model_label']}: accuracy (%)\nSAT counts a question only if every option order is right."),
            mo.ui.table(_table, selection=None),
            mo.download(data=_zip.read_bytes(), filename=_zip.name, label="Download predictions, scores, judge replies and traces"),
        ])
    _view
    return


if __name__ == "__main__":
    app.run()
