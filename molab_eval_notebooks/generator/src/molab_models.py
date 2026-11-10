"""Model wrappers used by the molab evaluation notebooks.

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
