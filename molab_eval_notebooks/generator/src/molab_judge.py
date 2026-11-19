"""Gemini-judge repair for one finished VLMEvalKit multiple-choice run (SAT, V*, MMVP).

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
