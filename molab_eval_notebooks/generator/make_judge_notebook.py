"""Build colab_eval_notebooks/llm_judge_repair.ipynb: re-score saved SAT / V* / MMVP answers with a Gemini judge
for the answers VLMEvalKit's exact matching could not read (the judge step ThinkMorph's paper ran with GPT-5)."""
from pathlib import Path

import nbformat

OUT = Path(__file__).parent / "out" / "colab_eval_notebooks" / "llm_judge_repair.ipynb"

INTRO = """# LLM-judge repair for SAT, V\\* and MMVP

Re-scores the answers your evaluation runs already saved. Nothing is regenerated, and no GPU is needed.

1. Collects each model's saved answers from your Hugging Face results repo (molab runs) and Google Drive (Colab runs).
2. Reads every answer with VLMEvalKit's exact-matching rules, exactly as your scores were computed.
3. Sends only the answers those rules could not read to **Gemini 3 Flash**, with VLMEvalKit's own judge prompt
   (the same step ThinkMorph's paper ran with GPT-5). On SAT, as in VLMEvalKit, a question already lost to a wrong
   answer in the other option order is not sent.
4. Re-scores and shows exact matching vs. with-judge scores for every model and benchmark.

**Before running:** add two Colab secrets (key icon on the left): `GEMINI_API_KEY`, and `HF_TOKEN` (read access to your
results repo), and allow this notebook to use them. A CPU runtime is enough.

Gemini's replies are cached on Drive, so running again only judges answers it has not seen before.
"""

SETTINGS = '''#@title 1. Settings
HF_RESULTS_REPO = "MMert123/molab-eval-results"  #@param {type:"string"}
DRIVE_FOLDER = "umm_eval_results"  #@param {type:"string"}
GEMINI_MODEL = "gemini-3-flash-preview"  #@param {type:"string"}
THINKING_LEVEL = "low"  #@param ["minimal", "low", "medium", "high"]
PARALLEL_REQUESTS = 4  #@param {type:"integer"}

# label -> model name used in the result folders (<model>__<benchmark>)
RUNS = {
    "BAGEL think": "BAGEL-7B-MoT",
    "BAGEL no-think": "BAGEL-7B-MoT-nothink",
    "Bagel-Zebra-CoT": "Bagel-Zebra-CoT",
    "ThinkMorph": "ThinkMorph-7B",
    "DeltaV": "DeltaV-2B",
    "Qwen2.5-VL-7B": "Qwen2.5-VL-7B-Instruct",
    "SenseNova-U1": "SenseNova-U1-8B-MoT",
    "SenseNova-U1 no-think": "SenseNova-U1-8B-MoT-nothink",
}
BENCHMARKS = {"SAT": "SAT_circular", "V*": "VStarBench", "MMVP": "MMVP"}
'''

LOAD = r'''#@title 2. Load the saved answers
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "google-genai", "huggingface_hub>=0.34,<1.0", "openpyxl"], check=True)

import copy as cp, hashlib, json, random, re, string, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from google.colab import drive, userdata

def secret(name):
    try:
        return (userdata.get(name) or "").strip()
    except Exception:
        return ""

GEMINI_KEY = secret("GEMINI_API_KEY") or secret("GOOGLE_API_KEY")
HF_TOKEN = secret("HF_TOKEN")
assert GEMINI_KEY, "Add your Gemini API key as the Colab secret GEMINI_API_KEY (key icon on the left) and allow this notebook to use it."
drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive") / DRIVE_FOLDER
OUT = DRIVE / "llm_judge" / GEMINI_MODEL
OUT.mkdir(parents=True, exist_ok=True)
CRASH = "Failed to obtain answer"

# ---- VLMEvalKit's answer reading, copied from the ThinkMorph fork (vlmeval/utils/matching_util.py,
# ---- vlmeval/dataset/utils/multiple_choice.py, vlmeval/smp/vlm.py) so scores match your runs exactly
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
    """VLMEvalKit's rule-based reading: the answer text, then an <answer>...</answer> or \boxed{} span."""
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

# ---- find each run's saved answers on Drive and on Hugging Face
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=HF_TOKEN or None)

def candidates(model, ds):
    name = f"{model}_{ds}"
    found = []
    folder = DRIVE / f"{model}__{ds}"
    if folder.is_dir():
        for p in folder.rglob(f"{name}.xlsx"):
            if not p.is_symlink():
                acc = p.with_name(f"{name}_acc.csv")
                found.append(("Google Drive", p, acc if acc.exists() else None))
    if HF_RESULTS_REPO:
        try:
            files = [f.path for f in api.list_repo_tree(HF_RESULTS_REPO, path_in_repo=f"{model}__{ds}", recursive=True,
                                                        repo_type="dataset") if f.path.endswith(f"/{name}.xlsx")]
        except Exception:
            files = []
        for f in files[:1]:
            p = Path(hf_hub_download(HF_RESULTS_REPO, f, repo_type="dataset", token=HF_TOKEN or None))
            try:
                acc = Path(hf_hub_download(HF_RESULTS_REPO, f[:-5] + "_acc.csv", repo_type="dataset", token=HF_TOKEN or None))
            except Exception:
                acc = None
            found.append(("Hugging Face", p, acc))
    return found

def official_score(acc):
    if acc is None:
        return None
    v = float(pd.read_csv(acc)["Overall"].iloc[0])
    return v * 100 if v <= 1 else v

DATA = {}
rows = []
for label, model in RUNS.items():
    for bench, ds in BENCHMARKS.items():
        best = None
        for src, path, acc in candidates(model, ds):
            df = pd.read_excel(path)
            valid = int((~df["prediction"].astype(str).str.startswith(CRASH)).sum())
            if best is None or valid > best[0]:
                best = (valid, src, df, acc)
        if best is None:
            rows.append({"model": label, "benchmark": bench, "source": "not found"})
            continue
        valid, src, df, acc = best
        DATA[(label, bench)] = dict(df=df, source=src, official=official_score(acc), circular="circular" in ds.lower())
        rows.append({"model": label, "benchmark": bench, "source": src, "answers": len(df),
                     "crashed answers": len(df) - valid})
found = pd.DataFrame(rows)
for c in ("answers", "crashed answers"):
    if c in found:
        found[c] = found[c].astype("Int64")
display(found)
print(f"{len(DATA)} of {len(RUNS) * len(BENCHMARKS)} model/benchmark runs found.")
'''

JUDGE = r'''#@title 3. Repair unreadable answers with Gemini
from google import genai

client = genai.Client(api_key=GEMINI_KEY, http_options={"api_version": "v1beta"})
CONFIG = {"thinking_config": {"thinking_level": THINKING_LEVEL}}

def call_gemini(prompt):
    for attempt in range(8):
        try:
            r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=CONFIG)
            return (r.text or "").strip()
        except Exception as e:
            msg = str(e)
            if any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "DEADLINE", "INTERNAL")):
                time.sleep(min(60, 2 ** attempt) + random.random())
                continue
            raise
    return None

# check the key, the model name and the thinking setting before starting
try:
    call_gemini("Reply with the single letter A.")
except Exception as e:
    if "thinking" in str(e).lower():
        print(f"This model does not accept thinking_level={THINKING_LEVEL!r}; continuing without it.")
        CONFIG = None
        call_gemini("Reply with the single letter A.")
    else:
        names = [m.name.split("/")[-1] for m in client.models.list() if "flash" in m.name]
        raise SystemExit(f"Gemini call failed: {str(e)[:300]}\nFlash models available to your key: {names}\n"
                         "Put one of them in GEMINI_MODEL (cell 1) and run again.")

# cached replies from earlier runs of this notebook
CACHE_FILE = OUT / "judge_cache.jsonl"
cache = {}
if CACHE_FILE.exists():
    for line in CACHE_FILE.read_text().splitlines():
        rec = json.loads(line)
        cache[rec["key"]] = rec
lock = threading.Lock()

def judge(prompt, choices):
    """VLMEvalKit's judge step: up to 3 tries for a reply that names an option (or Z)."""
    key = hashlib.sha1(f"{GEMINI_MODEL}|{prompt}".encode()).hexdigest()
    if key in cache:
        return cache[key]["letter"], cache[key]["raw"], True
    letter, raw = None, None
    for _ in range(3):
        raw = call_gemini(prompt)
        if raw is None:
            break
        letter = can_infer(raw, choices) or None
        if letter:
            break
    if letter:  # only definite outcomes are cached; failures are retried next time
        with lock:
            cache[key] = {"key": key, "letter": letter, "raw": raw}
            with CACHE_FILE.open("a") as f:
                f.write(json.dumps(cache[key]) + "\n")
    return letter, raw, False

# decide which answers need the judge
jobs = []
for (label, bench), run in DATA.items():
    df = run["df"]
    df["choices"] = [choices_of(r) for _, r in df.iterrows()]
    df["crashed"] = df["prediction"].astype(str).str.startswith(CRASH)
    df["rules"] = [None if c else read_by_rules(p, ch) for p, ch, c in zip(df["prediction"], df["choices"], df["crashed"])]
    df["group"] = (df["index"] % 1_000_000) if run["circular"] else df["index"]
    rule_wrong = df["rules"].notna() & (df["rules"] != df["answer"])
    lost = df["group"].isin(df.loc[rule_wrong, "group"])  # SAT: question already wrong in another option order
    if not run["circular"]:
        lost = pd.Series(False, index=df.index)
    empty = df["prediction"].fillna("").astype(str).str.strip() == ""  # nothing to judge (counts as wrong)
    df["needs_judge"] = df["rules"].isna() & ~df["crashed"] & ~empty & ~lost
    df["judge"], df["judge_raw"] = None, None
    for i in df.index[df["needs_judge"]]:
        r = df.loc[i]
        jobs.append((label, bench, i, build_prompt(r["question"], build_option_str(r["choices"]), r["prediction"]), r["choices"]))

print(f"{len(jobs)} answers need the judge ({sum(1 for j in jobs if hashlib.sha1(f'{GEMINI_MODEL}|{j[3]}'.encode()).hexdigest() in cache)} already cached).")
done, t0 = [0], time.time()

def work(job):
    label, bench, i, prompt, choices = job
    letter, raw, _ = judge(prompt, choices)
    with lock:
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(jobs):
            print(f"  {done[0]}/{len(jobs)} judged ({time.time() - t0:.0f} s)")
    return label, bench, i, letter, raw

with ThreadPoolExecutor(max_workers=max(1, PARALLEL_REQUESTS)) as pool:
    for label, bench, i, letter, raw in pool.map(work, jobs):
        df = DATA[(label, bench)]["df"]
        df.at[i, "judge"], df.at[i, "judge_raw"] = letter, raw
print("Judging finished.")
'''

RESULTS = r'''#@title 4. Scores: exact matching vs. with the Gemini judge
def score(df, col, circular):
    right = df[col] == df["answer"]
    if circular:
        return 100 * right.groupby(df["group"]).all().mean()
    return 100 * right.mean()

detail = []
for (label, bench), run in DATA.items():
    df, circ = run["df"], run["circular"]
    # an answer read by the rules keeps that letter; otherwise the judge's letter (Z or none counts as wrong)
    df["final"] = [r if isinstance(r, str) else (j if isinstance(j, str) else None) for r, j in zip(df["rules"], df["judge"])]
    exact = score(df, "rules", circ)
    judged = score(df, "final", circ)
    nj = df["needs_judge"]
    detail.append({
        "model": label, "benchmark": bench, "source": run["source"],
        "exact matching": round(exact, 2),
        "official (acc.csv)": None if run["official"] is None else round(run["official"], 2),
        "with Gemini judge": round(judged, 2), "change": round(judged - exact, 2),
        "unreadable answers": int(df["rules"].isna().sum()),
        "sent to judge": int(nj.sum()),
        "judge found an option": int((nj & df["judge"].isin(list(string.ascii_uppercase[:8]))).sum()),
        "of which right": int((nj & (df["judge"] == df["answer"])).sum()),
        "judge: no answer (Z)": int((nj & (df["judge"] == "Z")).sum()),
        "judge failed": int((nj & df["judge"].isna()).sum()),
        "crashed answers": int(df["crashed"].sum()),
    })
    keep = [c for c in ["index", "category", "question", "A", "B", "C", "D", "answer", "prediction"] if c in df]
    df[keep + ["rules", "judge", "judge_raw", "final"]].to_excel(OUT / f"answers_{RUNS[label]}__{BENCHMARKS[bench]}.xlsx", index=False)

detail = pd.DataFrame(detail)
mismatch = detail[detail["official (acc.csv)"].notna() & ((detail["official (acc.csv)"] - detail["exact matching"]).abs() > 0.01)]
if len(mismatch):
    print("WARNING: recomputed exact-matching scores differ from the saved acc.csv for:")
    display(mismatch[["model", "benchmark", "exact matching", "official (acc.csv)"]])

order = {k: i for i, k in enumerate(RUNS)}
table = detail.pivot(index="model", columns="benchmark", values=["exact matching", "with Gemini judge"])
table = table.swaplevel(axis=1)
table = table[[(b, m) for b in BENCHMARKS for m in ("exact matching", "with Gemini judge") if (b, m) in table.columns]]
table = table.sort_index(key=lambda s: s.map(order))
print(f"Accuracy (%). Judge: {GEMINI_MODEL}, thinking level {THINKING_LEVEL}. SAT counts a question only if every option order is right.")
display(table)
display(detail.sort_values(["benchmark", "model"], key=lambda s: s.map(order) if s.name == "model" else s))
detail.to_csv(OUT / "judge_scores.csv", index=False)
table.to_csv(OUT / "judge_scores_table.csv")
print(f"Saved to Google Drive: {OUT} (scores, per-run answer files with the judge's replies, and the reply cache).")
'''


def main():
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(INTRO)] + [
        nbformat.v4.new_code_cell(src) for src in (SETTINGS, LOAD, JUDGE, RESULTS)]
    nb.metadata = {"colab": {"provenance": [], "name": "llm_judge_repair.ipynb"},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
