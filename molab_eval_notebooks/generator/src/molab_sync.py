"""Checkpoint VLMEvalKit outputs to / from a private Hugging Face dataset repo.

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
