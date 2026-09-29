"""QBench CLI. usage: python cli.py run <hf-model-id> [--modules agentic,knowledge] [--submit]
--submit uploads the result straight to the shared q-project/qbench-results leaderboard dataset (needs an
HF token with write access: `hf auth login` first)."""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qbench import loading, report
from qbench.modules import MODULE_REGISTRY

RESULTS_REPO = "q-project/qbench-results"


def submit(row):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    try:
        path = hf_hub_download(RESULTS_REPO, "results.json", repo_type="dataset")
        rows = json.load(open(path))
    except Exception:
        rows = []
    rows = [r for r in rows if r["model_id"] != row["model_id"]]
    rows.append(row)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(rows, f, indent=2)
        tmp_path = f.name
    api.upload_file(path_or_fileobj=tmp_path, path_in_repo="results.json", repo_id=RESULTS_REPO,
                     repo_type="dataset", commit_message=f"Add {row['model_id']} (submitted locally)")
    os.unlink(tmp_path)
    print(f"submitted to https://huggingface.co/datasets/{RESULTS_REPO}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("model_id")
    r.add_argument("--modules", default="agentic,knowledge")
    r.add_argument("--out", default=None)
    r.add_argument("--n-per-task", type=int, default=200)
    r.add_argument("--trust-remote-code", action="store_true")
    r.add_argument("--subfolder", default=None)
    r.add_argument("--submit", action="store_true", help="upload the result to the shared leaderboard")
    r.add_argument("--submitted-by", default=None, help="name/handle shown on the leaderboard")
    a = p.parse_args()

    if a.cmd == "run":
        modules = a.modules.split(",")
        model = loading.load(a.model_id, trust_remote_code=a.trust_remote_code, subfolder=a.subfolder)
        print("computing model fingerprint...", file=sys.stderr)
        fingerprint = model.fingerprint()
        results = {}
        for name in modules:
            if name not in MODULE_REGISTRY:
                print(f"unknown module: {name} (available: {list(MODULE_REGISTRY)})", file=sys.stderr)
                continue
            print(f"running module: {name}...", file=sys.stderr)
            kwargs = {"n_per_task": a.n_per_task} if name == "knowledge" else {}
            results[name] = MODULE_REGISTRY[name](model, **kwargs)

        out_path = a.out or f"results/{a.model_id.replace('/', '__')}.json"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump({"model_id": a.model_id, "fingerprint": fingerprint, "results": results},
                   open(out_path, "w"), indent=2)
        print(f"wrote {out_path}", file=sys.stderr)
        print(f"fingerprint: {fingerprint}", file=sys.stderr)
        print(report.render(a.model_id, results))

        if a.submit:
            row = {"model_id": a.model_id, "fingerprint": fingerprint,
                   "submitted_by": a.submitted_by or "anonymous",
                   "submitted_at": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
            if "agentic" in results:
                ag = results["agentic"]
                row.update(single_turn_score=ag["single_turn_score"], multi_turn_score=ag["multi_turn_score"],
                           repetition_collapses=ag["repetition_collapses"])
            if "knowledge" in results:
                kn = results["knowledge"]
                row.update(arc_easy_accuracy=kn["arc_easy_accuracy"],
                           truthfulqa_mc1_accuracy=kn["truthfulqa_mc1_accuracy"],
                           combined_knowledge_score=kn["combined_knowledge_score"])
            submit(row)


if __name__ == "__main__":
    main()
