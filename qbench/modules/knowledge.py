"""General knowledge / reasoning module: ARC-Easy + TruthfulQA MC1, scored via loglikelihood-of-each-choice
(standard small-model eval technique -- no generation/parsing ambiguity, just "which choice does the model
assign the highest probability to"). Small samples (default 200 each) so a full run stays in the minutes
range on one GPU, matching the point of benchmarking SMALL models."""
import random
from datasets import load_dataset


def _mc_accuracy(model, examples, question_key, choices_fn, answer_idx_fn):
    correct = 0
    fails = []
    for ex in examples:
        q = ex[question_key]
        choices = choices_fn(ex)
        context = f"Question: {q}\nAnswer:"
        scores = [model.loglikelihood(context, " " + c) for c in choices]
        pred = max(range(len(scores)), key=lambda i: scores[i])
        gold = answer_idx_fn(ex)
        ok = pred == gold
        correct += ok
        if not ok and len(fails) < 5:
            fails.append({"q": q, "choices": choices, "pred": choices[pred], "gold": choices[gold]})
    return correct / len(examples), fails


def _load_arc_easy(n, seed):
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def _load_truthfulqa_mc1(n, seed):
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    idx = list(range(len(ds))); random.Random(seed).shuffle(idx)
    return [ds[i] for i in idx[:n]]


def run(model, n_per_task=200, seed=0):
    arc = _load_arc_easy(n_per_task, seed)
    arc_acc, arc_fails = _mc_accuracy(
        model, arc, "question",
        choices_fn=lambda ex: ex["choices"]["text"],
        answer_idx_fn=lambda ex: ex["choices"]["label"].index(ex["answerKey"]),
    )

    tqa = _load_truthfulqa_mc1(n_per_task, seed)
    tqa_acc, tqa_fails = _mc_accuracy(
        model, tqa, "question",
        choices_fn=lambda ex: ex["mc1_targets"]["choices"],
        answer_idx_fn=lambda ex: ex["mc1_targets"]["labels"].index(1),
    )

    return {
        "arc_easy_accuracy": arc_acc,
        "truthfulqa_mc1_accuracy": tqa_acc,
        "combined_knowledge_score": (arc_acc + tqa_acc) / 2,
        "arc_easy_fails": arc_fails,
        "truthfulqa_fails": tqa_fails,
        "n_per_task": n_per_task,
    }
