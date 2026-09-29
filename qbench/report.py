"""Renders a QBench results.json into a short Markdown scorecard."""


def render(model_id, results):
    lines = [f"# QBench report: {model_id}", ""]
    if "agentic" in results:
        a = results["agentic"]
        lines += [
            "## Agentic / tool-use",
            f"- Single-turn accuracy: **{a['single_turn_score']*100:.1f}%** (n={len(a['single_turn_results'])})",
            f"- Multi-turn accuracy (two 8-turn scripts, natural phrasing): **{a['multi_turn_score']*100:.1f}%** (n={a['multi_turn_n']})",
            f"- Repetition collapses: **{a['repetition_collapses']}/{a['multi_turn_n']}**",
            "",
        ]
    if "knowledge" in results:
        k = results["knowledge"]
        lines += [
            "## General knowledge / reasoning",
            f"- ARC-Easy accuracy: **{k['arc_easy_accuracy']*100:.1f}%** (n={k['n_per_task']})",
            f"- TruthfulQA MC1 accuracy: **{k['truthfulqa_mc1_accuracy']*100:.1f}%** (n={k['n_per_task']})",
            f"- Combined knowledge score: **{k['combined_knowledge_score']*100:.1f}%**",
            "",
        ]
    return "\n".join(lines)
