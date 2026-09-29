"""Agentic / tool-use module: can the model get the right final answer on tasks needing a calculation, and
does it stay reliable across a natural multi-turn conversation?

v1 of this module copied its phrasing almost verbatim from Q-Agent's own training-data generator
(circuit_corpus.py) and its multi-turn connectives from Q-Agent's own v5 training augmentation
(multiturn_corpus.py's CONNECTIVES list) -- a real benchmark-contamination bug caught by a user who didn't
believe Q-U-164M could legitimately beat SmolLM2-360M. Every question template and connective here is
deliberately phrased from scratch, checked against both of those files, so no model in this family has an
unfair in-distribution advantage from wording alone. Numbers are drawn from a seeded RNG for a larger, more
statistically stable sample instead of a handful of hand-picked values."""
import random
import re

RNG = random.Random(20260929)

PRIMES_UNDER_100 = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}


def _is_prime(n):
    return n in PRIMES_UNDER_100 if n < 100 else all(n % d for d in range(2, int(n ** 0.5) + 1)) and n > 1


def _fmt_list(xs):
    return ",".join(str(x) for x in xs)


def _gen_single_turn(n):
    items = []
    for _ in range(n):
        kind = RNG.choice(["add", "sub", "mul", "pct", "prime", "sort", "div"])
        if kind == "add":
            a, b = RNG.randint(20, 900), RNG.randint(20, 900)
            q = RNG.choice([f"Add {a} and {b} for me.", f"If you combine {a} and {b}, what do you get?"])
            items.append((q, str(a + b)))
        elif kind == "sub":
            a = RNG.randint(100, 999); b = RNG.randint(10, a - 1)
            q = RNG.choice([f"Take {b} away from {a} — what's left?", f"{a} minus {b} equals what?"])
            items.append((q, str(a - b)))
        elif kind == "mul":
            a, b = RNG.randint(11, 99), RNG.randint(11, 99)
            q = RNG.choice([f"Multiply {a} by {b}.", f"What do you get if you multiply {a} and {b}?"])
            items.append((q, str(a * b)))
        elif kind == "pct":
            a = RNG.randint(40, 900); p = RNG.choice([5, 10, 20, 25, 50])
            q = RNG.choice([f"A ${a} purchase includes a {p}% service charge. How much is the charge?",
                             f"What's {p}% of {a}?"])
            items.append((q, str(round(a * p / 100, 2)).rstrip("0").rstrip(".")))
        elif kind == "prime":
            n_ = RNG.randint(30, 99)
            q = RNG.choice([f"Would you classify {n_} as a prime number?", f"Is {n_} prime?"])
            items.append((q, "yes" if _is_prime(n_) else "no"))
        elif kind == "sort":
            xs = RNG.sample(range(1, 100), 4)
            q = RNG.choice([f"Please arrange these from least to greatest: {_fmt_list(xs)}",
                             f"Put these in ascending order: {_fmt_list(xs)}"])
            items.append((q, _fmt_list(sorted(xs))))
        elif kind == "div":
            b = RNG.randint(2, 12); a = b * RNG.randint(5, 90)
            q = RNG.choice([f"Divide {a} by {b}.", f"What's {a} split into {b} equal parts?"])
            items.append((q, str(a // b)))
    return items


def _gen_multi_turn_script(seed):
    r = random.Random(seed)
    connectives = ["By the way, ", "Building on that, ", "Switching topics — ", "While I have you, ",
                   "Real quick: ", "On a different note, ", "Following up, ", "To finish up, "]
    base = _gen_single_turn(8)  # reuse the same varied templates/kinds as the single-turn pool
    r.shuffle(connectives)
    script = []
    for i, (q, expect) in enumerate(base):
        text = q if i == 0 else connectives[i - 1] + q[0].lower() + q[1:]
        script.append((text, expect))
    return script


SINGLE_TURN = _gen_single_turn(20)
MULTI_TURN_SCRIPTS = [_gen_multi_turn_script(1), _gen_multi_turn_script(2)]


def _extract_answer(text):
    """Best-effort: pull the last number, or 'yes'/'no', or a comma-separated number list, from free text --
    robust to different phrasing styles ("42", "The answer is 42.", "= 42", "yes it is")."""
    text = text.strip().lower()
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?)+", text)  # comma-separated list first
    if nums:
        return re.sub(r"\s+", "", nums[-1])
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else None


def _is_repetitive(text, min_run=6):
    words = text.split()
    for i in range(len(words) - min_run):
        if len(set(words[i:i + min_run])) <= 2:
            return True
    return False


def _correct(got, expect):
    return got is not None and got.replace(" ", "") == expect.replace(" ", "")


def run(model, max_new_tokens=110):
    single_results = []
    for q, expect in SINGLE_TURN:
        text = model.generate([{"role": "user", "content": q}], max_new_tokens=max_new_tokens)
        got = _extract_answer(text)
        single_results.append({"q": q, "expect": expect, "got": got, "answer_text": text[:200],
                                "correct": _correct(got, expect)})

    multi_results = []
    collapses = 0
    for script in MULTI_TURN_SCRIPTS:
        history = []
        for q, expect in script:
            history.append({"role": "user", "content": q})
            text = model.generate(history, max_new_tokens=max_new_tokens)
            history.append({"role": "assistant", "content": text})
            got = _extract_answer(text)
            collapsed = _is_repetitive(text)
            collapses += collapsed
            multi_results.append({"q": q, "expect": expect, "got": got, "answer_text": text[:200],
                                   "correct": _correct(got, expect), "repetition_collapse": collapsed})

    single_score = sum(r["correct"] for r in single_results) / len(single_results)
    multi_score = sum(r["correct"] for r in multi_results) / len(multi_results)
    return {
        "single_turn_score": single_score,
        "multi_turn_score": multi_score,
        "repetition_collapses": collapses,
        "multi_turn_n": len(multi_results),
        "single_turn_results": single_results,
        "multi_turn_results": multi_results,
    }
