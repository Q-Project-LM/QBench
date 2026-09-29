"""Generic HF causal-LM wrapper for QBench. Deliberately does NOT give any model external tool-execution
help (no forced-correct-answer logits processor, unlike Q-Agent's own internal dev tooling in
~/ml-intern-runs/q-agent/verify_v3.py) -- QBench measures what a model actually says on its own, uniformly
across every model it benchmarks, so scores are comparable."""
import hashlib
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fixed prompts used only to fingerprint a model's actual weight behavior (see ChatModel.fingerprint) --
# unrelated to any benchmark task, so they're not part of what's being scored.
_FINGERPRINT_PROMPTS = ["The capital of France is", "2 + 2 =", "Once upon a time,"]


class ChatModel:
    """Common interface every QBench module codes against."""

    def generate(self, messages: list[dict], max_new_tokens: int = 128) -> str:
        raise NotImplementedError

    def loglikelihood(self, context: str, continuation: str) -> float:
        """Sum log-prob the model assigns to `continuation` given `context` -- used for multiple-choice
        knowledge-benchmark scoring (compare the loglikelihood of each answer choice, pick the highest)."""
        raise NotImplementedError

    def fingerprint(self) -> str:
        """A short hash tying a submitted result to the actual weights that produced it -- the leaderboard
        has no way to verify anyone's claimed model_id otherwise, since scores are computed locally and
        just uploaded. Not cryptographically airtight (someone could still fabricate a matching forward
        pass), but it does mean a submitted result can be reproduced and checked by re-running the same
        model_id: a mismatched fingerprint on rerun is a clear, checkable red flag."""
        raise NotImplementedError


class HFChatModel(ChatModel):
    # float32 default: fp16 triggered a CUDA assert in GPT-2's attention masking path on this stack, and
    # every model QBench targets is small enough that fp32 costs negligible extra time.
    def __init__(self, model_id: str, device: str = None, dtype=torch.float32,
                 trust_remote_code: bool = False, subfolder: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"trust_remote_code": trust_remote_code}
        if subfolder:
            kw["subfolder"] = subfolder
        self.tok = AutoTokenizer.from_pretrained(model_id, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw).to(self.device).eval()
        if self.tok.pad_token_id is None:
            self.tok.pad_token_id = self.tok.eos_token_id
        cfg = self.model.config
        self.max_ctx = (getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)
                         or getattr(cfg, "n_ctx", None) or 4096)

    @torch.no_grad()
    def generate(self, messages, max_new_tokens=128):
        # apply_chat_template's return type (bare tensor vs. BatchEncoding) has varied across transformers
        # versions -- tokenize=False + a plain tokenizer call sidesteps that entirely.
        if self.tok.chat_template:
            prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        else:
            # base (non-instruct) models like GPT-2 have no chat template at all -- fall back to a plain
            # transcript. This disadvantages base models relative to instruct ones (expected: a model never
            # trained to follow this turn-taking format has no fair way to do so), but it's the only way to
            # give them a shot, and the same fallback format is used for every chat-template-less model.
            prompt = "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in messages) + "Assistant:"
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        # A growing multi-turn transcript can outrun a small model's own context window (GPT-2: 1024) --
        # on some stacks that shows up as an async CUDA assert deep in the attention mask code rather than
        # a clean Python error, so truncate defensively instead of letting it happen. Keep the most recent
        # tokens: for a live conversation, what was just said matters more than the opening turn.
        budget = self.max_ctx - max_new_tokens
        if ids.shape[1] > budget:
            ids = ids[:, -budget:]
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                   pad_token_id=self.tok.pad_token_id)
        # skip_special_tokens=False: some models (e.g. Q-U-164M) use special-token ids as *content*
        # delimiters around a tool result, not just structural markers -- silently dropping them merges
        # adjacent numbers into one unparseable digit blob ("347*86" + "29822" -> "8629822"). Harmless for
        # ordinary chat models: their structural tags (<|im_end|> etc.) just show up literally in
        # answer_text, which the regex-based scorers in each module already ignore.
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)

    @torch.no_grad()
    def loglikelihood(self, context, continuation):
        ctx_ids = self.tok(context, return_tensors="pt").input_ids.to(self.device)
        full_ids = self.tok(context + continuation, return_tensors="pt").input_ids.to(self.device)
        cont_len = full_ids.shape[1] - ctx_ids.shape[1]
        if cont_len <= 0:
            return float("-inf")
        logits = self.model(full_ids).logits[0]  # (seq, vocab)
        logp = torch.log_softmax(logits[-cont_len - 1:-1].float(), dim=-1)
        target = full_ids[0, -cont_len:]
        return logp.gather(-1, target.unsqueeze(-1)).sum().item()

    @torch.no_grad()
    def fingerprint(self) -> str:
        cfg_bytes = json.dumps(self.model.config.to_dict(), sort_keys=True).encode()
        h = hashlib.sha256(cfg_bytes)
        for p in _FINGERPRINT_PROMPTS:
            ids = self.tok(p, return_tensors="pt").input_ids.to(self.device)
            logits = self.model(ids).logits[0, -1].float()
            top = logits.topk(8)
            h.update(json.dumps({"ids": top.indices.tolist(),
                                  "logits": [round(x, 3) for x in top.values.tolist()]}).encode())
        return h.hexdigest()[:16]


def load(model_id: str, trust_remote_code: bool = False, subfolder: str = None) -> ChatModel:
    return HFChatModel(model_id, trust_remote_code=trust_remote_code, subfolder=subfolder)
