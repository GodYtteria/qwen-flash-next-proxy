# Roadmap

Work queue, easy to hard.

- ✅ Ban non-Latin scripts and single-token emoji with `logit_bias` (104,205 of 248,320 tokens).
- ✅ Add readable-code rules to the system prompt (OpenAI and Anthropic request shapes).
- ✅ Pass streaming (SSE) and image requests through unchanged.
- ✅ Rebuild the ban list for any byte-level BPE tokenizer (`build_ban_list.py`).
- ⬜ Strip multi-token emoji from streamed output (the bias cannot reach them).
- ⬜ Tests for `merge_bias` and `inject_style` (empty body, list and dict `logit_bias`, existing system prompt).
- ⬜ Optional API key check for use beyond loopback.
