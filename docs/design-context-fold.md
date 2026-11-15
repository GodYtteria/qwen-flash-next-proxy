# Design: incremental context fold in qwen-flash-next-proxy

Status: approved and implemented in `compress.py`, `proxy.py`, and `tests/`. Not committed.

## Problem

Long agent sessions (opencode, pi) grow the chat history until it hits the 262,144-token window or slows prefill.
The proxy currently only adds a logit ban and style rules; it does not reduce history.

Measured limits that shape the design (from memory notes, Flash-Next UD-IQ3_XXS on this box):
- Prefill about 1,190 tok/s from 32K to 120K. Re-summarizing a 150K history from scratch costs about 2 minutes, so the fold must be incremental.
- Prefix cache hit depends on a stable prompt prefix. Any rewrite of early history invalidates the cache from that point.

## Scope

In scope:
- OpenAI-style `POST /v1/chat/completions` (what opencode uses through the `strata` provider).
- One new module `compress.py` that folds old history into one summary message.
- A hook in `proxy.py` that calls it before the existing `merge_bias`.
- Tests, and a README section for the new environment variables.

Out of scope:
- Anthropic `/v1/messages` and `/v1/completions` (passed through unchanged).
- Decompression or retrieval tools, multi-tier summaries, DB or session store, UI, any code from billion-context.
- Changes to the logit ban, style rules, or systemd unit.

This is a clean-room design. Behavior is inspired by the public README of billion-context (incremental, prefix-cache friendly summaries). No billion-context source is copied, so no parity checklist applies. Differences are listed below.

## Design

### Trigger and boundary

- Token estimate: `ceil(characters / 3.5)` over message text. Crude, but it only gates when to fold, and the upstream usage field is not available before the call.
- Fold runs when the estimate of the un-folded middle exceeds `FOLD_TRIGGER_TOKENS` (default 120,000).
- Protected zones, never folded:
  - the system message(s) at the start,
  - the first user message,
  - the last `FOLD_KEEP_RECENT` messages (default 16).
- The boundary moves forward until it does not split an assistant `tool_calls` message from its `tool` results.

### Summary message

- Created by one non-streaming upstream call to the same server (`UPSTREAM`), `max_tokens` 1,024, `temperature` 0.2, `reasoning_effort` low, no `logit_bias` change (the existing ban is applied to the summary call too, as it is a normal chat request).
- Prompt keeps: decisions, user constraints, file paths, function names, error messages and their fixes, open tasks. Drops: greetings, repeated tool output, large file bodies.
- Inserted as one `system` message right after the protected head (leading system messages and the first user message), prefixed `[folded history]`.
- Incremental: a later fold summarizes only the new un-folded messages plus the previous summary. Summaries are not re-summarized from scratch.

### Session identity and cache stability

- Session key: SHA-256 of the first user message text (truncated to 16 hex chars). Stored in memory only.
- The summary changes only when a new fold runs. Between folds the request prefix is byte-identical, so the prefix cache holds.
- Store: `OrderedDict` capped at `FOLD_MAX_SESSIONS` (default 32), evicted LRU, guarded by a `threading.Lock`. Per-session lock so two requests of one session do not fold twice.
- A restart of the proxy drops the summaries. The next request re-folds the middle once. This is accepted and stated in the README.

### Failure handling

- Summary call fails (network error, non-200, empty content): log one JSON line with the session key, status, and error class, then forward the original request unchanged. No silent drop.
- A summary that is empty or longer than the middle it replaces is rejected and logged, and the request is forwarded unchanged.
- Bad JSON in the request body: existing behavior (forward unchanged).
- Logs contain no message content, no keys, no full prompts.

### Configuration

All defaults live in one place at the top of `compress.py` and are read once at startup; invalid values stop the proxy with a clear message:
`FOLD_ENABLED` (default `1`), `FOLD_TRIGGER_TOKENS`, `FOLD_KEEP_RECENT`, `FOLD_MAX_SESSIONS`, `FOLD_SUMMARY_MAX_TOKENS`.

## Tests (stdlib `unittest`, must fail on wrong code)

- Under threshold: body bytes identical to input except the existing logit merge.
- Over threshold: exactly one summary call; output has one `[folded history]` message; protected head and tail unchanged.
- Tool pairing: a boundary that would split an `assistant.tool_calls` / `tool` pair is moved; asserted on a fixture.
- Cache stability: two identical-prefix requests in a row make one summary call total and the second body keeps the first summary byte-for-byte.
- Summary failure: upstream returns 500 → original body forwarded, error logged, no exception to the client.
- Empty and oversized summary: rejected, forwarded unchanged.
- Unicode and CJK content in the middle: summary input and output kept as UTF-8.
- Zero messages, one message, no `messages` key: passthrough.
- Config validation: non-integer or negative env value fails at startup.

## Success check

- Before merge: all unit tests pass; gate command recorded with its exit code and output tail.
- Functional: start the proxy on 8081 against the live Flash server, replay a recorded opencode session of at least 150K estimated tokens, and confirm one fold, a request body under the trigger afterward, and a reply that still answers a question about content from the folded part (checked by hand on three questions).
- Speed: the fold call adds one prefill of the un-folded middle; report the measured seconds. No target beyond "reported".

## Risks

- Quality: Flash IQ3_XXS summaries may drop details. Mitigation: the three-question check above; fall back to passthrough if a summary is rejected.
- Token estimate error: could fold too early or too late. Mitigation: trigger well below the window (120K of 262K).
- First fold after restart is slow (minutes at full history). Mitigation: documented; `FOLD_ENABLED=0` turns it off.
- Summary call competes with the main request on the single GPU. Mitigation: fold runs inline before forwarding, so requests are serialized per session.

## Files

- `compress.py` (new): trigger, boundary, summary call, store.
- `proxy.py`: one hook call before `merge_bias` for chat completions.
- `tests/test_compress.py` (new).
- `README.md`: environment variables and the restart note.
- `docs/design-context-fold.md`: this file (uncommitted until approved).

## Decisions I made (change any of these)

- Clean-room, not a port of billion-context.
- Chat completions only; Anthropic path untouched.
- In-memory summaries, lost on restart.
- Defaults: trigger 120,000, keep 16, cap 32 sessions, summary max 1,024 tokens.

Approve this design to start coding, or name what to change.
