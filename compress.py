import hashlib
import json
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

DEFAULT_ENABLED = True
DEFAULT_TRIGGER_TOKENS = 120000
DEFAULT_KEEP_RECENT = 16
DEFAULT_MAX_SESSIONS = 32
DEFAULT_SUMMARY_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SECONDS = 900
CHARACTERS_PER_TOKEN = 3.5
SUMMARY_TEMPERATURE = 0.2
SUMMARY_REASONING_EFFORT = "low"
SUMMARY_PREFIX = "[folded history]"
SUMMARY_INSTRUCTION = (
    "You fold older parts of a coding-agent conversation into a summary the agent reads later. "
    "Keep: decisions, user constraints and preferences, file paths, function and variable names, "
    "error messages and how they were fixed, commands that were run and their results, open tasks. "
    "Drop: greetings, repeated or large tool output, file bodies that can be re-read. "
    "Write plain English. Do not invent facts. If a previous summary is given, merge it in."
)


@dataclass(frozen=True)
class FoldSettings:
    enabled: bool
    trigger_tokens: int
    keep_recent: int
    max_sessions: int
    summary_max_tokens: int
    timeout_seconds: int


def read_settings(environment=os.environ):
    return FoldSettings(
        enabled=parse_flag(environment, "FOLD_ENABLED", DEFAULT_ENABLED),
        trigger_tokens=parse_positive_int(environment, "FOLD_TRIGGER_TOKENS", DEFAULT_TRIGGER_TOKENS),
        keep_recent=parse_positive_int(environment, "FOLD_KEEP_RECENT", DEFAULT_KEEP_RECENT),
        max_sessions=parse_positive_int(environment, "FOLD_MAX_SESSIONS", DEFAULT_MAX_SESSIONS),
        summary_max_tokens=parse_positive_int(environment, "FOLD_SUMMARY_MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS),
        timeout_seconds=parse_positive_int(environment, "FOLD_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS),
    )


def parse_flag(environment, name, default):
    raw = environment.get(name)
    if raw is None:
        return default
    if raw == "1":
        return True
    if raw == "0":
        return False
    raise ValueError(f"{name} must be 0 or 1, got {raw!r}")


def parse_positive_int(environment, name, default):
    raw = environment.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


class FoldError(Exception):
    pass


@dataclass
class Fold:
    fingerprint: str
    covered_end: int
    summary: str


@dataclass
class SessionEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    fold: Fold | None = None


class FoldStore:
    def __init__(self, max_sessions):
        self.max_sessions = max_sessions
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def session(self, key):
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                entry = SessionEntry()
                self.entries[key] = entry
                while len(self.entries) > self.max_sessions:
                    self.entries.popitem(last=False)
            else:
                self.entries.move_to_end(key)
            return entry


def log_event(event, **fields):
    record = {"event": event, "time": round(time.time(), 3), **fields}
    sys.stderr.write(json.dumps(record, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def message_text(message):
    content = message.get("content")
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        parts = [str(part.get("text") or "") if isinstance(part, dict) else str(part) for part in content]
    else:
        parts = []
    for call in message.get("tool_calls") or []:
        if isinstance(call, dict) and isinstance(call.get("function"), dict):
            function = call["function"]
            parts.append(f"{function.get('name', '')} {function.get('arguments', '')}")
    return "\n".join(part for part in parts if part)


def estimate_tokens(messages):
    return sum(math.ceil(len(message_text(message)) / CHARACTERS_PER_TOKEN) for message in messages)


def head_length(messages):
    index = 0
    while index < len(messages) and messages[index].get("role") == "system":
        index += 1
    if index < len(messages) and messages[index].get("role") == "user":
        index += 1
    return index


def tail_start(messages, head_end, keep_recent):
    start = max(head_end, len(messages) - keep_recent)
    while start > head_end and messages[start].get("role") == "tool":
        start -= 1
    return start


def fingerprint(messages):
    encoded = json.dumps(messages, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def session_key(messages):
    for message in messages:
        if message.get("role") == "user":
            return hashlib.sha256(message_text(message).encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return None


def transcript(messages):
    return "\n\n".join(f"{message.get('role', 'unknown')}: {message_text(message)}" for message in messages)


def summary_request(previous_summary, pending, model, settings):
    user_content = ""
    if previous_summary:
        user_content += "Previous summary:\n" + previous_summary + "\n\n"
    user_content += "Conversation to fold in:\n" + transcript(pending)
    body = {
        "messages": [
            {"role": "system", "content": SUMMARY_INSTRUCTION},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": settings.summary_max_tokens,
        "temperature": SUMMARY_TEMPERATURE,
        "reasoning_effort": SUMMARY_REASONING_EFFORT,
        "stream": False,
    }
    if model:
        body["model"] = model
    return body, estimate_tokens([{"content": user_content}])


def summarize(previous_summary, pending, model, settings, complete):
    body, input_tokens = summary_request(previous_summary, pending, model, settings)
    reply = complete(body)
    if not isinstance(reply, str) or not reply.strip():
        raise FoldError("summary is empty or not text")
    text = reply.strip()
    if estimate_tokens([{"content": text}]) >= input_tokens:
        raise FoldError("summary is not shorter than its input")
    return text


def summary_message(text):
    return {"role": "system", "content": f"{SUMMARY_PREFIX}\n{text}"}


def fold_messages(messages, settings, store, complete, model=None):
    head_end = head_length(messages)
    key = session_key(messages)
    if not settings.enabled or key is None:
        return messages
    start = tail_start(messages, head_end, settings.keep_recent)
    entry = store.session(key)
    with entry.lock:
        fold = entry.fold
        if fold is not None and not (
            fold.covered_end <= start and fingerprint(messages[:fold.covered_end]) == fold.fingerprint
        ):
            fold = None
            entry.fold = None
        covered_start = fold.covered_end if fold is not None else head_end
        pending = messages[covered_start:start]
        if estimate_tokens(pending) <= settings.trigger_tokens:
            if fold is None:
                return messages
            return messages[:head_end] + [summary_message(fold.summary)] + messages[fold.covered_end:]
        previous_summary = fold.summary if fold is not None else None
        try:
            text = summarize(previous_summary, pending, model, settings, complete)
        except (FoldError, OSError, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            log_event("fold_failed", session=key, error=type(error).__name__, detail=str(error))
            if fold is None:
                return messages
            return messages[:head_end] + [summary_message(fold.summary)] + messages[fold.covered_end:]
        entry.fold = Fold(fingerprint=fingerprint(messages[:start]), covered_end=start, summary=text)
        log_event("fold_done", session=key, folded_messages=start - head_end, summary_chars=len(text))
        return messages[:head_end] + [summary_message(text)] + messages[start:]


def fold_request_body(body, settings, store, complete):
    if not settings.enabled:
        return body
    try:
        request = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    if not isinstance(request, dict) or not isinstance(request.get("messages"), list):
        return body
    messages = request["messages"]
    if not all(isinstance(message, dict) for message in messages):
        return body
    folded = fold_messages(messages, settings, store, complete, request.get("model"))
    if folded is messages:
        return body
    request["messages"] = folded
    return json.dumps(request).encode("utf-8")
