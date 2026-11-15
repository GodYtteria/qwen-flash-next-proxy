import contextlib
import io
import json
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import compress


def settings(trigger=50, keep=2, sessions=8, enabled=True):
    return compress.FoldSettings(
        enabled=enabled,
        trigger_tokens=trigger,
        keep_recent=keep,
        max_sessions=sessions,
        summary_max_tokens=64,
        timeout_seconds=5,
    )


def conversation(pairs, text="x" * 200, first="build the parser"):
    messages = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": first}]
    for index in range(pairs):
        messages.append({"role": "assistant", "content": f"step {index} {text}"})
        messages.append({"role": "user", "content": f"follow {index} {text}"})
    return messages


class FakeSummarizer:
    def __init__(self, reply="Summary: parser built, file parser.py, constraint: no deps."):
        self.reply = reply
        self.calls = []

    def __call__(self, body):
        self.calls.append(body)
        return self.reply


class Silence(contextlib.ContextDecorator):
    def __enter__(self):
        self.buffer = io.StringIO()
        self.redirect = contextlib.redirect_stderr(self.buffer)
        self.redirect.__enter__()
        return self.buffer

    def __exit__(self, *exc):
        self.redirect.__exit__(*exc)
        return False


class ReadSettingsTests(unittest.TestCase):
    def test_defaults_when_environment_empty(self):
        loaded = compress.read_settings({})
        self.assertEqual(loaded.trigger_tokens, 120000)
        self.assertEqual(loaded.keep_recent, 16)
        self.assertEqual(loaded.max_sessions, 32)
        self.assertEqual(loaded.summary_max_tokens, 1024)
        self.assertEqual(loaded.timeout_seconds, 900)
        self.assertTrue(loaded.enabled)

    def test_disabled_flag(self):
        self.assertFalse(compress.read_settings({"FOLD_ENABLED": "0"}).enabled)

    def test_rejects_bad_flag(self):
        with self.assertRaises(ValueError):
            compress.read_settings({"FOLD_ENABLED": "yes"})

    def test_rejects_zero_trigger(self):
        with self.assertRaises(ValueError):
            compress.read_settings({"FOLD_TRIGGER_TOKENS": "0"})

    def test_rejects_non_integer(self):
        with self.assertRaises(ValueError):
            compress.read_settings({"FOLD_KEEP_RECENT": "many"})

    def test_reads_custom_value(self):
        self.assertEqual(compress.read_settings({"FOLD_TRIGGER_TOKENS": "5000"}).trigger_tokens, 5000)


class FoldTests(unittest.TestCase):
    def test_under_trigger_passes_through_unchanged(self):
        messages = conversation(2, text="short")
        summarizer = FakeSummarizer()
        result = compress.fold_messages(messages, settings(trigger=100000), compress.FoldStore(8), summarizer)
        self.assertIs(result, messages)
        self.assertEqual(summarizer.calls, [])

    def test_over_trigger_folds_middle_into_one_summary(self):
        messages = conversation(20)
        summarizer = FakeSummarizer()
        result = compress.fold_messages(messages, settings(), compress.FoldStore(8), summarizer)
        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(result[0], messages[0])
        self.assertEqual(result[1], messages[1])
        self.assertEqual(result[2]["role"], "system")
        self.assertTrue(result[2]["content"].startswith(compress.SUMMARY_PREFIX))
        self.assertIn("parser.py", result[2]["content"])
        self.assertEqual(result[-2:], messages[-2:])
        self.assertLess(len(result), len(messages))

    def test_summary_request_carries_the_middle_and_no_style(self):
        messages = conversation(20)
        summarizer = FakeSummarizer()
        compress.fold_messages(messages, settings(), compress.FoldStore(8), summarizer, model="qwen")
        body = summarizer.calls[0]
        self.assertEqual(body["model"], "qwen")
        self.assertFalse(body["stream"])
        self.assertIn("step 0", body["messages"][1]["content"])
        self.assertNotIn("follow 19", body["messages"][1]["content"])

    def test_tool_pair_is_not_split(self):
        messages = conversation(20)
        messages[-3] = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]}
        messages[-2] = {"role": "tool", "tool_call_id": "c1", "content": "y" * 200}
        messages[-1] = {"role": "user", "content": "thanks"}
        result = compress.fold_messages(messages, settings(keep=2), compress.FoldStore(8), FakeSummarizer())
        self.assertIn(messages[-3], result)
        self.assertIn(messages[-2], result)
        self.assertEqual(result[-2:], messages[-2:])

    def test_second_identical_request_reuses_summary(self):
        messages = conversation(20)
        store = compress.FoldStore(8)
        summarizer = FakeSummarizer()
        first = compress.fold_messages(messages, settings(), store, summarizer)
        second = compress.fold_messages(list(messages), settings(), store, summarizer)
        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(first, second)

    def test_growth_refolds_incrementally_with_previous_summary(self):
        store = compress.FoldStore(8)
        summarizer = FakeSummarizer(reply="Summary part one: parser.py.")
        messages = conversation(20)
        compress.fold_messages(messages, settings(), store, summarizer)
        grown = messages + [{"role": "user", "content": "more " + "z" * 800} for _ in range(3)]
        summarizer.reply = "Summary part two: parser.py and tests."
        result = compress.fold_messages(grown, settings(), store, summarizer)
        self.assertEqual(len(summarizer.calls), 2)
        self.assertIn("Summary part one", summarizer.calls[1]["messages"][1]["content"])
        self.assertIn("part two", result[2]["content"])

    def test_summary_failure_forwards_original(self):
        def failing(body):
            raise OSError("connection refused")

        messages = conversation(20)
        with Silence() as logged:
            result = compress.fold_messages(messages, settings(), compress.FoldStore(8), failing)
        self.assertIs(result, messages)
        record = json.loads(logged.getvalue().splitlines()[0])
        self.assertEqual(record["event"], "fold_failed")
        self.assertEqual(record["error"], "OSError")
        self.assertNotIn("build the parser", logged.getvalue())

    def test_failed_refold_keeps_previous_summary(self):
        store = compress.FoldStore(8)
        messages = conversation(20)
        compress.fold_messages(messages, settings(), store, FakeSummarizer(reply="Good summary of parser.py."))
        grown = messages + [{"role": "user", "content": "more " + "z" * 800} for _ in range(3)]

        def failing(body):
            raise OSError("timeout")

        with Silence():
            result = compress.fold_messages(grown, settings(), store, failing)
        self.assertIn("Good summary of parser.py.", result[2]["content"])

    def test_empty_summary_is_rejected(self):
        messages = conversation(20)
        with Silence() as logged:
            result = compress.fold_messages(messages, settings(), compress.FoldStore(8), FakeSummarizer(reply="   "))
        self.assertIs(result, messages)
        self.assertIn("summary is empty", logged.getvalue())

    def test_summary_not_shorter_than_input_is_rejected(self):
        messages = conversation(20)
        with Silence() as logged:
            result = compress.fold_messages(messages, settings(), compress.FoldStore(8),
                                            FakeSummarizer(reply="w" * 100000))
        self.assertIs(result, messages)
        self.assertIn("not shorter", logged.getvalue())

    def test_disabled_never_calls_summarizer(self):
        summarizer = FakeSummarizer()
        messages = conversation(20)
        result = compress.fold_messages(messages, settings(enabled=False), compress.FoldStore(8), summarizer)
        self.assertIs(result, messages)
        self.assertEqual(summarizer.calls, [])

    def test_no_user_message_passes_through(self):
        messages = [{"role": "system", "content": "only system " + "q" * 900}]
        summarizer = FakeSummarizer()
        self.assertIs(compress.fold_messages(messages, settings(), compress.FoldStore(8), summarizer), messages)
        self.assertEqual(summarizer.calls, [])

    def test_empty_messages_passes_through(self):
        messages = []
        self.assertIs(compress.fold_messages(messages, settings(), compress.FoldStore(8), FakeSummarizer()), messages)

    def test_unicode_content_reaches_summarizer_intact(self):
        messages = conversation(20, text="模型 ไทย Tagalog 测试 " * 20)
        summarizer = FakeSummarizer()
        compress.fold_messages(messages, settings(), compress.FoldStore(8), summarizer)
        self.assertIn("模型 ไทย Tagalog 测试", summarizer.calls[0]["messages"][1]["content"])

    def test_store_evicts_least_recent_session(self):
        store = compress.FoldStore(1)
        summarizer = FakeSummarizer()
        first_session = conversation(20, first="alpha task")
        second_session = conversation(20, first="beta task")
        compress.fold_messages(first_session, settings(sessions=1), store, summarizer)
        compress.fold_messages(second_session, settings(sessions=1), store, summarizer)
        compress.fold_messages(first_session, settings(sessions=1), store, summarizer)
        self.assertEqual(len(summarizer.calls), 3)

    def test_concurrent_same_session_folds_once(self):
        messages = conversation(20)
        store = compress.FoldStore(8)
        calls = []
        lock = threading.Lock()

        def slow(body):
            with lock:
                calls.append(body)
            time.sleep(0.2)
            return "Summary of parser.py work."

        threads = [threading.Thread(target=compress.fold_messages,
                                    args=(list(messages), settings(), store, slow)) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len(calls), 1)


class RequestBodyTests(unittest.TestCase):
    def test_malformed_tool_call_function_does_not_crash(self):
        messages = conversation(20)
        messages[3]["tool_calls"] = [{"function": "not a dict"}]
        body = json.dumps({"model": "m", "messages": messages}).encode()
        folded = compress.fold_request_body(body, settings(), compress.FoldStore(8), FakeSummarizer())
        self.assertIsInstance(folded, bytes)

    def test_non_text_summary_is_forwarded_original(self):
        messages = conversation(20)
        body = json.dumps({"model": "m", "messages": messages}).encode()
        folded = compress.fold_request_body(body, settings(), compress.FoldStore(8), lambda request: ["not", "text"])
        self.assertIs(folded, body)

    def test_disabled_returns_same_bytes_without_parsing(self):
        body = b"not json"
        self.assertIs(compress.fold_request_body(body, settings(enabled=False), compress.FoldStore(8), FakeSummarizer()), body)

    def test_invalid_json_returns_same_bytes(self):
        body = b"not json"
        self.assertIs(compress.fold_request_body(body, settings(), compress.FoldStore(8), FakeSummarizer()), body)

    def test_non_chat_body_returns_same_bytes(self):
        body = json.dumps({"model": "m", "prompt": "hi"}).encode()
        self.assertIs(compress.fold_request_body(body, settings(), compress.FoldStore(8), FakeSummarizer()), body)

    def test_non_object_message_passes_through(self):
        body = json.dumps({"model": "m", "messages": ["not a message"] + conversation(20)[1:]}).encode()
        self.assertIs(compress.fold_request_body(body, settings(), compress.FoldStore(8), FakeSummarizer()), body)

    def test_lone_surrogate_in_history_does_not_crash(self):
        messages = conversation(20)
        messages[3]["content"] = "broken \ud800 text"
        body = json.dumps({"model": "m", "messages": messages}).encode()
        folded = compress.fold_request_body(body, settings(), compress.FoldStore(8), FakeSummarizer())
        self.assertTrue(json.loads(folded)["messages"][2]["content"].startswith(compress.SUMMARY_PREFIX))

    def test_folded_body_is_valid_json_with_summary(self):
        body = json.dumps({"model": "m", "messages": conversation(20)}).encode()
        folded = compress.fold_request_body(body, settings(), compress.FoldStore(8), FakeSummarizer())
        parsed = json.loads(folded)
        self.assertEqual(parsed["model"], "m")
        self.assertTrue(parsed["messages"][2]["content"].startswith(compress.SUMMARY_PREFIX))


if __name__ == "__main__":
    unittest.main()
