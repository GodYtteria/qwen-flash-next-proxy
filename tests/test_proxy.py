import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import proxy


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class CompleteSummaryTests(unittest.TestCase):
    def test_sends_ban_to_upstream_chat_endpoint_and_returns_content(self):
        reply = {"choices": [{"message": {"content": "Summary of parser.py."}}]}
        with mock.patch.object(proxy.urllib.request, "urlopen", return_value=FakeResponse(reply)) as opened:
            text = proxy.complete_summary({"messages": [{"role": "user", "content": "fold"}], "stream": False})
        request = opened.call_args.args[0]
        sent = json.loads(request.data)
        self.assertEqual(request.full_url, proxy.UPSTREAM + "/v1/chat/completions")
        self.assertEqual(sent["logit_bias"], proxy.BAN)
        self.assertFalse(sent["stream"])
        self.assertEqual(text, "Summary of parser.py.")


if __name__ == "__main__":
    unittest.main()
