"""Inject a standing logit_bias (ban non-Latin script tokens) into every chat request, then forward to Strata.

Point your LLM client at this proxy instead of the engine. Keeps English/Tagalog/code; bans CJK, Hangul, Cyrillic,
Arabic and the rest. A client-supplied logit_bias wins per token. Streaming and non-streaming both pass through.
"""
import json
import os
import socketserver
import urllib.request
from http.server import BaseHTTPRequestHandler

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:8080")
LISTEN_HOST = os.environ.get("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "8081"))
HERE = os.path.dirname(os.path.abspath(__file__))
BAN_FILE = os.environ.get("BAN_FILE", os.path.join(HERE, "ban_ids.json"))
STYLE_FILE = os.environ.get("STYLE_FILE", os.path.join(HERE, "style.txt"))
BIAS_PATHS = ("/v1/chat/completions", "/v1/messages", "/v1/completions")
MAX_BODY = int(os.environ.get("MAX_BODY", str(64 * 1024 * 1024)))

with open(BAN_FILE, encoding="utf-8") as fh:
    BAN = {str(t): -100.0 for t in json.load(fh)}
STYLE = ""
if os.path.exists(STYLE_FILE):
    with open(STYLE_FILE, encoding="utf-8") as fh:
        STYLE = fh.read().strip()


def merge_bias(body):
    """Return body bytes with the ban merged in; client entries override. None when body is not JSON we should touch."""
    try:
        obj = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(obj, dict):
        return None
    client = obj.get("logit_bias") or {}
    if isinstance(client, list):
        client = {str(p[0]): p[1] for p in client if isinstance(p, (list, tuple)) and len(p) == 2}
    merged = dict(BAN)
    if isinstance(client, dict):
        merged.update({str(k): v for k, v in client.items()})
    obj["logit_bias"] = merged
    if STYLE:
        inject_style(obj)
    return json.dumps(obj).encode()


def inject_style(obj):
    """Add the style directive without dropping a client's own system prompt. Handles OpenAI and Anthropic shapes."""
    if "messages" in obj and isinstance(obj["messages"], list):
        msgs = obj["messages"]
        if "system" in obj:                       # Anthropic: top-level system
            obj["system"] = _join_system(obj.get("system"))
        elif msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
            msgs[0]["content"] = _join_text(msgs[0].get("content"), STYLE)
        else:
            msgs.insert(0, {"role": "system", "content": STYLE})
    elif "system" in obj:
        obj["system"] = _join_system(obj.get("system"))


def _join_system(existing):
    if isinstance(existing, list):               # Anthropic content-block list
        return existing + [{"type": "text", "text": STYLE}]
    if isinstance(existing, str) and existing.strip():
        return existing.rstrip() + "\n\n" + STYLE
    return STYLE


def _join_text(existing, extra):
    if isinstance(existing, list):
        return existing + [{"type": "text", "text": extra}]
    if isinstance(existing, str) and existing.strip():
        return existing.rstrip() + "\n\n" + extra
    return extra


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _relay(self, method):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            self.send_error(413, "body too large")
            return
        body = self.rfile.read(length) if length else b""
        if method == "POST" and self.path in BIAS_PATHS and body:
            patched = merge_bias(body)
            if patched is not None:
                body = patched
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
        headers["Content-Length"] = str(len(body))
        headers["Accept-Encoding"] = "identity"
        req = urllib.request.Request(UPSTREAM + self.path, data=body if method == "POST" else None,
                                     headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=36000)
        except urllib.error.HTTPError as e:
            resp = e
        except OSError as e:
            self.send_error(502, f"upstream unreachable: {e}")
            return
        self.send_response(resp.status)
        passthrough = [(k, v) for k, v in resp.getheaders()
                       if k.lower() not in ("transfer-encoding", "connection", "content-encoding")]
        chunked = any(k.lower() == "content-length" for k, _ in passthrough) is False
        for k, v in passthrough:
            self.send_header(k, v)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                if chunked:
                    self.wfile.write(b"%X\r\n%s\r\n" % (len(chunk), chunk))
                else:
                    self.wfile.write(chunk)
            if chunked:
                self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            resp.close()

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print(f"lang-ban proxy on {LISTEN_HOST}:{LISTEN_PORT} -> {UPSTREAM}  ({len(BAN)} tokens banned)", flush=True)
    Server((LISTEN_HOST, LISTEN_PORT), Handler).serve_forever()
