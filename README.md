# qwen-flash-next-proxy

A small local proxy for Qwen3.8-Flash-Next (or any OpenAI/Anthropic-compatible server, such as Strata) that keeps
answers in English, Tagalog and code, and asks for complete, readable code.

Point your client at the proxy instead of the server. For every chat request it:

1. Adds a `logit_bias` that bans 104,205 of the model's 248,320 tokens (42%): CJK, Hangul, Cyrillic, Arabic and
   other scripts, plus single-token emoji. Latin script, digits, punctuation and symbols stay, so English, Tagalog
   and code are untouched. A `logit_bias` your client sends wins for the tokens it names.
2. Adds the rules in `style.txt` to the system prompt: full words ("Error", not "Err") while keeping real language
   syntax such as Rust `Err` and Go `err`, complete code with no `...`, no comments, no emoji.

Requests pass through unchanged otherwise, including streaming (SSE) and images.

## Requirements

Python 3.10 or newer. No packages: the standard library only. The upstream server must support `logit_bias`
(Strata does on CUDA, HIP and Metal).

## Run

```sh
python proxy.py
```

Then use `http://127.0.0.1:8081/v1` in your client. Settings, by environment variable:

| Variable | Default | Meaning |
| --- | --- | --- |
| `UPSTREAM` | `http://127.0.0.1:8080` | the model server |
| `LISTEN_HOST` | `127.0.0.1` | address to listen on (loopback only by default) |
| `LISTEN_PORT` | `8081` | port to listen on |
| `BAN_FILE` | `ban_ids.json` | token ids to ban |
| `STYLE_FILE` | `style.txt` | system-prompt rules; delete the file to turn this off |
| `MAX_BODY` | 64 MiB | largest request body accepted |

To run it at login as a user service, copy `systemd/qwen-flash-next-proxy.service` to `~/.config/systemd/user/`,
fix the paths in it, then `systemctl --user enable --now qwen-flash-next-proxy`.

## Rebuild the ban list for another tokenizer

`ban_ids.json` matches Qwen3.8-Flash-Next's tokenizer (Qwen3.6-35B-A3B uses the same one). For another model:

```sh
python build_ban_list.py /path/to/tokenizer-dir ban_ids.json
```

The folder must hold the model's `vocab.json`.

## Limits

- An emoji the tokenizer splits into several byte tokens is not one bannable token, so some emoji still get
  through the bias. The style rules also ask for no emoji.
- A token ban cannot remove comments or rename words: `#` and `//` are also code (`//` is Python floor division).
  The style rules handle those instead.
- The proxy does not check authentication. Keep it on loopback, or put it behind something that does.

## License

MIT
