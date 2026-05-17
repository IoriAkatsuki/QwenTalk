#!/usr/bin/env python3
"""Loop-test rolling compression plus SQLite memory recall.

The default corpus is a public-domain Project Gutenberg fantasy text, not
Tolkien's copyrighted The Lord of the Rings.
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from edge_memory import memory_stats, recall_memory, store_memory

DEFAULT_URLS = [
    "https://www.gutenberg.org/cache/epub/67090/pg67090.txt",
    "https://www.gutenberg.org/files/67090/67090-0.txt",
]
DEFAULT_CORPUS = Path("test_corpus/worm_ouroboros_gutenberg.txt")
CHAT_URL = "http://127.0.0.1:8080/v1/chat/completions"


def download_corpus(path: Path, urls: list[str]) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 100_000:
        return path, "cached"
    last_error = ""
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QwenTalk-memory-test/0.1"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
            if len(data) < 100_000:
                last_error = f"{url}: too small ({len(data)} bytes)"
                continue
            path.write_bytes(data)
            return path, url
        except http.client.IncompleteRead as exc:
            data = exc.partial or b""
            if len(data) >= 100_000:
                path.write_bytes(data)
                return path, f"{url} (partial {len(data)} bytes after IncompleteRead)"
            last_error = f"{url}: incomplete read ({len(data)} bytes)"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = f"{url}: {exc}"
    raise RuntimeError(f"failed to download corpus: {last_error}")


def load_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = re.sub(r"\r\n?", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw


def pick_chunks(text: str, rounds: int, chunk_chars: int) -> list[str]:
    if len(text) < chunk_chars:
        return [text]
    start = max(0, text.find("*** START"))
    end = text.find("*** END")
    body = text[start:end if end > start else len(text)]
    body = body or text
    span = max(1, len(body) - chunk_chars)
    chunks = []
    for i in range(rounds):
        pos = int(i * span / max(1, rounds - 1))
        chunk = body[pos:pos + chunk_chars]
        if len(chunk) < chunk_chars // 2:
            chunk = body[-chunk_chars:]
        chunks.append(chunk)
    return chunks


def chat_compress(chunk: str, checkpoint: str, max_tokens: int) -> tuple[str, dict[str, Any]]:
    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You compress chat history for a local edge agent. "
                    "Keep factual anchors exact. Do not include hidden thinking."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Compress this source into compact memory notes under 120 English words. "
                    "You MUST preserve this checkpoint line exactly:\n"
                    f"{checkpoint}\n\n"
                    "Source excerpt:\n"
                    f"{chunk}"
                ),
            },
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=240) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    content = body["choices"][0]["message"].get("content") or ""
    usage = body.get("usage", {})
    usage["elapsed_s"] = round(elapsed, 3)
    return content.strip(), usage


def run(args: argparse.Namespace) -> dict[str, Any]:
    corpus_path, source = download_corpus(args.corpus, DEFAULT_URLS)
    text = load_text(corpus_path)
    chunks = pick_chunks(text, args.rounds, args.chunk_chars)

    rows = []
    for i, chunk in enumerate(chunks, start=1):
        key = f"QTLOOP-{int(time.time())}-{i:02d}"
        value = f"anchor-{i:02d}-{len(chunk)}"
        checkpoint = f"CHECKPOINT {key} VALUE {value}"
        compressed, usage = chat_compress(chunk, checkpoint, args.max_tokens)
        saved = store_memory(
            content=compressed,
            memory_type="short_term",
            importance=0.82,
            tags=["compression_loop", f"round-{i}", key],
            source="memory_compression_loop",
            summary=f"compressed round {i}",
            metadata={"checkpoint": checkpoint, "corpus_source": source},
        )
        rows.append({
            "round": i,
            "key": key,
            "value": value,
            "checkpoint": checkpoint,
            "memory_id": saved.get("id"),
            "chunk_chars": len(chunk),
            "compressed_chars": len(compressed),
            "compression_ratio": round(len(compressed) / max(1, len(chunk)), 4),
            "checkpoint_in_summary": checkpoint in compressed,
            "usage": usage,
        })

    recovered = 0
    recall_rows = []
    for row in rows:
        result = recall_memory(row["key"], top_k=1, reinforce=True)
        top = result.get("results", [{}])[0] if result.get("results") else {}
        content = top.get("content", "")
        ok = row["key"] in content and row["value"] in content
        recovered += int(ok)
        recall_rows.append({
            "round": row["round"],
            "key": row["key"],
            "recovered": ok,
            "top_memory_id": top.get("id"),
            "top_score": top.get("score"),
            "top_retention": top.get("retention"),
            "top_access_count": top.get("access_count"),
        })

    return {
        "corpus_path": str(corpus_path),
        "corpus_source": source,
        "corpus_chars": len(text),
        "rounds": len(rows),
        "chunk_chars": args.chunk_chars,
        "stored": rows,
        "recall": recall_rows,
        "retention_rate": round(recovered / max(1, len(rows)), 4),
        "memory_stats": memory_stats(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test 64K rolling compression and SQLite memory retention")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--chunk-chars", type=int, default=4500)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=Path("/tmp/qwentalk_memory_compression_loop.json"))
    args = parser.parse_args()

    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
