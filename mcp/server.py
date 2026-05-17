#!/usr/bin/env python3
"""QwenTalk MCP stdio server."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from dataclasses import dataclass
from typing import Any

from tools import TOOL_REGISTRY, TOOL_SCHEMAS, call_tool

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "qwentalk-board-tools"
SERVER_VERSION = "0.1.0"


@dataclass(frozen=True)
class JsonRpcError:
    code: int
    message: str
    data: Any | None = None


class McpServer:
    def __init__(self) -> None:
        self.tool_defs = self._build_tool_defs()

    @staticmethod
    def _build_tool_defs() -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for item in TOOL_SCHEMAS:
            fn = item.get("function", {})
            name = fn.get("name")
            if not name or name not in TOOL_REGISTRY:
                continue
            tools.append({
                "name": name,
                "description": fn.get("description", ""),
                "inputSchema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return tools

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        request_id = request.get("id")
        method = request.get("method")
        if method is None:
            return self._error(request_id, JsonRpcError(-32600, "Missing JSON-RPC method"))
        if request_id is None and (method.startswith("notifications/") or method == "initialized"):
            return None
        try:
            if method == "initialize":
                return self._ok(request_id, self._initialize(request.get("params") or {}))
            if method == "ping":
                return self._ok(request_id, {})
            if method == "tools/list":
                return self._ok(request_id, {"tools": self.tool_defs})
            if method == "tools/call":
                return self._ok(request_id, self._call_tool(request.get("params") or {}))
            return self._error(request_id, JsonRpcError(-32601, f"Unknown method: {method}"))
        except Exception as exc:
            logging.exception("MCP method failed: %s", method)
            return self._error(
                request_id,
                JsonRpcError(
                    -32000,
                    f"Server error while handling {method}: {exc}",
                    data={"traceback": traceback.format_exc(limit=8)},
                ),
            )

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        }

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str) or not name:
            return self._tool_error("Missing tool name")
        if name not in TOOL_REGISTRY:
            return self._tool_error(f"Unknown QwenTalk tool: {name}")
        if not isinstance(args, dict):
            return self._tool_error("Tool arguments must be a JSON object")

        logging.info("tool call: %s args=%s", name, json.dumps(args, ensure_ascii=False, sort_keys=True))
        result = call_tool(name, args)
        is_error = isinstance(result, dict) and "error" in result
        text = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
        response: dict[str, Any] = {
            "content": [{"type": "text", "text": text}],
            "isError": bool(is_error),
        }
        if not is_error and isinstance(result, dict):
            response["structuredContent"] = result
        return response

    @staticmethod
    def _tool_error(message: str) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": message}], "isError": True}

    @staticmethod
    def _ok(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, error: JsonRpcError) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": error.code, "message": error.message}
        if error.data is not None:
            payload["data"] = error.data
        return {"jsonrpc": "2.0", "id": request_id, "error": payload}


def serve_stdio() -> int:
    server = McpServer()
    logging.info("%s ready with %d tools", SERVER_NAME, len(server.tool_defs))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            })
            continue
        if not isinstance(request, dict):
            _write_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "JSON-RPC request must be an object"},
            })
            continue
        response = server.handle(request)
        if response is not None:
            _write_response(response)
    return 0


def _write_response(response: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Expose QwenTalk hardware tools over MCP stdio")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[qwentalk-mcp] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    return serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
