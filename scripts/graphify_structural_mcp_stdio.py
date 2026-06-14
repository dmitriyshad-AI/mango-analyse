#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path

from mango_mvp.graphify_structural import curated_guidance, graph_source_hints, graph_stats, stale_banner


LOCAL_TOOLS = [
    {
        "name": "query_graph",
        "description": "Read-only local search over Mango Graphify structural graph. Returns source path hints.",
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string"}, "token_budget": {"type": "integer", "default": 2000}},
            "required": ["question"],
        },
    },
    {
        "name": "get_node",
        "description": "Read-only node lookup by label or id.",
        "inputSchema": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]},
    },
    {
        "name": "get_neighbors",
        "description": "Read-only direct neighbor lookup.",
        "inputSchema": {"type": "object", "properties": {"label": {"type": "string"}}, "required": ["label"]},
    },
    {
        "name": "graph_stats",
        "description": "Read-only graph statistics.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "shortest_path",
        "description": "Read-only shortest path between two labels/ids in the local graph.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string"},
                "target": {"type": "string"},
                "max_hops": {"type": "integer", "default": 8},
            },
            "required": ["source", "target"],
        },
    },
]


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in re.findall(r"[\w_]+", text) if len(token) > 2]


def _find_node(nodes: list[dict], needle: str) -> dict | None:
    folded = needle.casefold()
    tokens = _tokens(needle)
    best: tuple[int, int, dict] | None = None
    for node in nodes:
        label = str(node.get("label") or "")
        node_id = str(node.get("id") or "")
        haystack = f"{label} {node_id} {node.get('source_file') or ''}".casefold()
        score = 0
        if folded == label.casefold() or folded == node_id.casefold():
            score += 1000
        if folded in haystack:
            score += 100
        score += sum(1 for token in tokens if token in haystack)
        if score:
            candidate = (score, -len(label), node)
            if best is None or candidate > best:
                best = candidate
    return best[2] if best else None


def _node_line(node: dict) -> str:
    return (
        f"{node.get('label') or node.get('id')} "
        f"[id={node.get('id')} src={node.get('source_file') or ''} loc={node.get('source_location') or ''}]"
    )


def _tool_response(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _handle_tool(name: str, arguments: dict, *, repo: Path, graph: Path, data: dict) -> dict:
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    banner = stale_banner(repo, graph)
    if name == "graph_stats":
        return _tool_response(f"{banner}\n" + json.dumps(graph_stats(graph), ensure_ascii=False, sort_keys=True))
    if name == "query_graph":
        question = str(arguments.get("question") or "")
        hints = graph_source_hints(graph, question, limit=20)
        guidance = curated_guidance(question)
        lines = [banner, "Source path hints:"]
        lines.extend(f"- {hint}" for hint in hints)
        if guidance:
            lines.append("Guard notes:")
            lines.extend(f"- {item}" for item in guidance)
        lines.append("Правило: это навигация; факт подтвердить чтением исходника.")
        return _tool_response("\n".join(lines))
    if name == "get_node":
        node = _find_node(nodes, str(arguments.get("label") or ""))
        return _tool_response(f"{banner}\n{_node_line(node) if node else 'Node not found; check raw source with rg.'}")
    if name == "get_neighbors":
        node = _find_node(nodes, str(arguments.get("label") or ""))
        if not node:
            return _tool_response(f"{banner}\nNode not found; check raw source with rg.")
        node_id = str(node.get("id"))
        by_id = {str(item.get("id")): item for item in nodes}
        lines = [banner, _node_line(node), "Neighbors:"]
        for edge in edges:
            src = str(edge.get("source"))
            dst = str(edge.get("target"))
            if src == node_id or dst == node_id:
                other = by_id.get(dst if src == node_id else src)
                if other:
                    lines.append(f"- {edge.get('relation')}: {_node_line(other)}")
            if len(lines) >= 42:
                lines.append("... truncated")
                break
        return _tool_response("\n".join(lines))
    if name == "shortest_path":
        source = _find_node(nodes, str(arguments.get("source") or ""))
        target = _find_node(nodes, str(arguments.get("target") or ""))
        if not source or not target:
            return _tool_response(f"{banner}\nSource or target not found; check raw source with rg.")
        max_hops = int(arguments.get("max_hops") or 8)
        adjacency: dict[str, set[str]] = {}
        for edge in edges:
            src = str(edge.get("source"))
            dst = str(edge.get("target"))
            adjacency.setdefault(src, set()).add(dst)
            adjacency.setdefault(dst, set()).add(src)
        start = str(source.get("id"))
        finish = str(target.get("id"))
        queue: deque[list[str]] = deque([[start]])
        seen = {start}
        path: list[str] | None = None
        while queue:
            current = queue.popleft()
            if len(current) - 1 > max_hops:
                continue
            if current[-1] == finish:
                path = current
                break
            for nxt in sorted(adjacency.get(current[-1], ())):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append([*current, nxt])
        by_id = {str(item.get("id")): item for item in nodes}
        if not path:
            return _tool_response(f"{banner}\nNo path found within {max_hops} hops; absence must be checked in raw source.")
        return _tool_response(banner + "\n" + "\n".join(_node_line(by_id[node_id]) for node_id in path if node_id in by_id))
    return {"isError": True, "content": [{"type": "text", "text": f"Unsupported read-only tool: {name}"}]}


def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def serve_stdio(*, repo: Path, graph: Path) -> int:
    data = json.loads(graph.read_text(encoding="utf-8"))
    for raw in sys.stdin:
        if not raw.strip():
            continue
        request = json.loads(raw)
        method = request.get("method")
        request_id = request.get("id")
        if method == "initialize":
            _send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "mango-graphify-structural", "version": "1"},
                    },
                }
            )
        elif method == "tools/list":
            _send({"jsonrpc": "2.0", "id": request_id, "result": {"tools": LOCAL_TOOLS}})
        elif method == "tools/call":
            params = request.get("params") or {}
            result = _handle_tool(
                str(params.get("name") or ""),
                dict(params.get("arguments") or {}),
                repo=repo,
                graph=graph,
                data=data,
            )
            _send({"jsonrpc": "2.0", "id": request_id, "result": result})
        elif method == "notifications/initialized":
            continue
        else:
            if request_id is not None:
                _send(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32601, "message": f"Unsupported method: {method}"},
                    }
                )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Mango Graphify map over local read-only stdio MCP.")
    parser.add_argument("--graph", required=True, type=Path, help="Path to graphify-out/graph.json outside git.")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repo root for revision banner.")
    args = parser.parse_args()
    graph = args.graph.resolve()
    if graph.suffix != ".json" or not graph.exists():
        raise SystemExit(f"graph JSON not found: {graph}")
    return serve_stdio(repo=args.repo.resolve(), graph=graph)


if __name__ == "__main__":
    raise SystemExit(main())
