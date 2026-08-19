"""HTTP transport for the checked-in local HydraDB graph-node."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from context_memory.core.logging import get_logger, timed_operation

logger = get_logger(__name__)


class HydraHttpError(RuntimeError):
    """Local graph-node HTTP failure; bearer token never appears in this error."""


HttpRequester = Callable[[str, str, Mapping[str, str], bytes], Mapping[str, object]]


class HydraHttpTransport:
    def __init__(self, base_url: str, auth_token: str, *, namespace: str = "default", graph_id: str = "default", cell_id: str = "cell-0", requester: HttpRequester | None = None) -> None:
        if not base_url.startswith(("http://", "https://")) or not auth_token or not namespace or not graph_id or not cell_id:
            raise ValueError("base_url, auth token, namespace, graph_id, and cell_id must be non-empty")
        self._url = f"{base_url.rstrip('/')}/v1/graphs/{graph_id}/query"
        self._headers = {"Authorization": f"Bearer {auth_token}", "X-Graph-Namespace": namespace, "Content-Type": "application/json", "Accept": "application/json"}
        self._cell_id, self._requester = cell_id, requester or _request_json

    def write(self, cypher: str, rows: Sequence[dict[str, object]], idempotency_key: str) -> str | None:
        with timed_operation(logger, "hydradb.write", {"rows_count": len(rows), "idempotency_key": idempotency_key}) as ctx:
            response = self._query(cypher, {"rows": list(rows)}, query_id=idempotency_key)
            bookmark = response.get("bookmark")
            if bookmark is not None and not isinstance(bookmark, str):
                raise HydraHttpError("HydraDB returned an invalid bookmark")
            ctx["bookmark"] = bookmark
            return bookmark

    def read(self, cypher: str, parameters: dict[str, object], bookmark: str | None) -> Sequence[dict[str, object]]:
        snippet = cypher[:60].replace("\n", " ") + "..." if len(cypher) > 60 else cypher
        with timed_operation(logger, "hydradb.read", {"query_snippet": snippet}) as ctx:
            response = self._query(cypher, parameters, bookmark=bookmark)
            columns, rows = response.get("columns"), response.get("rows")
            if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns) or not isinstance(rows, list):
                raise HydraHttpError("HydraDB returned invalid query rows")
            result: list[dict[str, object]] = []
            for row in rows:
                if not isinstance(row, list) or len(row) != len(columns):
                    raise HydraHttpError("HydraDB returned a malformed query row")
                result.append({column: _decode_value(value) for column, value in zip(columns, row, strict=True)})
            ctx["result_rows"] = len(result)
            return result

    def _query(self, cypher: str, parameters: Mapping[str, object], *, query_id: str | None = None, bookmark: str | None = None) -> Mapping[str, object]:
        payload: dict[str, object] = {"cell_id": self._cell_id, "query": cypher, "parameters": parameters}
        if query_id is not None:
            payload["query_id"] = query_id
        if bookmark is not None:
            payload["bookmark"] = bookmark
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        try:
            return self._requester("POST", self._url, self._headers, body)
        except HydraHttpError:
            raise
        except Exception as error:
            raise HydraHttpError(f"local HydraDB request failed: {type(error).__name__}") from error


def _decode_value(value: object) -> object:
    if not isinstance(value, Mapping):
        raise HydraHttpError("HydraDB returned an untyped query value")
    value_type = value.get("type")
    if value_type == "null":
        return None
    if value_type in {"vertex_id", "integer", "signed_integer", "float", "boolean", "string"}:
        return value.get("value")
    if value_type == "list" and isinstance(value.get("value"), list):
        return [_decode_value(item) for item in value["value"]]
    raise HydraHttpError("HydraDB returned an unsupported query value")


def _request_json(method: str, url: str, headers: Mapping[str, str], body: bytes) -> Mapping[str, object]:
    request = Request(url, data=body, headers=dict(headers), method=method)
    try:
        with urlopen(request, timeout=15) as response:  # noqa: S310 - caller configures a local endpoint
            raw = response.read()
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise HydraHttpError(f"HydraDB returned HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise HydraHttpError(f"HydraDB network error: {error.reason}") from error
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HydraHttpError("HydraDB returned invalid JSON") from error
    if not isinstance(decoded, dict):
        raise HydraHttpError("HydraDB returned non-object JSON")
    return decoded
