# Ingestion Package Structure

**Status:** Living. Scope: ingestion only; retrieval/chat remain untouched.

```text
src/context_memory/
  core/                 # application-owned contracts and validation routes
  ingestion/            # generic ingest, graph-write orchestration, source adapters
    sources/            # LongMemEval and future source-specific mapping
  persistence/          # PostgreSQL evidence/audit implementation route
  client/               # local HydraDB HTTP client route
```

This follows the upstream separation where relevant: application-owned core
contracts, public client transport, and query/mutation orchestration. It does
not copy upstream `engine`, `shard`, `sparse_kernel`, writer leases, or indexer
folders: those are HydraDB implementation authority, not ingestion-app code.
Retrieval and chat packages are intentionally not introduced or moved here.

Legacy `domain/`, `application/`, `adapters/`, and `query/` folders were
removed. Imports, scripts, and tests use this structure directly.
