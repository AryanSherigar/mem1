#!/usr/bin/env python3
"""Ingest local LongMemEval JSON through the generic IngestionService."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg

from context_memory.ingestion.sources.longmemeval import adapt_longmemeval_instance
from context_memory.persistence.postgres import PostgresChunkStore
from context_memory.ingestion.service import IngestionService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to local LongMemEval JSON array")
    parser.add_argument("--database-url", default=os.environ.get("CONTEXT_MEMORY_DATABASE_URL"))
    parser.add_argument("--limit", type=int, help="Maximum benchmark instances to process")
    parser.add_argument("--dry-run", action="store_true", help="Validate/map only; never opens a database")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--input must contain a JSON array of LongMemEval instances")
    instances = payload[: args.limit] if args.limit is not None else payload
    batches = [adapt_longmemeval_instance(instance, f"longmemeval-run:{index}") for index, instance in enumerate(instances)]
    if args.dry_run:
        print(f"validated_batches={len(batches)} validated_records={sum(len(batch.records) for batch in batches)}")
        return 0
    if not args.database_url:
        raise ValueError("--database-url or CONTEXT_MEMORY_DATABASE_URL is required unless --dry-run is used")
    with psycopg.connect(args.database_url) as connection:
        service = IngestionService(PostgresChunkStore(connection))
        results = [service.ingest(batch) for batch in batches]
    print(f"ingested_batches={len(results)} ingested_records={sum(result.accepted_record_count for result in results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
