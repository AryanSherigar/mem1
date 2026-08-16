from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path

from context_memory.persistence.migrations import MigrationError, apply_migrations, discover_migrations

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "db" / "migrations"


class FakeCursor:
    def __init__(self, checksums: dict[str, str]) -> None:
        self.checksums = checksums
        self._selected: tuple[str] | None = None
        self.executed: list[str] = []

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        self.executed.append(query)
        if query.startswith("SELECT checksum"):
            assert params is not None
            checksum = self.checksums.get(str(params[0]))
            self._selected = (checksum,) if checksum else None
        elif query.startswith("INSERT INTO schema_migrations"):
            assert params is not None
            self.checksums[str(params[0])] = str(params[1])

    def fetchone(self) -> tuple[str] | None:
        return self._selected

    def __enter__(self) -> FakeCursor:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.checksums: dict[str, str] = {}
        self.cursor_instance = FakeCursor(self.checksums)

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def transaction(self):
        return nullcontext()


class MigrationTests(unittest.TestCase):
    def test_migrations_are_discovered(self) -> None:
        migrations = discover_migrations(MIGRATIONS)
        self.assertEqual([migration.version for migration in migrations], ["0001", "0002", "0003"])
        self.assertIn("evidence_chunks", migrations[0].sql)
        self.assertIn("extraction_attempts", migrations[1].sql)
        self.assertIn("graph_write_manifests", migrations[2].sql)

    def test_apply_is_idempotent(self) -> None:
        connection = FakeConnection()
        self.assertEqual(apply_migrations(connection, MIGRATIONS), ("0001", "0002", "0003"))
        self.assertEqual(apply_migrations(connection, MIGRATIONS), ())

    def test_checksum_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            migration = directory / "0001_example.sql"
            migration.write_text("SELECT 1;", encoding="utf-8")
            connection = FakeConnection()
            apply_migrations(connection, directory)
            migration.write_text("SELECT 2;", encoding="utf-8")
            with self.assertRaisesRegex(MigrationError, "checksum changed"):
                apply_migrations(connection, directory)

    def test_duplicate_version_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            (directory / "0001_first.sql").write_text("SELECT 1;", encoding="utf-8")
            (directory / "0001_second.sql").write_text("SELECT 2;", encoding="utf-8")
            with self.assertRaisesRegex(MigrationError, "duplicate migration version"):
                discover_migrations(directory)
