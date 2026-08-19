"""Forward-only, checksum-verified SQL migration runner."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol


class MigrationError(RuntimeError):
    """Raised for invalid migration layout or checksum drift."""


class Cursor(Protocol):
    def execute(self, query: str, params: tuple[object, ...] | None = None) -> object: ...

    def fetchone(self) -> tuple[str] | None: ...

    def __enter__(self) -> Cursor: ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...


class Connection(Protocol):
    def cursor(self) -> Cursor: ...

    def transaction(self) -> object: ...


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    checksum: str

    @property
    def sql(self) -> str:
        return self.path.read_text(encoding="utf-8")


MIGRATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def discover_migrations(directory: Path) -> tuple[Migration, ...]:
    paths = sorted(directory.glob("[0-9][0-9][0-9][0-9]_*.sql"))
    migrations: list[Migration] = []
    versions: set[str] = set()
    for path in paths:
        version = path.name.split("_", maxsplit=1)[0]
        if version in versions:
            raise MigrationError(f"duplicate migration version: {version}")
        versions.add(version)
        checksum = sha256(path.read_bytes()).hexdigest()
        migrations.append(Migration(version=version, path=path, checksum=checksum))
    return tuple(migrations)


def apply_migrations(connection: Connection, directory: Path) -> tuple[str, ...]:
    """Apply new SQL files in order; applied files must retain their checksum."""
    migrations = discover_migrations(directory)
    applied: list[str] = []
    with connection.transaction():
        with connection.cursor() as cursor:
            cursor.execute(MIGRATION_TABLE_SQL)
            for migration in migrations:
                cursor.execute(
                    "SELECT checksum FROM schema_migrations WHERE version = %s",
                    (migration.version,),
                )
                existing = cursor.fetchone()
                if existing is not None:
                    if existing[0] != migration.checksum:
                        raise MigrationError(f"checksum changed for applied migration {migration.version}")
                    continue
                cursor.execute(migration.sql)
                cursor.execute(
                    "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                    (migration.version, migration.checksum),
                )
                applied.append(migration.version)
    return tuple(applied)
