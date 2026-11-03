"""Engine/session management."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from localmind.db.models import Base


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - trivial
    cursor = dbapi_conn.cursor()
    # SQLite ignores FOREIGN KEY constraints (incl. ON DELETE CASCADE) unless enabled.
    cursor.execute("PRAGMA foreign_keys=ON")
    # WAL lets the Streamlit reader and the API writer work concurrently.
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


class Database:
    """Owns the engine and session factory.

    Args:
        url: SQLAlchemy URL, e.g. ``sqlite:///data/localmind.db`` or ``sqlite://`` (memory).
    """

    def __init__(self, url: str) -> None:
        parsed = make_url(url)
        kwargs: dict = {}
        if parsed.get_backend_name() == "sqlite":
            # FastAPI serves requests from a thread pool; SQLite connections are
            # thread-bound by default.
            kwargs["connect_args"] = {"check_same_thread": False}
            if parsed.database in (None, "", ":memory:"):
                kwargs["poolclass"] = StaticPool  # one shared in-memory DB (tests)
            else:
                Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(url, **kwargs)
        if parsed.get_backend_name() == "sqlite":
            event.listen(self.engine, "connect", _sqlite_pragmas)
        # expire_on_commit=False: objects stay readable after the session closes,
        # which we need because services return ORM objects to the API layer.
        self._factory = sessionmaker(self.engine, expire_on_commit=False)

    def create_all(self) -> None:
        """Create missing tables, then add missing nullable columns.

        ``create_all`` never alters existing tables, so a column added in a later
        version would crash older databases. ``add_missing_columns`` covers the
        additive case. Anything more (renames, type changes) needs Alembic,
        which is listed as future work.
        """
        Base.metadata.create_all(self.engine)
        add_missing_columns(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commit on success, rollback on error."""
        session = self._factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


def add_missing_columns(engine: Engine) -> list[str]:
    """``ALTER TABLE ... ADD COLUMN`` for nullable model columns missing in the DB."""
    inspector = inspect(engine)
    added: list[str] = []
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            existing = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                if not column.nullable:  # can't add NOT NULL without a default
                    raise RuntimeError(
                        f"Column {table.name}.{column.name} is NOT NULL; write a migration"
                    )
                ddl = column.type.compile(dialect=engine.dialect)
                conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {ddl}'))
                added.append(f"{table.name}.{column.name}")
    return added
