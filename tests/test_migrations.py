"""Older databases gain new nullable columns on startup."""

from sqlalchemy import create_engine, inspect, text

from localmind.db import Database


def test_add_missing_columns_on_old_schema(tmp_path) -> None:
    path = tmp_path / "old.db"
    # Simulate a Phase-4 database whose messages table lacks the `details` column.
    db = Database(f"sqlite:///{path}")
    db.create_all()
    with db.engine.begin() as conn:
        conn.execute(text("ALTER TABLE messages DROP COLUMN details"))
    db.engine.dispose()

    Database(f"sqlite:///{path}").create_all()
    cols = {c["name"] for c in inspect(create_engine(f"sqlite:///{path}")).get_columns("messages")}
    assert "details" in cols
