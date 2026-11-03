from localmind.db import Database
from localmind.db import repositories as repo


def test_delete_conversation_cascades_messages(db: Database) -> None:
    with db.session() as s:
        conv = repo.create_conversation(s, model="llama3.2")
        repo.add_message(s, conv.id, "user", "hi")
        cid = conv.id
    with db.session() as s:
        assert repo.delete_conversation(s, cid)
    with db.session() as s:
        assert repo.list_messages(s, cid) == []
        assert not repo.delete_conversation(s, cid)


def test_search_matches_title_or_message_content(db: Database) -> None:
    with db.session() as s:
        a = repo.create_conversation(s, model="m", title="Pandas tips")
        b = repo.create_conversation(s, model="m", title="Untitled")
        repo.add_message(s, b.id, "user", "How do I use PANDAS groupby?")
        repo.create_conversation(s, model="m", title="Other")
        ids = {a.id, b.id}
    with db.session() as s:
        assert {c.id for c in repo.list_conversations(s, query="pandas")} == ids
        assert len(repo.list_conversations(s)) == 3


def test_list_messages_after_id(db: Database) -> None:
    with db.session() as s:
        conv = repo.create_conversation(s, model="m")
        first = repo.add_message(s, conv.id, "user", "1")
        repo.add_message(s, conv.id, "assistant", "2")
        cid, fid = conv.id, first.id
    with db.session() as s:
        assert [m.content for m in repo.list_messages(s, cid, after_id=fid)] == ["2"]
