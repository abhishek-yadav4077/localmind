"""FeedbackService: local-first, LangSmith best effort."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from localmind.chat import ChatService
from localmind.exceptions import LocalMindError
from localmind.tracing import Tracer, TracingStatus, disabled_tracer
from localmind.tracing.feedback import FeedbackService, MessageNotFoundError
from tests.fakes import FakeFactory


def make(db, settings, client: MagicMock | None):
    if client is None:
        tracer = disabled_tracer(settings)
    else:
        tracer = Tracer(TracingStatus(True, "p", "ok", client=client), settings)
    chat = ChatService(db, settings, llm_factory=FakeFactory(), tracer=tracer)
    conv = chat.create_conversation()
    msg = chat.reply(conv.id, "hi")
    return chat, FeedbackService(db, tracer), msg


def test_feedback_saved_locally_without_langsmith(db, settings) -> None:
    _, fb, msg = make(db, settings, None)
    assert msg.run_id is None
    out = fb.submit(msg.id, thumbs_up=True, comment="  nice ")
    assert out.score == 1 and out.comment == "nice"
    assert not out.synced_to_langsmith and out.sync_error is None
    assert fb.get(msg.id).score == 1


def test_feedback_synced_and_vote_change_updates(db, settings) -> None:
    client = MagicMock()
    _, fb, msg = make(db, settings, client)
    assert msg.run_id is not None

    first = fb.submit(msg.id, thumbs_up=True)
    assert first.synced_to_langsmith
    assert client.create_feedback.call_args.args[0] == msg.run_id

    second = fb.submit(msg.id, thumbs_up=False, comment="wrong")
    assert second.langsmith_feedback_id == first.langsmith_feedback_id
    client.update_feedback.assert_called_once()
    assert client.create_feedback.call_count == 1
    assert fb.get(msg.id).score == 0  # still one row per message


def test_langsmith_failure_keeps_local_vote(db, settings) -> None:
    client = MagicMock()
    client.create_feedback.side_effect = RuntimeError("401 unauthorized")
    _, fb, msg = make(db, settings, client)
    out = fb.submit(msg.id, thumbs_up=False)
    assert not out.synced_to_langsmith
    assert "401" in (out.sync_error or "")
    assert fb.get(msg.id).score == 0


def test_feedback_validation(db, settings) -> None:
    chat, fb, msg = make(db, settings, None)
    with pytest.raises(MessageNotFoundError):
        fb.submit(9999, True)
    user_msg = chat.get_messages(msg.conversation_id)[0]
    with pytest.raises(LocalMindError):
        fb.submit(user_msg.id, True)


def test_deleting_conversation_removes_feedback(db, settings) -> None:
    chat, fb, msg = make(db, settings, None)
    fb.submit(msg.id, True)
    chat.delete_conversation(msg.conversation_id)
    assert fb.get(msg.id) is None
