import asyncio

import bot


def call_async(fn, *args, **kwargs):
    return asyncio.run(fn(*args, **kwargs))


def test_healthz_reports_loaded_contexts():
    data = call_async(bot.healthz)
    assert data["status"] == "ok"
    assert set(data["contexts_loaded"].keys()) == {"category", "merchant", "customer", "trigger"}


def test_context_push_is_idempotent_by_version():
    payload = {"slug": "dentists", "voice": {"tone": "peer_clinical"}}
    first = call_async(bot.push_context, bot.CtxBody(scope="category", context_id="dentists", version=1, payload=payload, delivered_at="2026-07-04T00:00:00Z"))
    second = call_async(bot.push_context, bot.CtxBody(scope="category", context_id="dentists", version=1, payload=payload, delivered_at="2026-07-04T00:00:00Z"))
    assert first["accepted"] is True
    assert second["accepted"] is False
    assert second["reason"] == "stale_version"


def test_compose_handles_merchant_trigger_without_api_keys():
    category = {"slug": "dentists"}
    merchant = {"merchant_id": "m_001", "identity": {"name": "Dr. Meera's Dental Clinic", "owner_first_name": "Meera"}, "offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}]}
    trigger = {"id": "trg_001", "kind": "research_digest", "payload": {"merchant_id": "m_001"}}
    result = bot.compose(category, merchant, trigger)
    assert result["send_as"] == "vera"
    assert result["cta"] in {"binary_yes_no", "binary_confirm_cancel"}
    assert "Dental Cleaning" in result["body"] or "draft" in result["body"].lower()


def test_compose_customer_facing_uses_merchant_on_behalf():
    category = {"slug": "dentists"}
    merchant = {"merchant_id": "m_001", "identity": {"name": "Dr. Meera's Dental Clinic", "owner_first_name": "Meera"}, "offers": [{"title": "Dental Cleaning @ ₹299", "status": "active"}]}
    customer = {"identity": {"name": "Priya", "language_pref": "hi-en mix"}}
    trigger = {"id": "trg_002", "kind": "recall_due", "customer_id": "c_001", "payload": {"merchant_id": "m_001", "customer_id": "c_001"}}
    result = bot.compose(category, merchant, trigger, customer)
    assert result["send_as"] == "merchant_on_behalf"
    assert "Priya" in result["body"]


def test_reply_flow_handles_commit_and_opt_out():
    bot.conversations.clear()
    commit = bot.reply_move("conv_1", "m_001", None, "merchant", "Yes, let\'s do it", 2)
    assert commit["action"] == "send"
    assert commit["cta"] == "binary_confirm_cancel"
    opt_out = bot.reply_move("conv_2", "m_001", None, "merchant", "not interested", 2)
    assert opt_out["action"] == "end"
