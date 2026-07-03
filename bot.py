"""
Vera - magicpin Merchant AI Assistant
Offline-safe FastAPI server for the judge harness.

This version keeps the required endpoints, but it does not call any external LLM
by default. That keeps local tests cheap and avoids burning the Gemini quota.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

GOOGLE_MODEL = "gemini-2.5-flash"
TEAM_NAME = os.getenv("TEAM_NAME", "Vera Dheera Soora")
TEAM_MEMBERS = os.getenv("TEAM_MEMBERS", "Candidate")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "candidate@example.com")
BOT_VERSION = "5.0.0"

app = FastAPI(title="MagicPin Vera Bot", version=BOT_VERSION)
START_TIME = time.time()

contexts: dict[tuple[str, str], dict[str, Any]] = {}
conversations: dict[str, dict[str, Any]] = {}


class TrafficTracker:
    def __init__(self) -> None:
        self.history: list[tuple[float, int]] = []

    def log_request(self, estimated_tokens: int) -> None:
        self.history.append((time.time(), estimated_tokens))
        self.clean()

    def clean(self) -> None:
        cutoff = time.time() - 60
        self.history = [item for item in self.history if item[0] >= cutoff]

    def get_stats(self) -> tuple[int, int]:
        self.clean()
        rpm = len(self.history)
        tpm = sum(item[1] for item in self.history)
        return rpm, tpm


tracker = TrafficTracker()


class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: str = ""


class TickBody(BaseModel):
    now: str
    available_triggers: list[str] = []


class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: str = ""
    turn_number: int = 1


class HealthzResponse(BaseModel):
    status: str
    uptime_seconds: int
    contexts_loaded: dict[str, int]


VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ctx(scope: str, context_id: str) -> Optional[dict[str, Any]]:
    entry = contexts.get((scope, context_id))
    return entry["payload"] if entry else None


def get_category(slug: str) -> Optional[dict[str, Any]]:
    return _ctx("category", slug)


def get_merchant(merchant_id: str) -> Optional[dict[str, Any]]:
    return _ctx("merchant", merchant_id)


def get_customer(customer_id: str) -> Optional[dict[str, Any]]:
    return _ctx("customer", customer_id)


def get_trigger(trigger_id: str) -> Optional[dict[str, Any]]:
    return _ctx("trigger", trigger_id)


def is_auto_reply(message: str) -> bool:
    msg = message.lower()
    patterns = [
        "thank you for contacting",
        "thanks for contacting",
        "our team will respond",
        "will get back to you",
        "automated assistant",
        "we have received your message",
        "aapki jaankari ke liye",
        "this is an automated",
        "out of office",
        "out of the office",
        "currently unavailable",
        "outside business hours",
    ]
    return any(pattern in msg for pattern in patterns)


def classify_message(message: str) -> str:
    msg = message.lower().strip()
    if is_auto_reply(message):
        return "auto_reply"
    if any(word in msg for word in ["not interested", "no thanks", "unsubscribe", "stop messaging", "mat bhejo", "nahi chahiye"]):
        return "opt_out"
    if any(word in msg for word in ["useless", "bakwas", "rubbish", "stupid bot", "idiot", "stop bothering"]):
        return "hostile"
    if any(word in msg for word in ["gst", "income tax", "loan", "insurance", "property", "legal advice"]):
        return "out_of_scope"
    if any(word in msg for word in ["yes", "go ahead", "let's do it", "lets do it", "confirm", "send it", "draft it", "shuru karo", "proceed"]):
        return "commit"
    if "?" in msg:
        return "question"
    return "engaged"


def _merchant_name(merchant: dict[str, Any]) -> str:
    return merchant.get("identity", {}).get("name", "Merchant")


def _owner_name(merchant: dict[str, Any]) -> str:
    return merchant.get("identity", {}).get("owner_first_name", "")


def _active_offer_title(merchant: dict[str, Any]) -> str:
    offers = [offer for offer in merchant.get("offers", []) if offer.get("status") == "active"]
    return offers[0].get("title", "") if offers else ""


def _conversation_key(merchant_id: str, trigger_id: str) -> str:
    digest = hashlib.md5(f"{merchant_id}:{trigger_id}".encode()).hexdigest()[:8]
    return f"conv_{merchant_id}_{trigger_id}_{digest}"


def compose(category: dict, merchant: dict, trigger: dict, customer: dict | None = None) -> dict:
    category_slug = category.get("slug", "")
    merchant_name = _merchant_name(merchant)
    owner_name = _owner_name(merchant)
    offer_title = _active_offer_title(merchant)
    trigger_kind = trigger.get("kind", "update")
    trigger_id = trigger.get("id", "trigger")
    customer_name = (customer or {}).get("identity", {}).get("name", "")
    customer_lang = (customer or {}).get("identity", {}).get("language_pref", "en")

    send_as = "merchant_on_behalf" if customer else "vera"
    suppression_key = trigger.get("suppression_key") or f"{merchant.get('merchant_id', 'unknown')}:{trigger_id}"

    if customer:
        body = (
            f"Hi {customer_name}, {merchant_name} here. "
            f"{offer_title or 'Your next booking'} is ready. Reply YES to continue."
        )
        cta = "binary_yes_no"
    elif trigger_kind in {"recall_due", "appointment_tomorrow", "trial_followup"}:
        body = (
            f"{owner_name + ', ' if owner_name else ''}{merchant_name} has a new {trigger_kind.replace('_', ' ')}. "
            f"I can draft the next step around {offer_title or 'your active offer'}. Reply YES to send it."
        )
        cta = "binary_yes_no"
    elif trigger_kind in {"commit", "active_planning_intent", "wedding_package_followup", "corporate_thali_planning"}:
        body = (
            f"{owner_name + ', ' if owner_name else ''}I’ve drafted the plan for {offer_title or 'this campaign'}. "
            f"Reply YES and I’ll send it."
        )
        cta = "binary_confirm_cancel"
    else:
        body = (
            f"{merchant_name}, I spotted a {trigger_kind.replace('_', ' ')} for {category_slug or 'your business'}. "
            f"I can draft a {offer_title or 'fresh offer'} message now. Reply YES to continue."
        )
        cta = "binary_yes_no"

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "suppression_key": suppression_key,
        "rationale": f"Deterministic heuristic for {trigger_kind}; grounded in current merchant/customer context.",
    }


def reply_move(conversation_id: str, merchant_id: str, customer_id: str | None, from_role: str, message: str, turn_number: int) -> dict:
    conv = conversations.setdefault(conversation_id, {"turns": [], "ended": False})
    conv["turns"].append({"from": from_role, "message": message, "ts": now_iso()})

    if conv["ended"]:
        return {"action": "end", "rationale": "Conversation previously ended"}

    intent = classify_message(message)
    merchant = get_merchant(merchant_id) or {}
    customer = get_customer(customer_id) if customer_id else None
    owner_name = _owner_name(merchant)
    offer_title = _active_offer_title(merchant)

    if intent == "auto_reply":
        if len([turn for turn in conv["turns"] if turn["from"] == "merchant" and is_auto_reply(turn["message"])]) >= 3:
            conv["ended"] = True
            return {"action": "end", "rationale": "3 auto-replies detected; closing."}
        body = f"Looks like an auto-reply. {owner_name or 'Owner'} can reply YES when free."
        conv["turns"].append({"from": "vera", "message": body, "ts": now_iso()})
        return {"action": "send", "body": body, "cta": "binary_yes_no", "rationale": "First auto-reply: one prompt then wait."}

    if intent == "opt_out":
        conv["ended"] = True
        return {"action": "end", "rationale": "Merchant opted out."}

    if intent == "hostile":
        conv["ended"] = True
        body = "Sorry for the interruption. I won’t message again."
        conv["turns"].append({"from": "vera", "message": body, "ts": now_iso()})
        return {"action": "send", "body": body, "cta": "none", "rationale": "Hostile message: one polite exit."}

    if intent == "commit":
        conv["ended"] = True
        body = f"Got it{', ' + owner_name if owner_name else ''}. I’ve moved into execution for {offer_title or 'your campaign'}."
        conv["turns"].append({"from": "vera", "message": body, "ts": now_iso()})
        return {"action": "send", "body": body, "cta": "binary_confirm_cancel", "rationale": "Commit detected: switch directly to execution."}

    if intent == "out_of_scope":
        body = f"That’s outside my scope. Back to {offer_title or 'your campaign'} - shall we continue?"
        conv["turns"].append({"from": "vera", "message": body, "ts": now_iso()})
        return {"action": "send", "body": body, "cta": "binary_yes_no", "rationale": "Out-of-scope redirected to the business goal."}

    if customer:
        body = f"Thanks {customer.get('identity', {}).get('name', 'there')} - {merchant.get('identity', {}).get('name', 'the clinic')} can help with that. Reply YES to continue."
        cta = "binary_yes_no"
    elif intent == "question":
        body = f"{owner_name + ', ' if owner_name else ''}I can answer that and draft the next step around {offer_title or 'your active offer'}. Reply YES."
        cta = "binary_yes_no"
    else:
        body = f"{merchant.get('identity', {}).get('name', 'Merchant')}, I can draft the next message around {offer_title or 'your active offer'}. Reply YES to continue."
        cta = "binary_yes_no"

    conv["turns"].append({"from": "vera", "message": body, "ts": now_iso()})
    return {"action": "send", "body": body, "cta": cta, "rationale": f"Handled {intent} with a deterministic reply."}


@app.get("/v1/stats")
async def get_traffic_stats() -> dict[str, Any]:
    rpm, tpm = tracker.get_stats()
    return {"rpm": rpm, "tpm": tpm, "history_count": len(tracker.history), "uptime": time.time() - START_TIME}


@app.get("/v1/healthz")
async def healthz() -> dict[str, Any]:
    counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    for scope, _context_id in contexts:
        counts[scope] = counts.get(scope, 0) + 1
    return {"status": "ok", "uptime_seconds": int(time.time() - START_TIME), "contexts_loaded": counts}


@app.get("/v1/metadata")
async def metadata() -> dict[str, Any]:
    return {
        "team_name": TEAM_NAME,
        "team_members": [member.strip() for member in TEAM_MEMBERS.split(",") if member.strip()],
        "contact_email": CONTACT_EMAIL,
        "model": GOOGLE_MODEL,
        "approach": "deterministic offline composer for local testing",
        "version": BOT_VERSION,
        "submitted_at": now_iso(),
    }


@app.post("/v1/context")
async def push_context(body: CtxBody) -> dict[str, Any]:
    if body.scope not in VALID_SCOPES:
        return {"accepted": False, "reason": "invalid_scope", "details": f"{body.scope} is not supported"}

    key = (body.scope, body.context_id)
    current = contexts.get(key)
    if current and current["version"] >= body.version:
        return {"accepted": False, "reason": "stale_version", "current_version": current["version"]}

    contexts[key] = {"version": body.version, "payload": body.payload, "delivered_at": body.delivered_at or now_iso()}
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": now_iso()}


@app.post("/v1/tick")
async def tick(body: TickBody) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    for trigger_id in body.available_triggers[:20]:
        trigger = get_trigger(trigger_id)
        if not trigger:
            continue
        merchant_id = trigger.get("merchant_id") or trigger.get("payload", {}).get("merchant_id")
        customer_id = trigger.get("customer_id") or trigger.get("payload", {}).get("customer_id")
        merchant = get_merchant(merchant_id) if merchant_id else None
        if not merchant:
            continue
        category_slug = merchant.get("category_slug") or merchant.get("identity", {}).get("category_slug")
        category = get_category(category_slug) if category_slug else {}
        customer = get_customer(customer_id) if customer_id else None
        composed = compose(category or {}, merchant, trigger, customer)
        conversation_id = _conversation_key(merchant.get("merchant_id", merchant_id or "unknown"), trigger_id)
        actions.append(
            {
                "conversation_id": conversation_id,
                "merchant_id": merchant.get("merchant_id", merchant_id or "unknown"),
                "customer_id": customer_id,
                "send_as": composed["send_as"],
                "trigger_id": trigger_id,
                "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
                "template_params": [merchant.get("identity", {}).get("name", "Merchant")],
                "body": composed["body"],
                "cta": composed["cta"],
                "suppression_key": composed["suppression_key"],
                "rationale": composed["rationale"],
            }
        )
    return {"actions": actions}


@app.post("/v1/reply")
async def reply(body: ReplyBody) -> dict[str, Any]:
    result = reply_move(body.conversation_id, body.merchant_id or "", body.customer_id, body.from_role, body.message, body.turn_number)
    if result["action"] == "end":
        conversations.setdefault(body.conversation_id, {"turns": [], "ended": True})["ended"] = True
    return result


@app.post("/v1/teardown")
async def teardown() -> dict[str, Any]:
    contexts.clear()
    conversations.clear()
    return {"status": "ok", "message": "State wiped"}


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "Vera bot"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("bot:app", host=host, port=port, log_level="info")
