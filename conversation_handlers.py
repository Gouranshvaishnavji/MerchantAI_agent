"""
conversation_handlers.py — Multi-turn conversation state machine for Vera.

"""

from dataclasses import dataclass, field
from typing import Optional, Literal
import re
import json


# ─────────────────────────────────────────────
# STATE MODEL
# ─────────────────────────────────────────────

ConversationPhase = Literal[
    "opening",        # First message sent, awaiting first reply
    "qualifying",     # Gathering merchant intent / preference
    "executing",      # Merchant committed — doing the work now
    "awaiting_owner", # Auto-reply detected, waiting for real owner
    "cooling_off",    # Merchant asked for time
    "closing"         # Graceful exit in progress
]


@dataclass
class ConversationState:
    conversation_id:  str
    merchant_id:      str
    customer_id:      Optional[str]
    trigger_id:       Optional[str]
    trigger_kind:     str = ""
    phase:            ConversationPhase = "opening"
    turns:            list[dict] = field(default_factory=list)
    auto_reply_count: int  = 0
    merchant_committed: bool = False
    merchant_opted_out: bool = False
    last_bot_body:    str  = ""
    meta:             dict = field(default_factory=dict)

    def add_turn(self, from_role: str, message: str):
        self.turns.append({"from": from_role, "message": message})

    def last_merchant_messages(self, n: int = 3) -> list[str]:
        return [t["message"] for t in self.turns if t["from"] == "merchant"][-n:]

    def all_bot_bodies(self) -> list[str]:
        return [t["message"] for t in self.turns if t["from"] == "vera"]

    def last_n_turns(self, n: int = 4) -> str:
        lines = []
        for t in self.turns[-n:]:
            lines.append(f"  [{t['from']}]: {t['message'][:140]}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# INTENT CLASSIFIERS
# ─────────────────────────────────────────────

AUTO_REPLY_PATTERNS = [
    r"thank.{0,10}you for contacting",
    r"our team will (get back|respond)",
    r"automated (message|assistant|reply)",
    r"we (have received|received) your (message|query)",
    r"aapki jaankari ke liye",
    r"madad ke liye shukriya.*automated",
    r"main ek automated",
]

COMMIT_PATTERNS = [
    r"\blet'?s? do it\b",
    r"\bok (do it|go|proceed|chalte hain)\b",
    r"\b(yes|haan|ha|bilkul) (please|karo|proceed|let'?s?)\b",
    r"\bconfirm\b",
    r"\bgo ahead\b",
    r"\bshuru karo\b",
    r"\bsend it\b",
    r"\bdraft it\b",
    r"\bdo it\b",
    r"\bwhat'?s? next\b",
]

OPT_OUT_PATTERNS = [
    r"\b(not interested|no thanks|nahi chahiye)\b",
    r"\bstop (messaging|sending|contacting)\b",
    r"\bband karo\b",
    r"\bmat bhejo\b",
    r"\bunsubscribe\b",
    r"\bdo not (message|contact|call)\b",
]

OUT_OF_SCOPE_PATTERNS = [
    r"\bgst (filing|return)\b",
    r"\bincome tax\b",
    r"\bproperty (dispute|loan)\b",
    r"\blegal advice\b",
    r"\binsurance claim\b",
]

HOSTILE_PATTERNS = [
    r"\b(useless|bakwas|rubbish|stupid bot|idiot)\b",
    r"\bwhy (are you|keep) (bothering|messaging)\b",
    r"\bstop (wasting|bothering)\b",
]


def classify_message(message: str) -> str:
    """Returns intent category string."""
    msg_lower = message.lower()

    for p in AUTO_REPLY_PATTERNS:
        if re.search(p, msg_lower):
            return "auto_reply"
    for p in OPT_OUT_PATTERNS:
        if re.search(p, msg_lower):
            return "opt_out"
    for p in HOSTILE_PATTERNS:
        if re.search(p, msg_lower):
            return "hostile"
    for p in OUT_OF_SCOPE_PATTERNS:
        if re.search(p, msg_lower):
            return "out_of_scope"
    for p in COMMIT_PATTERNS:
        if re.search(p, msg_lower):
            return "commit"

    if "?" in message or any(
        w in msg_lower for w in ["kya", "how", "when", "where", "what", "kaisa", "kitna", "kab"]
    ):
        return "question"

    return "engaged"


# ─────────────────────────────────────────────
# PHASE TRANSITIONS
# ─────────────────────────────────────────────

def transition_phase(state: ConversationState, intent: str) -> ConversationPhase:
    """Next phase from current phase + intent."""
    current = state.phase

    if intent == "auto_reply":
        state.auto_reply_count += 1
        if state.auto_reply_count >= 3:
            return "closing"
        return "awaiting_owner"

    if intent == "opt_out":
        state.merchant_opted_out = True
        return "closing"

    if intent == "hostile":
        return "closing"

    if intent == "commit":
        state.merchant_committed = True
        return "executing"

    if intent in ("question", "engaged") and current in ("opening", "qualifying"):
        return "qualifying"

    # Stay in executing once committed
    if current == "executing":
        return "executing"

    return current


# ─────────────────────────────────────────────
# REPLY PROMPT BUILDER
# ─────────────────────────────────────────────

def build_reply_prompt(
    state:            ConversationState,
    merchant_message: str,
    merchant:         dict,
    category:         dict,
    trigger:          dict,
    customer:         Optional[dict],
) -> str:
    """Build a tight, phase-aware reply prompt."""

    intent    = classify_message(merchant_message)
    new_phase = transition_phase(state, intent)
    state.phase = new_phase

    identity  = merchant.get("identity", {})
    owner     = identity.get("owner_first_name", "")
    m_name    = identity.get("name", "")
    langs     = identity.get("languages", ["en"])
    offers    = [o["title"] for o in merchant.get("offers", []) if o.get("status") == "active"]
    cust_agg  = merchant.get("customer_aggregate", {})
    perf      = merchant.get("performance", {})
    signals   = merchant.get("signals", [])
    trg_kind  = state.trigger_kind or trigger.get("kind", "")
    trg_payload = trigger.get("payload", {})

    history = state.last_n_turns(4)
    history += f"\n  [merchant NOW (turn {len(state.turns)+1})]: {merchant_message[:200]}"

    # Phase-specific instructions
    phase_instructions = {
        "executing": (
            "Merchant committed. ACTION mode. "
            "Draft the specific artifact (campaign post / WhatsApp message / booking slot / compliance SOP) right now in the message body. "
            f"Reference real numbers from context: offer='{offers[0] if offers else ''}', "
            f"lapsed={cust_agg.get('lapsed_180d_plus') or cust_agg.get('lapsed_90d_plus') or 0} customers, "
            f"views={perf.get('views',0)}, trigger={trg_kind}. "
            "End with a binary confirm/cancel. Do NOT ask another qualifying question."
        ),
        "awaiting_owner": (
            f"Auto-reply count: {state.auto_reply_count}. "
            "count=1: send one short message prompting the owner to respond. "
            "count=2: action=wait (86400s). "
            "count>=3: action=end."
        ),
        "closing": (
            "Merchant opted out / hostile / too many auto-replies. "
            "action=end (or one polite exit line then end). Do NOT pitch anything."
        ),
        "qualifying": (
            "Merchant is engaged but hasn't committed yet. "
            "Answer their question specifically using merchant data. "
            f"Advance toward one clear commitment. Signal: {signals[:2]}. "
            "Do not ask more than one question per turn."
        ),
        "opening": (
            "First reply received. Acknowledge it, add value, move toward a clear ask."
        ),
        "cooling_off": (
            "Merchant asked for time. Acknowledge, leave a single re-engagement hook, then wait."
        ),
    }

    use_hindi = "hi" in langs

    return f"""\
You are Vera, magicpin's AI assistant. Mid-conversation with merchant. Phase={new_phase}.
Intent detected: {intent}

PHASE INSTRUCTION:
{phase_instructions.get(new_phase, 'Continue helpfully.')}

MERCHANT  : {m_name} | owner={owner} | langs={langs}
OFFERS    : {offers}
TRIGGER   : {trg_kind} | payload={json.dumps(trg_payload, ensure_ascii=False)[:120]}
CUST AGG  : total={cust_agg.get('total_unique_ytd')} lapsed={cust_agg.get('lapsed_180d_plus') or cust_agg.get('lapsed_90d_plus')}
CATEGORY  : {category.get('slug','')} tone={category.get('voice',{}).get('tone','')}
HINDI MIX : {'yes — use natural Hindi-English mix' if use_hindi else 'no'}

CONVERSATION SO FAR:
{history}

RULES:
- Under 100 words.
- One CTA. No URLs.
- DO NOT repeat: "{state.last_bot_body[:80]}"
- action ∈ {{send, wait, end}}
- rationale must match the message (judge cross-checks)

OUTPUT JSON only:
{{"action": "send"|"wait"|"end", "body": "...", "cta": "binary_yes_no"|"open_ended"|"binary_confirm_cancel"|"none", "wait_seconds": 86400, "rationale": "..."}}"""


# ─────────────────────────────────────────────
# MAIN RESPOND FUNCTION (tiebreaker round)
# ─────────────────────────────────────────────

def respond(
    state:            ConversationState,
    merchant_message: str,
    merchant:         dict = None,
    category:         dict = None,
    trigger:          dict = None,
    customer:         dict = None,
) -> dict:
    """
    Multi-turn reply handler for replay/tiebreaker round.
    Given current state + merchant message, returns Vera's next move.
    """
    from bot import call_llm_reply  # avoid circular at module level

    state.add_turn("merchant", merchant_message)
    intent = classify_message(merchant_message)

    # ── Fast-path: opt-out ────────────────────────────────────────────────────
    if intent == "opt_out":
        state.phase = "closing"
        return {
            "action":   "end",
            "rationale": "Merchant explicitly opted out. Conversation closed.",
        }

    # ── Fast-path: auto-reply sequence ───────────────────────────────────────
    if intent == "auto_reply":
        state.auto_reply_count += 1

        if state.auto_reply_count >= 3:
            state.phase = "closing"
            return {"action": "end", "rationale": "3 consecutive auto-replies. Closing."}

        if state.auto_reply_count == 2:
            return {
                "action":       "wait",
                "wait_seconds": 86400,
                "rationale":    "Second auto-reply — owner not present. Wait 24h.",
            }

        # First auto-reply: send one prompt for the owner
        owner      = (merchant or {}).get("identity", {}).get("owner_first_name", "")
        resp_body  = f"Looks like an auto-reply 🙂 When {owner or 'the owner'} is free, just reply YES to continue."
        state.add_turn("vera", resp_body)
        state.last_bot_body = resp_body
        return {
            "action":   "send",
            "body":     resp_body,
            "cta":      "binary_yes_no",
            "rationale": "Auto-reply (first occurrence) — prompting for owner.",
        }

    # ── Fast-path: hostile ────────────────────────────────────────────────────
    if intent == "hostile":
        state.phase = "closing"
        body = "Apologies for the interruption — won't message again. Restart anytime with 'Hi Vera'. 🙏"
        state.add_turn("vera", body)
        return {
            "action":   "send",
            "body":     body,
            "cta":      "none",
            "rationale": "Hostile message — one polite exit.",
        }

    # ── Fast-path: out-of-scope ───────────────────────────────────────────────
    if intent == "out_of_scope":
        owner = (merchant or {}).get("identity", {}).get("owner_first_name", "")
        body  = (
            f"That's outside my scope — for GST/legal, your CA or portal is the right move. "
            f"Back to what we were working on{' ' + owner if owner else ''} — shall we continue?"
        )
        state.add_turn("vera", body)
        state.last_bot_body = body
        return {
            "action":   "send",
            "body":     body,
            "cta":      "binary_yes_no",
            "rationale": "Out-of-scope deflected; redirected to original topic.",
        }

    # ── Fast-path: commit → execution ────────────────────────────────────────
    if intent == "commit" and state.phase != "executing":
        state.merchant_committed = True
        state.phase = "executing"
        owner   = (merchant or {}).get("identity", {}).get("owner_first_name", "")
        offers  = [o["title"] for o in (merchant or {}).get("offers", []) if o.get("status") == "active"]
        cust_agg = (merchant or {}).get("customer_aggregate", {})
        lapsed  = cust_agg.get("lapsed_180d_plus") or cust_agg.get("lapsed_90d_plus") or 0
        offer_hint = f" Draft will use your '{offers[0]}' offer." if offers else ""
        scope_hint = f" Targeting {lapsed} lapsed customers." if lapsed else ""
        name_part  = f"Got it {owner}! " if owner else "Got it! "
        body = f"{name_part}Proceeding now.{offer_hint}{scope_hint} Confirm to send."
        state.add_turn("vera", body)
        state.last_bot_body = body
        return {
            "action":   "send",
            "body":     body,
            "cta":      "binary_confirm_cancel",
            "rationale": "Merchant committed — switched to execution mode with specific scope.",
        }

    # ── LLM-powered reply ─────────────────────────────────────────────────────
    prompt = build_reply_prompt(
        state,
        merchant_message,
        merchant  or {},
        category  or {},
        trigger   or {},
        customer,
    )
    raw    = call_llm_reply(prompt)

    result = {}
    try:
        clean  = re.sub(r"```(?:json)?|```", "", raw).strip()
        result = json.loads(clean)
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
            except Exception:
                pass

    action = result.get("action", "send")
    if action not in {"send", "wait", "end"}:
        action = "send"

    body = result.get("body", "").strip()
    if action == "send":
        body = re.sub(r'https?://\S+', '', body).strip()
        if body:
            state.add_turn("vera", body)
            state.last_bot_body = body

    if action == "end":
        state.phase = "closing"

    response = {"action": action, "rationale": result.get("rationale", "")}
    if action == "send":
        response["body"] = body
        response["cta"]  = result.get("cta", "open_ended")
    elif action == "wait":
        response["wait_seconds"] = result.get("wait_seconds", 86400)

    return response