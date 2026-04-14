"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                                                                  ║
║   ███╗   ███╗ ██████╗ ███╗   ██╗███████╗████████╗███████╗██████╗               ║
║   ████╗ ████║██╔═══██╗████╗  ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗              ║
║   ██╔████╔██║██║   ██║██╔██╗ ██║███████╗   ██║   █████╗  ██████╔╝              ║
║   ██║╚██╔╝██║██║   ██║██║╚██╗██║╚════██║   ██║   ██╔══╝  ██╔══██╗              ║
║   ██║ ╚═╝ ██║╚██████╔╝██║ ╚████║███████║   ██║   ███████╗██║  ██║              ║
║   ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝              ║
║                                                                                  ║
║              ████████╗██████╗  █████╗ ██████╗ ██╗███╗   ██╗ ██████╗            ║
║              ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║██╔════╝            ║
║                 ██║   ██████╔╝███████║██║  ██║██║██╔██╗ ██║██║  ███╗           ║
║                 ██║   ██╔══██╗██╔══██║██║  ██║██║██║╚██╗██║██║   ██║           ║
║                 ██║   ██║  ██║██║  ██║██████╔╝██║██║ ╚████║╚██████╔╝           ║
║                 ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝           ║
║                                                                                  ║
║   ┌─────────────────────────────────────────────────────────────────────────┐   ║
║   │                    🚦  A L E R T   R O U T E R  🚦                      │   ║
║   │              Pipeline Stage 5 — Signal Distribution Engine              │   ║
║   │        news_engine → dup_filter → validator → signal_engine → [YOU]     │   ║
║   └─────────────────────────────────────────────────────────────────────────┘   ║
║                                                                                  ║
║   Module  : alert_router.py                                                      ║
║   Version : 1.0.0                                                                ║
║   Role    : Route high-value signals to correct delivery channels                ║
║   Outputs : terminal | telegram | discord | n8n | human_queue | exec_queue      ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import sys
import textwrap
import traceback
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 ▸ ENUMS & CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

ROUTER_VERSION = "1.0.0"
ROUTER_BUILD   = "MONSTER-TRADING-AI"

# Confidence threshold that promotes a signal to the execution queue
EXECUTION_CONFIDENCE_THRESHOLD = 0.80

# Signal types recognised as actionable (buy/sell)
ACTIONABLE_SIGNALS = frozenset({"BUY", "SELL"})

# Verdict values that classify an article as suspect / needing human review
SUSPECT_VERDICTS  = frozenset({"SUSPECT", "MANUAL_REVIEW", "UNVERIFIED"})

# Verdict values that keep an article on the watchlist only
NO_TRADE_VERDICTS = frozenset({"NO_TRADE", "NOISE", "NEUTRAL"})

# Fields we expect from signal_engine.py — used for safe-skip validation
REQUIRED_FIELDS = {
    "signal_type",       # BUY | SELL | HOLD | NO_TRADE
    "signal_strength",   # CRITICAL | HIGH | MEDIUM | LOW
    "confidence_score",  # float 0-1
    "verdict",           # from fake_news_validator
}


class DeliveryChannel(str, Enum):
    TERMINAL        = "terminal"
    TELEGRAM        = "telegram"
    DISCORD         = "discord"
    N8N_WEBHOOK     = "n8n_webhook"
    HUMAN_REVIEW    = "human_review"
    EXECUTION_QUEUE = "execution_queue"
    WATCHLIST       = "watchlist"


class AlertPriority(str, Enum):
    CRITICAL = "CRITICAL"   # Immediate action — execution candidate
    HIGH     = "HIGH"       # Fast human review
    MEDIUM   = "MEDIUM"     # Standard review
    LOW      = "LOW"        # Watchlist / informational
    NOISE    = "NOISE"      # Discard / log only


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 ▸ CHANNEL FORMATTERS (plugin-ready)
# ──────────────────────────────────────────────────────────────────────────────

class TerminalFormatter:
    """Rich, colour-coded terminal banner for live monitoring."""

    COLOUR = {
        "CRITICAL": "\033[1;31m",   # bold red
        "HIGH":     "\033[1;33m",   # bold yellow
        "MEDIUM":   "\033[1;36m",   # bold cyan
        "LOW":      "\033[0;37m",   # light grey
        "NOISE":    "\033[2;37m",   # dim grey
    }
    RESET  = "\033[0m"
    SIGNAL_ICON = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸", "NO_TRADE": "🚫"}
    CHANNEL_ICON = {
        DeliveryChannel.EXECUTION_QUEUE: "⚡",
        DeliveryChannel.HUMAN_REVIEW:    "🔍",
        DeliveryChannel.WATCHLIST:        "👀",
        DeliveryChannel.TELEGRAM:         "✈️",
        DeliveryChannel.DISCORD:          "💬",
        DeliveryChannel.N8N_WEBHOOK:      "🔗",
        DeliveryChannel.TERMINAL:         "🖥",
    }

    @classmethod
    def format(cls, article: dict, meta: dict) -> str:
        priority   = meta.get("alert_priority", "LOW")
        colour     = cls.COLOUR.get(priority, cls.RESET)
        reset      = cls.RESET
        channel    = meta.get("delivery_channel", "unknown")
        sig_type   = article.get("signal_type", "UNKNOWN")
        sig_icon   = cls.SIGNAL_ICON.get(sig_type, "❓")
        ch_icon    = cls.CHANNEL_ICON.get(channel, "📌")
        ticker     = article.get("ticker", "???")
        headline   = textwrap.shorten(article.get("title", "No title"), 72)
        confidence = article.get("confidence_score", 0.0)
        reason     = meta.get("router_reason", "")
        ts         = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        lines = [
            f"{colour}{'─'*78}{reset}",
            f"{colour}  {sig_icon}  [{priority}]  {ticker}  ▸  {sig_type}  "
            f"(conf: {confidence:.2f})  {ch_icon} → {channel.upper()}{reset}",
            f"  📰  {headline}",
            f"  💡  {reason}",
            f"  🕒  {ts}",
            f"{colour}{'─'*78}{reset}",
        ]
        return "\n".join(lines)


class TelegramFormatter:
    """
    Builds a Telegram Bot API sendMessage payload.
    Plugin hook: pass to telegram_sender.py → requests.post(TELEGRAM_URL, json=payload)
    """

    @staticmethod
    def format(article: dict, meta: dict) -> dict:
        sig_type   = article.get("signal_type", "UNKNOWN")
        ticker     = article.get("ticker", "N/A")
        confidence = article.get("confidence_score", 0.0)
        priority   = meta.get("alert_priority", "LOW")
        channel    = meta.get("delivery_channel", "unknown")
        reason     = meta.get("router_reason", "")
        headline   = textwrap.shorten(article.get("title", ""), 200)
        source     = article.get("source", "unknown")
        exec_cand  = "✅ YES" if meta.get("execution_candidate") else "❌ NO"
        human_conf = "🔍 YES" if meta.get("requires_human_confirmation") else "🤖 NO"

        emoji_map  = {"BUY": "📈", "SELL": "📉", "HOLD": "⏸", "NO_TRADE": "🚫"}
        sig_emoji  = emoji_map.get(sig_type, "❓")

        text = (
            f"🤖 *MONSTER TRADING AI* — Alert\n"
            f"{'━'*30}\n"
            f"{sig_emoji} *Signal:* `{sig_type}` — *{ticker}*\n"
            f"⚠️ *Priority:* `{priority}`\n"
            f"📊 *Confidence:* `{confidence:.2%}`\n"
            f"📡 *Channel:* `{channel}`\n"
            f"⚡ *Exec Candidate:* {exec_cand}\n"
            f"🔍 *Human Review:* {human_conf}\n"
            f"📰 _{headline}_\n"
            f"🔗 *Source:* {source}\n"
            f"💡 _{reason}_"
        )
        return {
            "parse_mode": "Markdown",
            "text": text,
            "disable_web_page_preview": True,
        }


class DiscordFormatter:
    """
    Builds a Discord Webhook embed payload.
    Plugin hook: pass to discord_sender.py → requests.post(WEBHOOK_URL, json=payload)
    """

    COLOUR_MAP = {
        "CRITICAL": 0xFF0000,   # red
        "HIGH":     0xFF8800,   # orange
        "MEDIUM":   0xFFFF00,   # yellow
        "LOW":      0x00BFFF,   # sky blue
        "NOISE":    0x808080,   # grey
    }

    @classmethod
    def format(cls, article: dict, meta: dict) -> dict:
        priority   = meta.get("alert_priority", "LOW")
        sig_type   = article.get("signal_type", "UNKNOWN")
        ticker     = article.get("ticker", "N/A")
        confidence = article.get("confidence_score", 0.0)
        channel    = meta.get("delivery_channel", "unknown")
        reason     = meta.get("router_reason", "")
        headline   = textwrap.shorten(article.get("title", ""), 256)
        source     = article.get("source", "unknown")
        ts         = datetime.now(timezone.utc).isoformat()

        embed = {
            "title":       f"{'📈' if sig_type=='BUY' else '📉' if sig_type=='SELL' else '⏸'} "
                           f"{ticker} — {sig_type}",
            "description": headline,
            "color":       cls.COLOUR_MAP.get(priority, 0x808080),
            "timestamp":   ts,
            "footer":      {"text": f"MONSTER TRADING AI • {ROUTER_VERSION}"},
            "fields": [
                {"name": "Priority",       "value": f"`{priority}`",           "inline": True},
                {"name": "Confidence",     "value": f"`{confidence:.2%}`",     "inline": True},
                {"name": "Channel",        "value": f"`{channel}`",            "inline": True},
                {"name": "Exec Candidate", "value": str(meta.get("execution_candidate")), "inline": True},
                {"name": "Human Review",   "value": str(meta.get("requires_human_confirmation")), "inline": True},
                {"name": "Source",         "value": source,                    "inline": True},
                {"name": "Router Reason",  "value": reason,                    "inline": False},
            ],
        }
        return {"embeds": [embed]}


class N8NFormatter:
    """
    Builds an n8n Webhook node payload.
    Plugin hook: pass to n8n_bridge.py → requests.post(N8N_WEBHOOK_URL, json=payload)
    Compatible with n8n's 'Webhook' trigger node — all fields top-level for easy mapping.
    """

    @staticmethod
    def format(article: dict, meta: dict) -> dict:
        return {
            # ── router metadata ──────────────────────────────────────────────
            "event_id":                  str(uuid.uuid4()),
            "event_type":                "trading_alert",
            "router_version":            ROUTER_VERSION,
            "routed_at":                 datetime.now(timezone.utc).isoformat(),
            # ── signal core ──────────────────────────────────────────────────
            "signal_type":               article.get("signal_type"),
            "signal_strength":           article.get("signal_strength"),
            "confidence_score":          article.get("confidence_score"),
            "ticker":                    article.get("ticker"),
            "verdict":                   article.get("verdict"),
            # ── article metadata ─────────────────────────────────────────────
            "title":                     article.get("title"),
            "source":                    article.get("source"),
            "url":                       article.get("url"),
            "published_at":              article.get("published_at"),
            # ── routing decision ─────────────────────────────────────────────
            "delivery_channel":          meta.get("delivery_channel"),
            "alert_priority":            meta.get("alert_priority"),
            "requires_human_confirmation": meta.get("requires_human_confirmation"),
            "execution_candidate":       meta.get("execution_candidate"),
            "router_reason":             meta.get("router_reason"),
            # ── downstream flags (for n8n IF nodes) ─────────────────────────
            "flag_send_telegram":        meta.get("delivery_channel") == DeliveryChannel.TELEGRAM,
            "flag_send_discord":         meta.get("delivery_channel") == DeliveryChannel.DISCORD,
            "flag_execute":              meta.get("execution_candidate", False),
            "flag_human_review":         meta.get("requires_human_confirmation", False),
        }


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 ▸ ROUTING ENGINE  (O(1) per article via dispatch table)
# ──────────────────────────────────────────────────────────────────────────────

class RoutingRule:
    """Immutable routing decision container."""

    __slots__ = (
        "delivery_channel",
        "alert_priority",
        "requires_human_confirmation",
        "execution_candidate",
        "router_reason",
    )

    def __init__(
        self,
        delivery_channel: DeliveryChannel,
        alert_priority: AlertPriority,
        requires_human_confirmation: bool,
        execution_candidate: bool,
        router_reason: str,
    ) -> None:
        self.delivery_channel             = delivery_channel
        self.alert_priority               = alert_priority
        self.requires_human_confirmation  = requires_human_confirmation
        self.execution_candidate          = execution_candidate
        self.router_reason                = router_reason

    def as_dict(self) -> dict:
        return {
            "delivery_channel":             self.delivery_channel,
            "alert_priority":               self.alert_priority,
            "requires_human_confirmation":  self.requires_human_confirmation,
            "execution_candidate":          self.execution_candidate,
            "router_reason":                self.router_reason,
        }


def _rule_execution_queue(reason: str = "CRITICAL signal with high confidence — execution candidate") -> RoutingRule:
    return RoutingRule(
        delivery_channel            = DeliveryChannel.EXECUTION_QUEUE,
        alert_priority              = AlertPriority.CRITICAL,
        requires_human_confirmation = False,
        execution_candidate         = True,
        router_reason               = reason,
    )


def _rule_human_review(priority: AlertPriority, reason: str) -> RoutingRule:
    return RoutingRule(
        delivery_channel            = DeliveryChannel.HUMAN_REVIEW,
        alert_priority              = priority,
        requires_human_confirmation = True,
        execution_candidate         = False,
        router_reason               = reason,
    )


def _rule_watchlist(reason: str) -> RoutingRule:
    return RoutingRule(
        delivery_channel            = DeliveryChannel.WATCHLIST,
        alert_priority              = AlertPriority.LOW,
        requires_human_confirmation = False,
        execution_candidate         = False,
        router_reason               = reason,
    )


def _rule_noise(reason: str) -> RoutingRule:
    return RoutingRule(
        delivery_channel            = DeliveryChannel.WATCHLIST,
        alert_priority              = AlertPriority.NOISE,
        requires_human_confirmation = False,
        execution_candidate         = False,
        router_reason               = reason,
    )


def _classify_article(article: dict) -> RoutingRule:
    """
    O(1) routing via layered conditional dispatch.
    Priority order (highest first):
      1. Execution queue  — CRITICAL + BUY/SELL + confidence > 0.80
      2. Human review     — SUSPECT / MANUAL_REVIEW verdict
      3. Human review     — HIGH strength actionable with lower confidence
      4. Watchlist        — NO_TRADE / NEUTRAL / NOISE
      5. Telegram/Discord — MEDIUM strength actionable
      6. Watchlist        — everything else
    """
    # Handle fallback field names from signal_engine
    verdict    = str(article.get("verdict") or article.get("verification_status", "")).upper()
    sig_type   = str(article.get("signal_type") or article.get("signal_direction", "")).upper()
    strength   = str(article.get("signal_strength", "")).upper()
    confidence = float(article.get("confidence_score", 0.0))

    is_actionable = sig_type in ACTIONABLE_SIGNALS
    is_suspect    = verdict in SUSPECT_VERDICTS
    is_no_trade   = verdict in NO_TRADE_VERDICTS or sig_type == "NO_TRADE"

    # ── Rule 1: Execution queue ──────────────────────────────────────────────
    if (
        strength == "CRITICAL"
        and is_actionable
        and confidence > EXECUTION_CONFIDENCE_THRESHOLD
        and not is_suspect
    ):
        return _rule_execution_queue(
            f"CRITICAL {sig_type} signal — confidence {confidence:.2%} exceeds "
            f"{EXECUTION_CONFIDENCE_THRESHOLD:.0%} threshold"
        )

    # ── Rule 2: Suspect / fake-news-flagged → mandatory human review ─────────
    if is_suspect:
        return _rule_human_review(
            AlertPriority.HIGH,
            f"Verdict '{verdict}' requires human validation before any action",
        )

    # ── Rule 3: HIGH strength actionable but confidence too low for auto-exec ─
    if strength == "HIGH" and is_actionable and not is_no_trade:
        return _rule_human_review(
            AlertPriority.HIGH,
            f"HIGH {sig_type} signal — confidence {confidence:.2%} below "
            f"auto-exec threshold; escalate to analyst",
        )

    # ── Rule 4: CRITICAL strength but NOT actionable → watchlist + flag ──────
    if strength == "CRITICAL" and not is_actionable:
        return _rule_human_review(
            AlertPriority.MEDIUM,
            f"CRITICAL non-actionable signal ({sig_type}) — monitor for escalation",
        )

    # ── Rule 5: Explicit NO_TRADE / NEUTRAL → watchlist ──────────────────────
    if is_no_trade:
        return _rule_watchlist(
            f"Signal type '{sig_type}' / verdict '{verdict}' — watchlist only"
        )

    # ── Rule 6: MEDIUM strength actionable → cross-channel broadcast ─────────
    if strength == "MEDIUM" and is_actionable:
        # Returns a Telegram route; Discord is added as secondary in final assembly
        return RoutingRule(
            delivery_channel            = DeliveryChannel.TELEGRAM,
            alert_priority              = AlertPriority.MEDIUM,
            requires_human_confirmation = True,
            execution_candidate         = False,
            router_reason               = f"MEDIUM {sig_type} signal — broadcast to analysts via Telegram/Discord",
        )

    # ── Rule 7: LOW strength or unknown → low-priority watchlist ─────────────
    return _rule_watchlist(
        f"LOW/unknown strength '{strength}' or unrecognised signal '{sig_type}' — watchlist"
    )


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 ▸ FORMAT ASSEMBLY
# ──────────────────────────────────────────────────────────────────────────────

def _build_formatted_alert(article: dict, rule: RoutingRule) -> dict:
    """
    Produce all channel-specific formatted payloads in one pass.
    downstream plugins import only the key they need.
    """
    meta = rule.as_dict()
    
    # Use fallback field names for compatibility
    sig_type = article.get("signal_type") or article.get("signal_direction", "UNKNOWN")
    verdict = article.get("verdict") or article.get("verification_status", "UNKNOWN")
    
    return {
        "terminal":  TerminalFormatter.format(article, meta),
        "telegram":  TelegramFormatter.format(article, meta),
        "discord":   DiscordFormatter.format(article, meta),
        "n8n":       N8NFormatter.format(article, meta),
        "raw_signal": {
            "ticker": article.get("ticker"),
            "signal_type": sig_type,
            "signal_strength": article.get("signal_strength"),
            "confidence_score": article.get("confidence_score"),
            "verdict": verdict,
            "title": article.get("title"),
            "source": article.get("source"),
            "url": article.get("url"),
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 ▸ PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def route_alerts(signal_articles: list[dict]) -> list[dict]:
    """
    Route a batch of signal-enriched articles to the correct delivery channel.

    Parameters
    ----------
    signal_articles : list[dict]
        Output from signal_engine.py.  Each dict must contain at minimum:
        signal_type, signal_strength, confidence_score, verdict.
        Malformed entries are safely skipped with a logged warning.

    Returns
    -------
    list[dict]
        One routed alert per valid input article.  Each dict contains:

        delivery_channel            : str   — target channel name
        alert_priority              : str   — CRITICAL | HIGH | MEDIUM | LOW | NOISE
        requires_human_confirmation : bool  — must a human approve before action?
        execution_candidate         : bool  — safe to forward to execution_bridge?
        router_reason               : str   — plain-English routing explanation
        formatted_alert             : dict  — channel payloads (terminal/telegram/discord/n8n/raw)
    """
    routed: list[dict] = []

    for idx, article in enumerate(signal_articles):
        try:
            _validate_article(article, idx)
        except _SkipEntry as exc:
            _warn(f"[SKIP] Entry {idx}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            _warn(f"[SKIP] Entry {idx} unexpected error during validation: {exc}")
            continue

        try:
            rule             = _classify_article(article)
            formatted_alert  = _build_formatted_alert(article, rule)

            routed.append({
                **rule.as_dict(),
                "formatted_alert": formatted_alert,
            })

        except Exception as exc:  # noqa: BLE001
            _warn(f"[SKIP] Entry {idx} routing/formatting failed: {exc}\n{traceback.format_exc()}")
            continue

    return routed


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 ▸ INTERNAL HELPERS
# ──────────────────────────────────────────────────────────────────────────────

class _SkipEntry(ValueError):
    """Raised when an article is structurally invalid and must be skipped."""


def _validate_article(article: dict, idx: int) -> None:
    if not isinstance(article, dict):
        raise _SkipEntry(f"Expected dict, got {type(article).__name__}")
    
    # Check for signal_type (or signal_direction as fallback)
    has_signal_type = "signal_type" in article or "signal_direction" in article
    if not has_signal_type:
        raise _SkipEntry(f"Missing signal_type or signal_direction")
    
    # Check for verdict (or verification_status as fallback)
    has_verdict = "verdict" in article or "verification_status" in article
    if not has_verdict:
        raise _SkipEntry(f"Missing verdict or verification_status")
    
    # Check for confidence_score
    if "confidence_score" not in article:
        raise _SkipEntry(f"Missing confidence_score")
    
    conf = article.get("confidence_score")
    if not isinstance(conf, (int, float)):
        raise _SkipEntry(f"confidence_score must be numeric, got {type(conf).__name__}")


def _warn(msg: str) -> None:
    print(f"\033[1;33m⚠  ALERT_ROUTER WARNING ▸ {msg}\033[0m", file=sys.stderr)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 ▸ STATISTICS / SUMMARY BANNER
# ──────────────────────────────────────────────────────────────────────────────

def _print_routing_summary(results: list[dict], total_input: int) -> None:
    skipped    = total_input - len(results)
    by_channel: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    exec_count = 0
    human_count = 0

    for r in results:
        ch = r.get("delivery_channel", "unknown")
        pr = r.get("alert_priority", "UNKNOWN")
        by_channel[ch]  = by_channel.get(ch, 0) + 1
        by_priority[pr] = by_priority.get(pr, 0) + 1
        if r.get("execution_candidate"):
            exec_count += 1
        if r.get("requires_human_confirmation"):
            human_count += 1

    print("\n\033[1;35m" + "═"*78 + "\033[0m")
    print("\033[1;35m  📊  ALERT ROUTER — ROUTING SUMMARY\033[0m")
    print("\033[1;35m" + "═"*78 + "\033[0m")
    print(f"  Total input   : {total_input}")
    print(f"  Routed        : {len(results)}")
    print(f"  Skipped       : {skipped}")
    print(f"  Exec queue    : \033[1;31m{exec_count}\033[0m")
    print(f"  Human review  : \033[1;33m{human_count}\033[0m")
    print()
    print("  ── By Channel ──────────────────────────────────")
    for ch, n in sorted(by_channel.items(), key=lambda x: -x[1]):
        bar = "█" * min(n * 4, 40)
        print(f"    {ch:<20} {bar} {n}")
    print()
    print("  ── By Priority ─────────────────────────────────")
    for pr, n in sorted(by_priority.items(), key=lambda x: -x[1]):
        print(f"    {pr:<12} {n}")
    print("\033[1;35m" + "═"*78 + "\033[0m\n")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 ▸ PLUGIN REGISTRY (for telegram_sender.py / n8n_bridge.py)
# ──────────────────────────────────────────────────────────────────────────────

# Plugins register here via register_plugin().  They receive (routed_alert: dict)
# and are called in route_alerts_with_plugins().

_PLUGIN_REGISTRY: list[Callable[[dict], None]] = []


def register_plugin(fn: Callable[[dict], None]) -> None:
    """
    Register a downstream plugin function.

    Example (telegram_sender.py):
        from alert_router import register_plugin

        def send_telegram(routed: dict):
            if routed["delivery_channel"] == "telegram":
                payload = routed["formatted_alert"]["telegram"]
                requests.post(TELEGRAM_URL, json={**payload, "chat_id": CHAT_ID})

        register_plugin(send_telegram)
    """
    _PLUGIN_REGISTRY.append(fn)


def route_alerts_with_plugins(signal_articles: list[dict]) -> list[dict]:
    """
    Like route_alerts() but fires all registered plugins for each routed alert.
    Use this entry-point when telegram_sender.py / n8n_bridge.py are loaded.
    """
    results = route_alerts(signal_articles)
    for routed in results:
        for plugin in _PLUGIN_REGISTRY:
            try:
                plugin(routed)
            except Exception as exc:  # noqa: BLE001
                _warn(f"Plugin '{plugin.__name__}' raised: {exc}")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 ▸ SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """
    Comprehensive smoke test covering every routing branch.
    Run via:  python alert_router.py --smoke
    """

    RESET  = "\033[0m"
    GREEN  = "\033[1;32m"
    RED    = "\033[1;31m"
    YELLOW = "\033[1;33m"
    CYAN   = "\033[1;36m"
    PURPLE = "\033[1;35m"

    print(f"\n{PURPLE}{'▓'*78}")
    print("  🧪  ALERT ROUTER — SMOKE TEST SUITE")
    print(f"{'▓'*78}{RESET}\n")

    # ── Test fixtures ────────────────────────────────────────────────────────
    fixtures = [
        # (label, article, expected_channel, expected_exec, expected_human)
        (
            "CRITICAL BUY  conf=0.92 → execution_queue",
            {
                "title": "Fed pivots: emergency rate cut signals bull run",
                "ticker": "SPY",
                "source": "Reuters",
                "url": "https://reuters.com/abc",
                "published_at": "2025-01-01T09:00:00Z",
                "signal_type": "BUY",
                "signal_strength": "CRITICAL",
                "confidence_score": 0.92,
                "verdict": "VERIFIED",
            },
            DeliveryChannel.EXECUTION_QUEUE, True, False,
        ),
        (
            "CRITICAL SELL conf=0.85 → execution_queue",
            {
                "title": "Bank collapse triggers mass liquidation",
                "ticker": "XLF",
                "source": "Bloomberg",
                "url": "https://bloomberg.com/xyz",
                "published_at": "2025-01-01T10:00:00Z",
                "signal_type": "SELL",
                "signal_strength": "CRITICAL",
                "confidence_score": 0.85,
                "verdict": "VERIFIED",
            },
            DeliveryChannel.EXECUTION_QUEUE, True, False,
        ),
        (
            "CRITICAL BUY  conf=0.60 → below threshold → watchlist",
            {
                "title": "Analyst upgrades sector",
                "ticker": "QQQ",
                "source": "CNBC",
                "url": "https://cnbc.com/a",
                "published_at": "2025-01-01T11:00:00Z",
                "signal_type": "BUY",
                "signal_strength": "CRITICAL",
                "confidence_score": 0.60,
                "verdict": "VERIFIED",
            },
            DeliveryChannel.WATCHLIST, False, False,  # low conf → falls through to watchlist
        ),
        (
            "SUSPECT verdict → human_review",
            {
                "title": "Unconfirmed rumour: CEO resigns",
                "ticker": "TSLA",
                "source": "Reddit",
                "url": "https://reddit.com/r/stocks",
                "published_at": "2025-01-01T12:00:00Z",
                "signal_type": "SELL",
                "signal_strength": "HIGH",
                "confidence_score": 0.75,
                "verdict": "SUSPECT",
            },
            DeliveryChannel.HUMAN_REVIEW, False, True,
        ),
        (
            "MANUAL_REVIEW verdict → human_review",
            {
                "title": "Conflicting earnings data",
                "ticker": "AAPL",
                "source": "SeekingAlpha",
                "url": "https://seekingalpha.com/b",
                "published_at": "2025-01-01T13:00:00Z",
                "signal_type": "HOLD",
                "signal_strength": "MEDIUM",
                "confidence_score": 0.55,
                "verdict": "MANUAL_REVIEW",
            },
            DeliveryChannel.HUMAN_REVIEW, False, True,
        ),
        (
            "NO_TRADE signal → watchlist",
            {
                "title": "Market quiet ahead of FOMC",
                "ticker": "IVV",
                "source": "MarketWatch",
                "url": "https://marketwatch.com/c",
                "published_at": "2025-01-01T14:00:00Z",
                "signal_type": "NO_TRADE",
                "signal_strength": "LOW",
                "confidence_score": 0.30,
                "verdict": "NO_TRADE",
            },
            DeliveryChannel.WATCHLIST, False, False,
        ),
        (
            "HIGH BUY conf=0.70 → human_review (below exec threshold)",
            {
                "title": "Strong earnings beat, guidance raised",
                "ticker": "NVDA",
                "source": "Bloomberg",
                "url": "https://bloomberg.com/d",
                "published_at": "2025-01-01T15:00:00Z",
                "signal_type": "BUY",
                "signal_strength": "HIGH",
                "confidence_score": 0.70,
                "verdict": "VERIFIED",
            },
            DeliveryChannel.HUMAN_REVIEW, False, True,
        ),
        (
            "MEDIUM BUY → telegram broadcast",
            {
                "title": "Sector rotation into energy stocks",
                "ticker": "XLE",
                "source": "Barrons",
                "url": "https://barrons.com/e",
                "published_at": "2025-01-01T16:00:00Z",
                "signal_type": "BUY",
                "signal_strength": "MEDIUM",
                "confidence_score": 0.65,
                "verdict": "VERIFIED",
            },
            DeliveryChannel.TELEGRAM, False, True,
        ),
        (
            "Malformed entry (missing fields) → safe skip",
            {
                "title": "Broken article",
                # missing signal_type, signal_strength, confidence_score, verdict
            },
            None, None, None,  # expect skip → not in results
        ),
        (
            "Non-dict entry → safe skip",
            "I am not a dict",  # type: ignore[arg-type]
            None, None, None,
        ),
    ]

    passed = 0
    failed = 0

    for label, article_or_bad, exp_channel, exp_exec, exp_human in fixtures:
        is_skip_test = exp_channel is None

        if is_skip_test:
            results = route_alerts([article_or_bad])  # type: ignore[arg-type]
            ok = len(results) == 0
            status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
            detail = "(correctly skipped)" if ok else f"(expected skip, got {len(results)} result(s))"
        else:
            results = route_alerts([article_or_bad])  # type: ignore[arg-type]
            if not results:
                ok     = False
                status = f"{RED}FAIL{RESET}"
                detail = "no output produced"
            else:
                r = results[0]
                ch_ok    = r["delivery_channel"] == exp_channel
                exec_ok  = (r["execution_candidate"] == exp_exec) if exp_exec is not None else True
                human_ok = (r["requires_human_confirmation"] == exp_human) if exp_human is not None else True
                ok       = ch_ok and exec_ok and human_ok
                status   = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
                details  = []
                if not ch_ok:
                    details.append(f"channel: got {r['delivery_channel']} ≠ {exp_channel}")
                if not exec_ok:
                    details.append(f"exec_candidate: got {r['execution_candidate']} ≠ {exp_exec}")
                if not human_ok:
                    details.append(f"human_confirm: got {r['requires_human_confirmation']} ≠ {exp_human}")
                detail = " | ".join(details) if details else r["router_reason"]

                # Print terminal banner for valid results
                if ok and results and "formatted_alert" in results[0]:
                    print(results[0]["formatted_alert"]["terminal"])

        print(f"  [{status}]  {label}")
        if detail:
            print(f"          {CYAN}↳ {detail}{RESET}")

        if ok:
            passed += 1
        else:
            failed += 1

    # ── Plugin test ───────────────────────────────────────────────────────────
    print(f"\n  {YELLOW}── Plugin Registry Test ────────────────────────────{RESET}")
    plugin_calls = []

    def _mock_plugin(routed: dict) -> None:
        plugin_calls.append(routed["delivery_channel"])

    register_plugin(_mock_plugin)
    route_alerts_with_plugins([
        {
            "title": "Plugin test article",
            "ticker": "PLUGIN",
            "source": "test",
            "url": "https://test.com",
            "published_at": "2025-01-01T17:00:00Z",
            "signal_type": "BUY",
            "signal_strength": "CRITICAL",
            "confidence_score": 0.91,
            "verdict": "VERIFIED",
        }
    ])
    plugin_ok = len(plugin_calls) == 1
    print(f"  [{'PASS' if plugin_ok else 'FAIL'}]  Plugin fired {len(plugin_calls)} time(s) — received channel: {plugin_calls}")

    if plugin_ok:
        passed += 1
    else:
        failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{PURPLE}{'═'*78}")
    print(f"  🏁  SMOKE TEST RESULTS — {passed}/{total} passed   {'✅ ALL CLEAR' if failed==0 else '❌ FAILURES DETECTED'}")
    print(f"{'═'*78}{RESET}\n")

    sys.exit(0 if failed == 0 else 1)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 ▸ CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def _print_startup_banner() -> None:
    print("""
\033[1;32m╔══════════════════════════════════════════════════════════════════════╗
║           🚦  MONSTER TRADING AI — ALERT ROUTER  v1.0.0            ║
║              Pipeline Stage 5 — Signal Distribution Engine          ║
╚══════════════════════════════════════════════════════════════════════╝\033[0m
""")


if __name__ == "__main__":
    if "--smoke" in sys.argv or "-s" in sys.argv:
        _smoke_test()
    else:
        _print_startup_banner()
        print("Usage:")
        print("  python alert_router.py --smoke          # run smoke test")
        print()
        print("API usage:")
        print("  from alert_router import route_alerts")
        print("  results = route_alerts(signal_articles)")
        print()
        print("Plugin usage:")
        print("  from alert_router import register_plugin, route_alerts_with_plugins")
        print("  register_plugin(my_telegram_sender)")
        print("  results = route_alerts_with_plugins(signal_articles)")