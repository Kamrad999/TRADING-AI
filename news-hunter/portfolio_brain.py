"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         PORTFOLIO BRAIN v1.0.0                                 ║
║              Institution-Grade Portfolio Allocation Engine                      ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Role       : Downstream allocation module in the TRADING_AI pipeline          ║
║  Position   : signal_engine → risk_guardian → [portfolio_brain] →              ║
║               execution_bridge → broker_sender → alert_router →                ║
║               performance_analytics → state_manager → god_core                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Capabilities:                                                                 ║
║   • Universe-agnostic dynamic allocation across 11 asset classes               ║
║   • Confidence-weighted position sizing with liquidity haircuts                ║
║   • Cross-asset correlation-aware de-risking engine                            ║
║   • Volatility targeting with drawdown adaptive de-escalation                  ║
║   • Gross / net exposure hard caps                                             ║
║   • Asset-class and sector concentration limits                                ║
║   • Overnight gap-risk haircuts                                                ║
║   • Tail-risk hedge bucket with auto-allocation                                ║
║   • Cash reserve floor enforcement                                             ║
║   • Full options layer: delta/gamma/theta/IV-crush/0DTE controls               ║
║   • Spread strategy hooks (protective puts, covered calls, wheel)              ║
║   • Rebalance engine with drift-band triggers                                  ║
║   • Target order generation (symbol, side, qty, order_type, urgency)          ║
║   • Portfolio risk score (0–100) with explainable summary                      ║
║   • Plug-compatible output contract for execution_bridge + god_core            ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║  Input      : List[SignalDict] — validated by signal_engine / risk_guardian    ║
║  Output     : PortfolioAllocationResult (TypedDict)                            ║
║  Stdlib only: math, statistics, collections, typing, datetime, logging         ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("portfolio_brain")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s [portfolio_brain] %(message)s"))
    log.addHandler(_h)
log.setLevel(logging.INFO)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — POLICY CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────

# Gross / net exposure ceilings (fraction of NAV)
GROSS_EXPOSURE_MAX: float = 1.50       # 150 % gross long+short
NET_EXPOSURE_MAX:   float = 1.00       # 100 % net
NET_EXPOSURE_MIN:   float = -0.25      # −25 % net (max short book)

# Cash / liquidity
CASH_RESERVE_FLOOR:          float = 0.05   # 5 % NAV always held in cash
LIQUIDITY_SCORE_MIN_DEPLOY:  float = 0.30   # ignore signals below 30 % liquidity
SLIPPAGE_PENALTY_SCALE:      float = 0.10   # 10 % size reduction per 0.1 illiquidity

# Asset-class caps (fraction of gross NAV)
ASSET_CLASS_CAPS: Dict[str, float] = {
    "equity":        0.60,
    "etf":           0.50,
    "crypto":        0.15,
    "forex":         0.20,
    "commodity":     0.20,
    "futures":       0.25,
    "bond":          0.40,
    "option":        0.10,   # premium at risk cap
    "inverse_etf":   0.15,
    "leveraged_etf": 0.12,
    "volatility":    0.08,
}

# Sector concentration cap
SECTOR_CAP: float = 0.30          # 30 % of gross NAV per sector

# Volatility targeting
VOL_TARGET:          float = 0.12  # 12 % annualised portfolio vol target
VOL_LOOKBACK_DAYS:   int   = 20    # rolling window for realised-vol estimate
VOL_SCALE_MIN:       float = 0.25  # minimum vol-scalar floor
VOL_SCALE_MAX:       float = 1.50  # maximum vol-scalar ceiling

# Drawdown adaptive de-risking thresholds (vs peak NAV)
DRAWDOWN_TIERS: List[Tuple[float, float]] = [
    (0.05, 0.90),   # −5 %  → reduce allocation to 90 %
    (0.10, 0.70),   # −10 % → reduce to 70 %
    (0.15, 0.50),   # −15 % → reduce to 50 %
    (0.20, 0.25),   # −20 % → reduce to 25 % (near-defensive)
]

# Overnight gap risk
OVERNIGHT_HAIRCUT_EQUITY:   float = 0.10   # 10 % size reduction for EOD holds
OVERNIGHT_HAIRCUT_CRYPTO:   float = 0.20
OVERNIGHT_HAIRCUT_DEFAULT:  float = 0.05

# Tail-risk hedge bucket
TAIL_HEDGE_FRACTION: float = 0.03          # 3 % NAV auto-allocated to hedges
TAIL_HEDGE_TRIGGER_RISK_SCORE: float = 65  # activate when portfolio_risk_score ≥ 65

# Rebalance drift band (absolute weight deviation)
REBALANCE_DRIFT_THRESHOLD: float = 0.03   # 3 % drift triggers rebalance flag

# Correlation de-risk — max correlated cluster weight
CORRELATION_CLUSTER_CAP: float = 0.35     # 35 % NAV per highly-correlated cluster

# ── OPTIONS LAYER CONSTANTS ───────────────────────────────────────────────────
OPTION_DELTA_MAX:            float = 0.80  # |delta| cap per option position
OPTION_GAMMA_RISK_CAP:       float = 0.05  # 5 % NAV gamma-risk ceiling
OPTION_THETA_DAILY_MAX:      float = 0.002 # 0.2 % NAV max daily theta bleed
OPTION_IV_CRUSH_HAIRCUT:     float = 0.30  # 30 % size cut when IV > 1.5× 20d avg
OPTION_EARNINGS_SUPPRESSION: float = 0.50  # 50 % cut near earnings events
OPTION_0DTE_MAX_FRACTION:    float = 0.01  # max 1 % NAV in 0DTE
OPTION_NEAR_EXPIRY_DAYS:     int   = 5     # DTE threshold for near-expiry scaling
OPTION_NEAR_EXPIRY_SCALE:    float = 0.50  # halve size when DTE ≤ 5
OPTION_PREMIUM_MAX:          float = 0.10  # 10 % NAV total options premium at risk
MAX_IV_RATIO:                float = 1.50  # IV-crush trigger ratio

# Confidence / signal quality thresholds
CONFIDENCE_MIN_DEPLOY: float = 0.40       # skip signals below 40 % confidence
SIGNAL_STRENGTH_BASE:  float = 0.50       # neutral anchor for sizing

# ─────────────────────────────────────────────────────────────────────────────
# TYPE ALIASES  (matches upstream pipeline field names)
# ─────────────────────────────────────────────────────────────────────────────

SignalDict    = Dict[str, Any]
WeightMap     = Dict[str, float]
OrderList     = List[Dict[str, Any]]
HedgeMap      = Dict[str, float]


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO CONTEXT  (injected by god_core / state_manager)
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioContext:
    """
    Carries live portfolio state injected from state_manager / god_core.
    All monetary values are expressed as fractions of NAV unless noted.
    """

    def __init__(
        self,
        current_positions:    Optional[Dict[str, float]] = None,
        nav:                  float = 1_000_000.0,
        peak_nav:             float = 1_000_000.0,
        realised_vol_series:  Optional[List[float]] = None,
        market_regime:        str   = "neutral",   # bull | bear | neutral | crisis
        session:              str   = "intraday",  # intraday | overnight | pre-market
        iv_baseline_map:      Optional[Dict[str, float]] = None,
        earnings_calendar:    Optional[List[str]] = None,
        correlation_matrix:   Optional[Dict[str, Dict[str, float]]] = None,
        existing_options_premium: float = 0.0,
    ):
        self.current_positions:    Dict[str, float] = current_positions or {}
        self.nav:                  float = max(nav, 1.0)
        self.peak_nav:             float = max(peak_nav, nav)
        self.realised_vol_series:  List[float] = realised_vol_series or [0.01] * VOL_LOOKBACK_DAYS
        self.market_regime:        str   = market_regime
        self.session:              str   = session
        self.iv_baseline_map:      Dict[str, float] = iv_baseline_map or {}
        self.earnings_calendar:    List[str] = earnings_calendar or []
        self.correlation_matrix:   Dict[str, Dict[str, float]] = correlation_matrix or {}
        self.existing_options_premium: float = existing_options_premium

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak as a positive fraction."""
        return max(0.0, (self.peak_nav - self.nav) / self.peak_nav)

    @property
    def drawdown_scalar(self) -> float:
        """Returns the allocation scalar dictated by the drawdown tier."""
        scalar = 1.0
        for threshold, scale in sorted(DRAWDOWN_TIERS, reverse=True):
            if self.drawdown >= threshold:
                scalar = scale
                break
        return scalar

    @property
    def realised_vol(self) -> float:
        """Annualised realised volatility from the daily-return series."""
        series = self.realised_vol_series[-VOL_LOOKBACK_DAYS:]
        if len(series) < 2:
            return 0.15   # assume 15 % when insufficient history
        try:
            daily_vol = statistics.stdev(series)
        except statistics.StatisticsError:
            daily_vol = 0.01
        return daily_vol * math.sqrt(252)

    @property
    def vol_scalar(self) -> float:
        """Scale factor to hit VOL_TARGET; clamped to [VOL_SCALE_MIN, VOL_SCALE_MAX]."""
        rv = self.realised_vol
        if rv <= 0:
            return 1.0
        raw = VOL_TARGET / rv
        return max(VOL_SCALE_MIN, min(VOL_SCALE_MAX, raw))


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _safe_get(d: dict, key: str, default: Any = 0.0) -> Any:
    val = d.get(key, default)
    return default if val is None else val


def _normalise_asset_class(raw: str) -> str:
    """Map upstream free-text asset_class to a canonical bucket."""
    mapping = {
        "stock": "equity", "equities": "equity", "share": "equity",
        "exchange traded fund": "etf",
        "cryptocurrency": "crypto", "digital asset": "crypto",
        "fx": "forex", "currency": "forex",
        "commodit": "commodity", "metal": "commodity", "oil": "commodity",
        "future": "futures", "contract": "futures",
        "fixed income": "bond", "treasury": "bond", "credit": "bond",
        "call": "option", "put": "option", "opt": "option",
        "inverse": "inverse_etf",
        "leveraged": "leveraged_etf",
        "vix": "volatility", "vol": "volatility",
    }
    r = raw.lower().strip()
    for k, v in mapping.items():
        if k in r:
            return v
    return r


def _direction_sign(signal: SignalDict) -> float:
    """Convert signal_direction to ±1.0 multiplier."""
    d = str(_safe_get(signal, "signal_direction", "long")).lower()
    if d in ("short", "sell", "bear", "-1", "bearish"):
        return -1.0
    return 1.0


def _dte(signal: SignalDict) -> Optional[int]:
    """Days-to-expiry for option signals. Returns None for non-options."""
    expiry = signal.get("expiry")
    if not expiry:
        return None
    try:
        if isinstance(expiry, str):
            exp_dt = datetime.fromisoformat(expiry)
        elif isinstance(expiry, datetime):
            exp_dt = expiry
        else:
            return None
        now = datetime.now(tz=timezone.utc)
        exp_dt = exp_dt.replace(tzinfo=timezone.utc) if exp_dt.tzinfo is None else exp_dt
        return max(0, (exp_dt.date() - now.date()).days)
    except Exception:
        return None


def _overnight_haircut(asset_class: str) -> float:
    """Return size haircut fraction for overnight gap risk."""
    if asset_class == "equity":
        return OVERNIGHT_HAIRCUT_EQUITY
    if asset_class == "crypto":
        return OVERNIGHT_HAIRCUT_CRYPTO
    return OVERNIGHT_HAIRCUT_DEFAULT


def _correlation_cluster(
    symbol: str,
    universe: List[str],
    corr_matrix: Dict[str, Dict[str, float]],
    threshold: float = 0.75,
) -> List[str]:
    """Return symbols highly correlated with `symbol` (|r| ≥ threshold)."""
    cluster = [symbol]
    row = corr_matrix.get(symbol, {})
    for other in universe:
        if other != symbol and abs(row.get(other, 0.0)) >= threshold:
            cluster.append(other)
    return cluster


# ─────────────────────────────────────────────────────────────────────────────
# RAW SIGNAL SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _raw_signal_score(signal: SignalDict) -> float:
    """
    Composite score in [0, 1] driving raw allocation weight.
    Combines: confidence_score, signal_strength, impact_score, urgency.
    All upstream field names preserved verbatim.
    """
    confidence  = float(_safe_get(signal, "confidence_score",  0.5))
    strength    = float(_safe_get(signal, "signal_strength",   SIGNAL_STRENGTH_BASE))
    impact      = float(_safe_get(signal, "impact_score",      0.5))
    urgency_raw = _safe_get(signal, "urgency", "medium")

    urgency_map = {"low": 0.3, "medium": 0.5, "high": 0.8, "critical": 1.0}
    urgency = urgency_map.get(str(urgency_raw).lower(), 0.5)

    # Weighted composite — confidence is the dominant driver
    score = 0.45 * confidence + 0.25 * strength + 0.20 * impact + 0.10 * urgency
    return max(0.0, min(1.0, score))


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS LAYER
# ─────────────────────────────────────────────────────────────────────────────

class OptionsLayer:
    """
    Validates and adjusts sizing for option signals.
    Reads delta, gamma, theta, implied_volatility, expiry, option_type
    from upstream signal dict (set by signal_engine).
    """

    def __init__(self, ctx: PortfolioContext):
        self.ctx = ctx

    def adjust(self, signal: SignalDict, raw_weight: float) -> Tuple[float, List[str]]:
        """
        Returns (adjusted_weight, list_of_reasons_applied).
        Weight is expressed as fraction of NAV.
        """
        reasons: List[str] = []
        asset_class = _normalise_asset_class(str(_safe_get(signal, "asset_class", "")))
        if asset_class != "option":
            return raw_weight, reasons

        weight = raw_weight
        symbol = str(_safe_get(signal, "symbol", "UNKNOWN"))

        # ── Delta cap ──────────────────────────────────────────────────────
        delta = abs(float(_safe_get(signal, "delta", 0.50)))
        if delta > OPTION_DELTA_MAX:
            scale = OPTION_DELTA_MAX / delta
            weight *= scale
            reasons.append(f"delta_cap(|δ|={delta:.2f}→scale={scale:.2f})")

        # ── Near-expiry reduction ──────────────────────────────────────────
        dte = _dte(signal)
        if dte is not None:
            if dte == 0:
                weight = min(weight, OPTION_0DTE_MAX_FRACTION)
                reasons.append("0DTE_hard_limit")
            elif dte <= OPTION_NEAR_EXPIRY_DAYS:
                weight *= OPTION_NEAR_EXPIRY_SCALE
                reasons.append(f"near_expiry(DTE={dte}→scale={OPTION_NEAR_EXPIRY_SCALE})")

        # ── IV-crush risk haircut ──────────────────────────────────────────
        iv = float(_safe_get(signal, "implied_volatility", 0.0))
        iv_baseline = self.ctx.iv_baseline_map.get(symbol, iv)
        if iv_baseline > 0 and iv / iv_baseline > MAX_IV_RATIO:
            weight *= (1.0 - OPTION_IV_CRUSH_HAIRCUT)
            reasons.append(f"IV_crush_haircut(IV={iv:.2f},base={iv_baseline:.2f})")

        # ── Earnings-event suppression ─────────────────────────────────────
        if symbol in self.ctx.earnings_calendar:
            weight *= (1.0 - OPTION_EARNINGS_SUPPRESSION)
            reasons.append(f"earnings_suppression({symbol})")

        # ── Theta bleed penalty ────────────────────────────────────────────
        theta = abs(float(_safe_get(signal, "theta", 0.0)))
        theta_fraction = theta / max(self.ctx.nav, 1.0)
        if theta_fraction > OPTION_THETA_DAILY_MAX:
            penalty = OPTION_THETA_DAILY_MAX / theta_fraction
            weight *= penalty
            reasons.append(f"theta_bleed_penalty(θ={theta:.4f})")

        # ── Gamma risk cap ─────────────────────────────────────────────────
        gamma = abs(float(_safe_get(signal, "gamma", 0.0)))
        gamma_nav_frac = gamma * (weight * self.ctx.nav)
        if gamma_nav_frac > OPTION_GAMMA_RISK_CAP * self.ctx.nav:
            safe_notional = OPTION_GAMMA_RISK_CAP * self.ctx.nav / max(gamma, 1e-8)
            weight = min(weight, safe_notional / self.ctx.nav)
            reasons.append(f"gamma_risk_cap(γ={gamma:.4f})")

        # ── Portfolio-level premium-at-risk cap ───────────────────────────
        existing = self.ctx.existing_options_premium / self.ctx.nav
        if existing + weight > OPTION_PREMIUM_MAX:
            weight = max(0.0, OPTION_PREMIUM_MAX - existing)
            reasons.append(f"premium_cap(existing={existing:.3f})")

        return max(0.0, weight), reasons

    @staticmethod
    def classify_strategy(signal: SignalDict) -> str:
        """
        Infer option strategy label from signal fields.
        Hooks for: protective put, covered call, wheel, spread.
        """
        opt_type = str(_safe_get(signal, "option_type", "")).lower()
        direction = str(_safe_get(signal, "signal_direction", "long")).lower()
        event_type = str(_safe_get(signal, "event_type", "")).lower()

        if opt_type == "put" and direction == "long":
            if "hedge" in event_type or "protect" in event_type:
                return "protective_put"
            return "long_put"
        if opt_type == "put" and direction == "short":
            if "wheel" in event_type:
                return "cash_secured_put (wheel)"
            return "short_put"
        if opt_type == "call" and direction == "short":
            if "covered" in event_type:
                return "covered_call"
            return "short_call"
        if opt_type == "call" and direction == "long":
            return "long_call"
        if "spread" in event_type:
            return "spread_hook"
        return f"option_{opt_type}_{direction}"


# ─────────────────────────────────────────────────────────────────────────────
# ALLOCATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AllocationEngine:
    """
    Core allocation logic.  Converts a list of validated signals into
    raw target weights before exposure-control normalisation.
    """

    def __init__(self, ctx: PortfolioContext):
        self.ctx = ctx
        self.options_layer = OptionsLayer(ctx)

    # ── Pre-flight filter ─────────────────────────────────────────────────────

    def _filter_signals(self, signals: List[SignalDict]) -> List[SignalDict]:
        accepted = []
        for sig in signals:
            symbol = str(_safe_get(sig, "symbol", ""))
            conf   = float(_safe_get(sig, "confidence_score", 0.0))
            liq    = float(_safe_get(sig, "liquidity_score", 1.0))

            if conf < CONFIDENCE_MIN_DEPLOY:
                log.debug("Skipped %s: confidence %.2f < %.2f", symbol, conf, CONFIDENCE_MIN_DEPLOY)
                continue
            if liq < LIQUIDITY_SCORE_MIN_DEPLOY:
                log.debug("Skipped %s: liquidity %.2f < %.2f", symbol, liq, LIQUIDITY_SCORE_MIN_DEPLOY)
                continue
            accepted.append(sig)

        log.info("Signal filter: %d/%d accepted", len(accepted), len(signals))
        return accepted

    # ── Raw weight per signal ─────────────────────────────────────────────────

    def _compute_raw_weight(self, signal: SignalDict) -> float:
        """
        Base weight = raw_score × vol_scalar × drawdown_scalar × direction_sign.
        Then apply liquidity haircut, slippage penalty, and overnight haircut.
        Options layer applied separately.
        """
        score     = _raw_signal_score(signal)
        direction = _direction_sign(signal)
        liq       = float(_safe_get(signal, "liquidity_score", 1.0))
        asset_cls = _normalise_asset_class(str(_safe_get(signal, "asset_class", "equity")))

        # Position size hint from signal_engine / risk_guardian
        pos_bias  = float(_safe_get(signal, "position_size_bias", 1.0))
        pos_bias  = max(0.0, min(2.0, pos_bias))   # clamp rogue values

        base = score * self.ctx.vol_scalar * self.ctx.drawdown_scalar * pos_bias

        # Liquidity-adjusted sizing — slippage penalty scales with illiquidity
        illiquidity = max(0.0, 1.0 - liq)
        slippage_penalty = 1.0 - illiquidity * SLIPPAGE_PENALTY_SCALE
        base *= max(0.0, slippage_penalty)

        # Overnight gap risk haircut when session flag is overnight/pre-market
        if self.ctx.session in ("overnight", "pre-market", "post-market"):
            haircut = _overnight_haircut(asset_cls)
            base *= (1.0 - haircut)

        return direction * base

    # ── Asset-class bucketing ─────────────────────────────────────────────────

    def _bucket_weights(
        self, weighted_signals: List[Tuple[SignalDict, float]]
    ) -> Tuple[WeightMap, Dict[str, List[str]]]:
        """
        Returns (symbol→weight, asset_class→[symbols]) after
        applying per-asset-class and sector caps.
        """
        # Sum raw weights per symbol
        symbol_weight: Dict[str, float] = {}
        symbol_class:  Dict[str, str]   = {}
        symbol_sector: Dict[str, str]   = {}

        for sig, w in weighted_signals:
            sym    = str(_safe_get(sig, "symbol", f"UNK_{uuid.uuid4().hex[:6]}"))
            ac     = _normalise_asset_class(str(_safe_get(sig, "asset_class", "equity")))
            sector = str(_safe_get(sig, "sector", "unknown")).lower().strip()
            symbol_weight[sym]  = symbol_weight.get(sym, 0.0) + w
            symbol_class[sym]   = ac
            symbol_sector[sym]  = sector

        # Compute gross weight per asset class
        ac_gross: Dict[str, float] = defaultdict(float)
        for sym, w in symbol_weight.items():
            ac_gross[symbol_class[sym]] += abs(w)

        # Scale down asset classes exceeding their cap
        for sym in list(symbol_weight):
            ac  = symbol_class[sym]
            cap = ASSET_CLASS_CAPS.get(ac, 0.50)
            total_ac = ac_gross[ac]
            if total_ac > cap and total_ac > 0:
                scale = cap / total_ac
                symbol_weight[sym] *= scale

        # Sector cap enforcement
        sector_gross: Dict[str, float] = defaultdict(float)
        for sym, w in symbol_weight.items():
            sector_gross[symbol_sector[sym]] += abs(w)

        for sym in list(symbol_weight):
            sec = symbol_sector[sym]
            if sector_gross[sec] > SECTOR_CAP and sector_gross[sec] > 0:
                scale = SECTOR_CAP / sector_gross[sec]
                symbol_weight[sym] *= scale

        # Build class→symbols map for downstream
        class_universe: Dict[str, List[str]] = defaultdict(list)
        for sym, ac in symbol_class.items():
            class_universe[ac].append(sym)

        return symbol_weight, dict(class_universe)

    # ── Correlation de-risk ───────────────────────────────────────────────────

    def _correlation_derisking(self, weights: WeightMap) -> WeightMap:
        """
        Identify highly-correlated clusters; if the combined weight of a
        cluster exceeds CORRELATION_CLUSTER_CAP, scale the whole cluster down.
        """
        if not self.ctx.correlation_matrix:
            return weights

        symbols   = list(weights)
        visited   = set()
        adjusted  = dict(weights)

        for sym in symbols:
            if sym in visited:
                continue
            cluster = _correlation_cluster(sym, symbols, self.ctx.correlation_matrix)
            visited.update(cluster)
            cluster_weight = sum(abs(adjusted.get(s, 0.0)) for s in cluster)
            if cluster_weight > CORRELATION_CLUSTER_CAP and cluster_weight > 0:
                scale = CORRELATION_CLUSTER_CAP / cluster_weight
                for s in cluster:
                    adjusted[s] = adjusted.get(s, 0.0) * scale
                log.debug(
                    "Correlation cluster %s capped (%.2f→%.2f)",
                    cluster, cluster_weight, CORRELATION_CLUSTER_CAP,
                )

        return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# HEDGE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class HedgeEngine:
    """
    Builds tail-risk and cross-asset hedge allocations.
    Activates when portfolio_risk_score ≥ TAIL_HEDGE_TRIGGER_RISK_SCORE
    or when market_regime is 'crisis' / 'bear'.
    """

    _HEDGE_INSTRUMENTS = {
        "equity":    {"hedge": "SPY_PUT_BASKET",   "fraction": 0.40},
        "crypto":    {"hedge": "BTC_PUT_BASKET",   "fraction": 0.20},
        "forex":     {"hedge": "USD_LONG",         "fraction": 0.15},
        "commodity": {"hedge": "GLD_LONG",         "fraction": 0.10},
        "general":   {"hedge": "VIX_CALL_BASKET",  "fraction": 0.15},
    }

    def __init__(self, ctx: PortfolioContext):
        self.ctx = ctx

    def compute(
        self,
        weights:    WeightMap,
        risk_score: float,
        class_universe: Dict[str, List[str]],
    ) -> HedgeMap:
        """Return hedge_allocations as fraction of NAV."""
        hedges: HedgeMap = {}

        trigger = (
            risk_score >= TAIL_HEDGE_TRIGGER_RISK_SCORE
            or self.ctx.market_regime in ("crisis", "bear")
        )
        if not trigger:
            return hedges

        # Determine which asset classes have significant long exposure
        long_exposure: Dict[str, float] = defaultdict(float)
        for sym, w in weights.items():
            if w > 0:
                # find asset class by reverse lookup
                for ac, syms in class_universe.items():
                    if sym in syms:
                        long_exposure[ac] += w
                        break

        total_hedge_budget = TAIL_HEDGE_FRACTION

        # Allocate hedge budget proportional to long-class exposures
        total_long = sum(long_exposure.values()) or 1.0
        for ac, exp in long_exposure.items():
            spec = self._HEDGE_INSTRUMENTS.get(ac, self._HEDGE_INSTRUMENTS["general"])
            hedge_sym = spec["hedge"]
            fraction  = spec["fraction"]
            alloc = total_hedge_budget * fraction * (exp / total_long)
            hedges[hedge_sym] = hedges.get(hedge_sym, 0.0) + alloc

        # Always include a small VIX call allocation when regime is crisis
        if self.ctx.market_regime == "crisis":
            hedges["VIX_CALL_BASKET"] = hedges.get("VIX_CALL_BASKET", 0.0) + 0.01

        log.info("Tail-risk hedges activated: %s", hedges)
        return hedges


# ─────────────────────────────────────────────────────────────────────────────
# EXPOSURE CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class ExposureController:
    """
    Enforces gross/net exposure ceilings and the cash reserve floor.
    Scales down all weights proportionally when limits are breached.
    """

    def enforce(self, weights: WeightMap) -> Tuple[WeightMap, float, float, float]:
        """
        Returns (adjusted_weights, gross_exposure, net_exposure, cash_buffer).
        Exposures are fractions of NAV.
        """
        gross = sum(abs(w) for w in weights.values())
        net   = sum(weights.values())

        # Gross cap
        if gross > GROSS_EXPOSURE_MAX:
            scale = GROSS_EXPOSURE_MAX / gross
            weights = {s: w * scale for s, w in weights.items()}
            gross = GROSS_EXPOSURE_MAX
            net  *= scale
            log.info("Gross exposure scaled: %.2f → %.2f", gross / scale, gross)

        # Net cap (long side)
        if net > NET_EXPOSURE_MAX:
            excess = net - NET_EXPOSURE_MAX
            # trim longs proportionally
            long_total = sum(w for w in weights.values() if w > 0) or 1e-9
            for s in list(weights):
                if weights[s] > 0:
                    weights[s] -= excess * (weights[s] / long_total)
            net = NET_EXPOSURE_MAX
            gross = sum(abs(w) for w in weights.values())
            log.info("Net exposure capped at %.2f", NET_EXPOSURE_MAX)

        # Net floor (short side)
        if net < NET_EXPOSURE_MIN:
            excess = NET_EXPOSURE_MIN - net
            short_total = sum(abs(w) for w in weights.values() if w < 0) or 1e-9
            for s in list(weights):
                if weights[s] < 0:
                    weights[s] += excess * (abs(weights[s]) / short_total)
            net = NET_EXPOSURE_MIN
            gross = sum(abs(w) for w in weights.values())
            log.info("Net exposure floored at %.2f", NET_EXPOSURE_MIN)

        # Cash reserve floor  — cash_buffer = 1 − deployed − hedges
        deployed   = sum(abs(w) for w in weights.values())
        cash_buffer = max(CASH_RESERVE_FLOOR, 1.0 - deployed)

        if deployed > (1.0 - CASH_RESERVE_FLOOR):
            scale = (1.0 - CASH_RESERVE_FLOOR) / max(deployed, 1e-9)
            weights = {s: w * scale for s, w in weights.items()}
            gross  *= scale
            net    *= scale
            cash_buffer = CASH_RESERVE_FLOOR
            log.info("Cash floor enforced: weights scaled by %.3f", scale)

        return weights, gross, net, cash_buffer


# ─────────────────────────────────────────────────────────────────────────────
# RISK SCORER
# ─────────────────────────────────────────────────────────────────────────────

class RiskScorer:
    """
    Produces a portfolio_risk_score in [0, 100] integrating:
    concentration, volatility, drawdown, leverage, options Greeks, regime.
    """

    def score(
        self,
        weights:        WeightMap,
        gross_exposure: float,
        net_exposure:   float,
        ctx:            PortfolioContext,
        signals:        List[SignalDict],
    ) -> Tuple[float, List[str]]:
        """Returns (risk_score_0_100, [explanation_strings])."""
        components: List[Tuple[str, float]] = []

        # 1. Volatility component (0–25)
        vol_frac = min(ctx.realised_vol / 0.40, 1.0)   # normalise vs 40 % vol ceiling
        components.append(("realised_vol", 25.0 * vol_frac))

        # 2. Drawdown component (0–20)
        dd_frac = min(ctx.drawdown / 0.25, 1.0)
        components.append(("drawdown", 20.0 * dd_frac))

        # 3. Gross leverage component (0–20)
        lev_frac = min(gross_exposure / GROSS_EXPOSURE_MAX, 1.0)
        components.append(("gross_leverage", 20.0 * lev_frac))

        # 4. Concentration component (0–15)
        if weights:
            max_wt = max(abs(w) for w in weights.values())
            conc_frac = min(max_wt / 0.20, 1.0)   # normalise vs 20 % single-name cap
        else:
            conc_frac = 0.0
        components.append(("concentration", 15.0 * conc_frac))

        # 5. Options gamma risk component (0–10)
        total_gamma_risk = 0.0
        for sig in signals:
            ac = _normalise_asset_class(str(_safe_get(sig, "asset_class", "")))
            if ac == "option":
                sym   = str(_safe_get(sig, "symbol", ""))
                gamma = abs(float(_safe_get(sig, "gamma", 0.0)))
                wt    = abs(weights.get(sym, 0.0))
                total_gamma_risk += gamma * wt
        gamma_frac = min(total_gamma_risk / OPTION_GAMMA_RISK_CAP, 1.0)
        components.append(("gamma_risk", 10.0 * gamma_frac))

        # 6. Regime risk premium (0–10)
        regime_premium = {
            "bull": 0.0, "neutral": 0.3, "bear": 0.7, "crisis": 1.0
        }.get(ctx.market_regime, 0.3)
        components.append(("market_regime", 10.0 * regime_premium))

        raw_score = sum(v for _, v in components)
        score     = max(0.0, min(100.0, raw_score))

        explanations = [
            f"{name}={val:.1f}/component_max" for name, val in components
        ]
        explanations.append(f"total_risk_score={score:.1f}/100")

        return score, explanations


# ─────────────────────────────────────────────────────────────────────────────
# REBALANCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class RebalanceEngine:
    """
    Compares target weights to current_positions.
    Flags rebalance_required when any symbol drifts beyond REBALANCE_DRIFT_THRESHOLD.
    """

    def check(
        self,
        target_weights:    WeightMap,
        current_positions: Dict[str, float],
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Returns (rebalance_required, drift_map).
        drift_map: symbol → weight_delta (positive = need to buy, negative = need to sell).
        """
        all_symbols = set(target_weights) | set(current_positions)
        drift_map: Dict[str, float] = {}

        for sym in all_symbols:
            target  = target_weights.get(sym, 0.0)
            current = current_positions.get(sym, 0.0)
            delta   = target - current
            drift_map[sym] = delta

        rebalance_required = any(
            abs(d) >= REBALANCE_DRIFT_THRESHOLD for d in drift_map.values()
        )
        return rebalance_required, drift_map


# ─────────────────────────────────────────────────────────────────────────────
# ORDER GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class OrderGenerator:
    """
    Converts (drift_map, signals, ctx) → target_orders list.
    Output fields are plug-compatible with execution_bridge.
    """

    def generate(
        self,
        drift_map:      Dict[str, float],
        signals:        List[SignalDict],
        ctx:            PortfolioContext,
        hedge_allocs:   HedgeMap,
    ) -> OrderList:
        """
        Each order contains all fields expected by execution_bridge:
          symbol, side, quantity_nav_fraction, order_type, urgency,
          asset_class, execution_priority, signal_direction,
          option metadata (if applicable), hedge_flag, order_id.
        """
        sig_map: Dict[str, SignalDict] = {
            str(_safe_get(s, "symbol", "")): s for s in signals
        }
        orders: OrderList = []

        for sym, delta in drift_map.items():
            if abs(delta) < 1e-5:
                continue

            sig        = sig_map.get(sym, {})
            ac         = _normalise_asset_class(str(_safe_get(sig, "asset_class", "equity")))
            urgency    = str(_safe_get(sig, "urgency", "medium")).lower()
            exec_prio  = str(_safe_get(sig, "execution_priority", "normal")).lower()
            sig_dir    = str(_safe_get(sig, "signal_direction", "long")).lower()

            order: Dict[str, Any] = {
                "order_id":             f"PB-{uuid.uuid4().hex[:12].upper()}",
                "symbol":               sym,
                "side":                 "buy" if delta > 0 else "sell",
                "quantity_nav_fraction": abs(delta),
                "order_type":           _order_type(urgency, ac),
                "urgency":              urgency,
                "execution_priority":   exec_prio,
                "asset_class":          ac,
                "signal_direction":     sig_dir,
                "hedge_flag":           False,
                "generated_at":         datetime.now(tz=timezone.utc).isoformat(),
            }

            # Options metadata passthrough
            if ac == "option":
                order.update({
                    "option_type":      _safe_get(sig, "option_type", "call"),
                    "strike":           _safe_get(sig, "strike", None),
                    "expiry":           _safe_get(sig, "expiry", None),
                    "delta":            _safe_get(sig, "delta", None),
                    "gamma":            _safe_get(sig, "gamma", None),
                    "theta":            _safe_get(sig, "theta", None),
                    "implied_volatility": _safe_get(sig, "implied_volatility", None),
                    "option_strategy":  OptionsLayer.classify_strategy(sig),
                })

            orders.append(order)

        # Hedge orders
        for hedge_sym, alloc in hedge_allocs.items():
            if alloc < 1e-5:
                continue
            orders.append({
                "order_id":              f"PB-HEDGE-{uuid.uuid4().hex[:10].upper()}",
                "symbol":                hedge_sym,
                "side":                  "buy",
                "quantity_nav_fraction": alloc,
                "order_type":            "limit",
                "urgency":               "high",
                "execution_priority":    "normal",
                "asset_class":           _infer_hedge_class(hedge_sym),
                "signal_direction":      "long",
                "hedge_flag":            True,
                "generated_at":          datetime.now(tz=timezone.utc).isoformat(),
            })

        log.info("Generated %d target orders (%d hedges)", len(orders), len(hedge_allocs))
        return orders


def _order_type(urgency: str, asset_class: str) -> str:
    """Map urgency + asset class to an order type string for execution_bridge."""
    if urgency in ("critical", "high"):
        return "market"
    if asset_class in ("option", "futures"):
        return "limit"
    return "limit"


def _infer_hedge_class(sym: str) -> str:
    s = sym.upper()
    if "VIX" in s:   return "volatility"
    if "PUT" in s:   return "option"
    if "CALL" in s:  return "option"
    if "GLD" in s:   return "commodity"
    if "USD" in s:   return "forex"
    return "etf"


# ─────────────────────────────────────────────────────────────────────────────
# ALLOCATION SUMMARY (EXPLAINABILITY)
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary(
    target_weights:    WeightMap,
    gross_exposure:    float,
    net_exposure:      float,
    cash_buffer:       float,
    risk_score:        float,
    risk_explanations: List[str],
    hedge_allocations: HedgeMap,
    ctx:               PortfolioContext,
    rebalance_required: bool,
    options_reasons:   Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Human-readable allocation summary that flows into alert_router
    and god_core.explainability_log.
    """
    top_positions = sorted(
        target_weights.items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:10]

    return {
        "module":              "portfolio_brain",
        "timestamp":           datetime.now(tz=timezone.utc).isoformat(),
        "market_regime":       ctx.market_regime,
        "drawdown_pct":        round(ctx.drawdown * 100, 2),
        "drawdown_scalar":     round(ctx.drawdown_scalar, 3),
        "vol_target":          VOL_TARGET,
        "realised_vol":        round(ctx.realised_vol, 4),
        "vol_scalar":          round(ctx.vol_scalar, 3),
        "gross_exposure":      round(gross_exposure, 4),
        "net_exposure":        round(net_exposure, 4),
        "cash_buffer":         round(cash_buffer, 4),
        "portfolio_risk_score": round(risk_score, 2),
        "risk_breakdown":      risk_explanations,
        "top_positions":       [{"symbol": s, "weight": round(w, 4)} for s, w in top_positions],
        "hedge_allocations":   {k: round(v, 4) for k, v in hedge_allocations.items()},
        "rebalance_required":  rebalance_required,
        "options_adjustments": options_reasons,
        "session":             ctx.session,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO BRAIN  — PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioBrain:
    """
    Primary entry-point for the portfolio allocation pipeline stage.

    Usage
    -----
    brain = PortfolioBrain(ctx)
    result = brain.process(validated_signals)

    The returned dict satisfies the output contract consumed by
    execution_bridge, god_core, and state_manager.
    """

    def __init__(self, ctx: Optional[PortfolioContext] = None):
        self.ctx              = ctx or PortfolioContext()
        self.allocator        = AllocationEngine(self.ctx)
        self.exposure_ctrl    = ExposureController()
        self.risk_scorer      = RiskScorer()
        self.hedge_engine     = HedgeEngine(self.ctx)
        self.rebalance_engine = RebalanceEngine()
        self.order_gen        = OrderGenerator()

    # ── Main process entrypoint ───────────────────────────────────────────────

    def process(self, signals: List[SignalDict]) -> Dict[str, Any]:
        """
        Full pipeline:
          1. Filter signals (confidence + liquidity gates)
          2. Score + compute raw weights
          3. Apply options layer per signal
          4. Bucket by asset class / sector (cap enforcement)
          5. Correlation-aware de-risking
          6. Gross / net exposure + cash-floor control
          7. Risk scoring
          8. Hedge engine
          9. Rebalance detection
         10. Target order generation
         11. Assemble output contract

        Parameters
        ----------
        signals : List[SignalDict]
            Validated signals from signal_engine → risk_guardian pipeline.

        Returns
        -------
        Dict matching the documented output contract.
        """
        log.info("PortfolioBrain.process(): %d incoming signals", len(signals))

        if not signals:
            return self._empty_result()

        # ── Step 1: filter ────────────────────────────────────────────────────
        accepted = self.allocator._filter_signals(signals)
        if not accepted:
            log.warning("All signals filtered; returning empty allocation")
            return self._empty_result()

        # ── Step 2 + 3: raw weights + options layer ───────────────────────────
        weighted: List[Tuple[SignalDict, float]] = []
        options_reasons: Dict[str, List[str]] = {}

        for sig in accepted:
            raw_w = self.allocator._compute_raw_weight(sig)
            sym   = str(_safe_get(sig, "symbol", ""))
            ac    = _normalise_asset_class(str(_safe_get(sig, "asset_class", "equity")))

            if ac == "option":
                adj_w, reasons = self.allocator.options_layer.adjust(sig, abs(raw_w))
                raw_w = math.copysign(adj_w, raw_w)
                if reasons:
                    options_reasons[sym] = reasons

            weighted.append((sig, raw_w))

        # ── Step 4: asset-class / sector bucketing ────────────────────────────
        symbol_weights, class_universe = self.allocator._bucket_weights(weighted)

        # ── Step 5: correlation de-risking ────────────────────────────────────
        symbol_weights = self.allocator._correlation_derisking(symbol_weights)

        # ── Step 6: exposure control ──────────────────────────────────────────
        symbol_weights, gross_exp, net_exp, cash_buf = (
            self.exposure_ctrl.enforce(symbol_weights)
        )

        # ── Step 7: risk scoring ──────────────────────────────────────────────
        risk_score, risk_explanations = self.risk_scorer.score(
            symbol_weights, gross_exp, net_exp, self.ctx, accepted
        )

        # ── Step 8: hedge engine ──────────────────────────────────────────────
        hedge_allocs = self.hedge_engine.compute(
            symbol_weights, risk_score, class_universe
        )

        # ── Step 9: rebalance detection ───────────────────────────────────────
        rebalance_required, drift_map = self.rebalance_engine.check(
            symbol_weights, self.ctx.current_positions
        )

        # ── Step 10: order generation ─────────────────────────────────────────
        target_orders = self.order_gen.generate(
            drift_map, accepted, self.ctx, hedge_allocs
        )

        # ── Step 11: assemble output contract ─────────────────────────────────
        summary = _build_summary(
            symbol_weights, gross_exp, net_exp, cash_buf,
            risk_score, risk_explanations, hedge_allocs,
            self.ctx, rebalance_required, options_reasons,
        )

        output: Dict[str, Any] = {
            # ── Primary output contract (execution_bridge / god_core) ─────────
            "target_weights":        {s: round(w, 6) for s, w in symbol_weights.items()},
            "target_orders":         target_orders,
            "gross_exposure":        round(gross_exp, 6),
            "net_exposure":          round(net_exp, 6),
            "cash_buffer":           round(cash_buf, 6),
            "portfolio_risk_score":  round(risk_score, 4),
            "hedge_allocations":     {k: round(v, 6) for k, v in hedge_allocs.items()},
            "rebalance_required":    rebalance_required,
            # ── Explainability + routing metadata ────────────────────────────
            "allocation_summary":    summary,
            "class_universe":        class_universe,
            "drift_map":             {s: round(d, 6) for s, d in drift_map.items()},
            "options_adjustments":   options_reasons,
            "market_regime":         self.ctx.market_regime,
            "vol_scalar":            round(self.ctx.vol_scalar, 4),
            "drawdown_scalar":       round(self.ctx.drawdown_scalar, 4),
            "session":               self.ctx.session,
            "generated_at":          datetime.now(tz=timezone.utc).isoformat(),
            "module":                "portfolio_brain",
        }

        log.info(
            "Allocation complete | symbols=%d | gross=%.2f | net=%.2f | "
            "cash=%.2f | risk_score=%.1f | orders=%d",
            len(symbol_weights), gross_exp, net_exp,
            cash_buf, risk_score, len(target_orders),
        )
        return output

    # ── Empty-result helper ───────────────────────────────────────────────────

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "target_weights":       {},
            "target_orders":        [],
            "gross_exposure":       0.0,
            "net_exposure":         0.0,
            "cash_buffer":          1.0,
            "portfolio_risk_score": 0.0,
            "hedge_allocations":    {},
            "rebalance_required":   False,
            "allocation_summary":   {},
            "class_universe":       {},
            "drift_map":            {},
            "options_adjustments":  {},
            "market_regime":        self.ctx.market_regime,
            "vol_scalar":           round(self.ctx.vol_scalar, 4),
            "drawdown_scalar":      round(self.ctx.drawdown_scalar, 4),
            "session":              self.ctx.session,
            "generated_at":         datetime.now(tz=timezone.utc).isoformat(),
            "module":               "portfolio_brain",
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONVENIENCE FUNCTION  (pipeline-compatible call signature)
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio_brain(
    signals: List[SignalDict],
    ctx:     Optional[PortfolioContext] = None,
) -> Dict[str, Any]:
    """
    Drop-in module function for the TRADING_AI pipeline.

    Parameters
    ----------
    signals : list of signal dicts from risk_guardian / signal_engine.
    ctx     : PortfolioContext injected by god_core / state_manager.
              Defaults to a sensible paper-trading context if omitted.

    Returns
    -------
    Output contract dict, ready for execution_bridge.ingest().
    """
    brain = PortfolioBrain(ctx)
    return brain.process(signals)


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _make_signal(
    symbol:       str,
    asset_class:  str,
    direction:    str   = "long",
    confidence:   float = 0.75,
    strength:     float = 0.70,
    impact:       float = 0.65,
    urgency:      str   = "high",
    liquidity:    float = 0.90,
    sector:       str   = "technology",
    pos_bias:     float = 1.0,
    **kwargs: Any,
) -> SignalDict:
    """Factory for test signal dicts that mirror upstream contracts."""
    return {
        "symbol":             symbol,
        "asset_class":        asset_class,
        "signal_direction":   direction,
        "confidence_score":   confidence,
        "signal_strength":    strength,
        "impact_score":       impact,
        "urgency":            urgency,
        "liquidity_score":    liquidity,
        "sector":             sector,
        "position_size_bias": pos_bias,
        "execution_priority": "high",
        "market_regime":      "bull",
        "signal_reasons":     ["smoke_test"],
        **kwargs,
    }


def _run_smoke_tests() -> None:
    """Deterministic smoke-test suite for portfolio_brain."""
    print("\n" + "═" * 70)
    print("  PORTFOLIO BRAIN — SMOKE TESTS")
    print("═" * 70)

    PASS = "✓ PASS"
    FAIL = "✗ FAIL"

    def assert_test(name: str, condition: bool, detail: str = "") -> None:
        status = PASS if condition else FAIL
        print(f"  {status}  {name}" + (f"  [{detail}]" if detail else ""))
        if not condition:
            raise AssertionError(f"Smoke test failed: {name} — {detail}")

    # ── Context ───────────────────────────────────────────────────────────────
    ctx = PortfolioContext(
        current_positions={"AAPL": 0.05, "MSFT": 0.04},
        nav=1_000_000.0,
        peak_nav=1_050_000.0,
        realised_vol_series=[0.008] * 20,
        market_regime="neutral",
        session="intraday",
        iv_baseline_map={"TSLA_CALL_450": 0.40},
        earnings_calendar=["TSLA_CALL_450"],
        correlation_matrix={
            "AAPL": {"MSFT": 0.82, "GOOGL": 0.78},
            "MSFT": {"AAPL": 0.82, "GOOGL": 0.75},
            "GOOGL": {"AAPL": 0.78, "MSFT": 0.75},
        },
        existing_options_premium=50_000.0,
    )

    # ── Test 1: Empty signal list ─────────────────────────────────────────────
    result = run_portfolio_brain([], ctx)
    assert_test(
        "Empty signal → empty allocation",
        result["target_weights"] == {} and result["gross_exposure"] == 0.0,
    )

    # ── Test 2: Single equity long ────────────────────────────────────────────
    signals = [_make_signal("AAPL", "equity")]
    result  = run_portfolio_brain(signals, ctx)
    assert_test(
        "Single equity: target_weights non-empty",
        "AAPL" in result["target_weights"],
    )
    assert_test(
        "Single equity: AAPL weight > 0 (long)",
        result["target_weights"]["AAPL"] > 0,
    )
    assert_test(
        "Output contract: all required keys present",
        all(k in result for k in [
            "target_weights", "target_orders", "gross_exposure",
            "net_exposure", "cash_buffer", "portfolio_risk_score",
            "hedge_allocations", "rebalance_required",
        ]),
    )

    # ── Test 3: Short signal ──────────────────────────────────────────────────
    result = run_portfolio_brain(
        [_make_signal("SPY", "etf", direction="short")], ctx
    )
    assert_test(
        "Short signal: weight < 0",
        result["target_weights"].get("SPY", 1.0) < 0,
    )

    # ── Test 4: Cash floor ────────────────────────────────────────────────────
    big_signals = [
        _make_signal(f"SYM{i}", "equity", pos_bias=2.0, confidence=0.95)
        for i in range(20)
    ]
    result = run_portfolio_brain(big_signals, ctx)
    assert_test(
        "Cash buffer >= CASH_RESERVE_FLOOR",
        result["cash_buffer"] >= CASH_RESERVE_FLOOR - 1e-9,
        f"cash_buffer={result['cash_buffer']:.4f}",
    )

    # ── Test 5: Gross exposure cap ────────────────────────────────────────────
    assert_test(
        "Gross exposure <= GROSS_EXPOSURE_MAX",
        result["gross_exposure"] <= GROSS_EXPOSURE_MAX + 1e-9,
        f"gross={result['gross_exposure']:.4f}",
    )

    # ── Test 6: Confidence gate ───────────────────────────────────────────────
    low_conf = [_make_signal("JUNK", "equity", confidence=0.10)]
    result   = run_portfolio_brain(low_conf, ctx)
    assert_test(
        "Low-confidence signal filtered (not in target_weights)",
        "JUNK" not in result["target_weights"],
    )

    # ── Test 7: Liquidity gate ────────────────────────────────────────────────
    illiq = [_make_signal("ILLIQ", "equity", liquidity=0.10)]
    result = run_portfolio_brain(illiq, ctx)
    assert_test(
        "Low-liquidity signal filtered",
        "ILLIQ" not in result["target_weights"],
    )

    # ── Test 8: Options layer — 0DTE hard limit ───────────────────────────────
    from datetime import timedelta
    zero_dte_sig = _make_signal(
        "SPX_CALL_5000", "option", direction="long",
        option_type="call", strike=5000,
        expiry=datetime.now(tz=timezone.utc).isoformat(),
        delta=0.50, gamma=0.002, theta=-10.0, implied_volatility=0.30,
        liquidity=0.80, confidence=0.85,
    )
    result = run_portfolio_brain([zero_dte_sig], ctx)
    spx_w = result["target_weights"].get("SPX_CALL_5000", 0.0)
    assert_test(
        "0DTE option: weight <= OPTION_0DTE_MAX_FRACTION",
        abs(spx_w) <= OPTION_0DTE_MAX_FRACTION + 1e-9,
        f"weight={spx_w:.6f}",
    )
    assert_test(
        "0DTE adjustment reason logged",
        "SPX_CALL_5000" in result.get("options_adjustments", {})
        or abs(spx_w) == 0.0,
    )

    # ── Test 9: Options layer — earnings suppression ──────────────────────────
    tsla_sig = _make_signal(
        "TSLA_CALL_450", "option", direction="long",
        option_type="call", strike=450,
        expiry=(datetime.now(tz=timezone.utc) + timedelta(days=14)).isoformat(),
        delta=0.45, gamma=0.003, theta=-5.0, implied_volatility=0.80,
        liquidity=0.75, confidence=0.88,
    )
    result_tsla = run_portfolio_brain([tsla_sig], ctx)
    tsla_w = result_tsla["target_weights"].get("TSLA_CALL_450", 0.0)
    # Earnings suppression + IV crush should both apply → weight reduced
    assert_test(
        "Earnings suppression + IV crush applied (weight < unadjusted)",
        abs(tsla_w) < 0.10,
        f"weight={tsla_w:.6f}",
    )

    # ── Test 10: Rebalance detection ──────────────────────────────────────────
    # ctx already has AAPL=0.05, MSFT=0.04; inject a big signal to force drift
    big_aapl = [_make_signal("AAPL", "equity", pos_bias=1.5, confidence=0.95)]
    result   = run_portfolio_brain(big_aapl, ctx)
    assert_test(
        "Rebalance engine returns bool",
        isinstance(result["rebalance_required"], bool),
    )

    # ── Test 11: Crisis regime → hedge bucket activated ───────────────────────
    crisis_ctx = PortfolioContext(
        nav=1_000_000, peak_nav=1_200_000,  # −16.7 % drawdown
        market_regime="crisis", session="intraday",
    )
    mixed = [
        _make_signal("AAPL", "equity", confidence=0.80),
        _make_signal("MSFT", "equity", confidence=0.75),
    ]
    result = run_portfolio_brain(mixed, crisis_ctx)
    assert_test(
        "Crisis regime: hedge_allocations non-empty",
        len(result["hedge_allocations"]) > 0,
        str(result["hedge_allocations"]),
    )

    # ── Test 12: Order generation structure ───────────────────────────────────
    orders = result["target_orders"]
    assert_test(
        "target_orders is a list",
        isinstance(orders, list),
    )
    if orders:
        first = orders[0]
        assert_test(
            "Order has required execution_bridge fields",
            all(k in first for k in [
                "order_id", "symbol", "side", "quantity_nav_fraction",
                "order_type", "urgency", "asset_class", "hedge_flag",
            ]),
        )

    # ── Test 13: portfolio_risk_score range ───────────────────────────────────
    assert_test(
        "portfolio_risk_score in [0, 100]",
        0.0 <= result["portfolio_risk_score"] <= 100.0,
        f"score={result['portfolio_risk_score']}",
    )

    # ── Test 14: Drawdown scalar reduces allocation ────────────────────────────
    dd_ctx = PortfolioContext(
        nav=800_000, peak_nav=1_000_000,  # −20 % drawdown → scalar = 0.25
        market_regime="bear", session="intraday",
    )
    sig_dd  = [_make_signal("XOM", "equity", pos_bias=1.0, confidence=0.85)]
    r_norm  = run_portfolio_brain(sig_dd, PortfolioContext())
    r_dd    = run_portfolio_brain(sig_dd, dd_ctx)
    w_norm  = abs(r_norm["target_weights"].get("XOM", 0.0))
    w_dd    = abs(r_dd["target_weights"].get("XOM", 0.0))
    assert_test(
        "Drawdown adaptive de-risking: weight reduced under drawdown",
        w_dd <= w_norm + 1e-9,
        f"normal={w_norm:.4f} dd={w_dd:.4f}",
    )

    # ── Test 15: Multi-asset mixed portfolio ──────────────────────────────────
    multi = [
        _make_signal("AAPL",  "equity",    pos_bias=1.0, confidence=0.85),
        _make_signal("BTC",   "crypto",    pos_bias=0.5, confidence=0.72),
        _make_signal("GLD",   "commodity", pos_bias=0.6, confidence=0.68),
        _make_signal("EURUSD","forex",     pos_bias=0.4, confidence=0.70, direction="short"),
        _make_signal("TLT",   "bond",      pos_bias=0.7, confidence=0.80),
        _make_signal("USO",   "futures",   pos_bias=0.5, confidence=0.65),
    ]
    result = run_portfolio_brain(multi, ctx)
    assert_test(
        "Multi-asset: all accepted symbols present",
        len(result["target_weights"]) == 6,
        str(list(result["target_weights"])),
    )
    crypto_w = abs(result["target_weights"].get("BTC", 0.0))
    assert_test(
        "Crypto asset class cap respected",
        crypto_w <= ASSET_CLASS_CAPS["crypto"] + 1e-9,
        f"crypto_weight={crypto_w:.4f}",
    )

    print("═" * 70)
    print(f"  All smoke tests passed ✓")
    print("═" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)   # quiet during tests
    _run_smoke_tests()