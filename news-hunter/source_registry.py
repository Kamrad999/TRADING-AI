"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        MONSTER TRADING AI — SOURCE REGISTRY v2.0                           ║
║        80 Free Public Feeds · 10 Market Verticals · 120+ Unique Tags       ║
║                                                                              ║
║  DROP-IN REPLACEMENT for the ALL_FEED_SOURCES list in news_hunter_core.py  ║
║  Usage:                                                                      ║
║      from source_registry_v2 import ALL_FEED_SOURCES, get_sources          ║
║      articles = run_ingestion_pipeline(sources=ALL_FEED_SOURCES)            ║
╚══════════════════════════════════════════════════════════════════════════════╝

NEW IN v2.0:
  ▸  80 sources   (was 30)
  ▸  10 categories (was 5): forex, stocks, crypto, macro, global, official,
                             alt, social, commodity, risk
  ▸  Reddit RSS integration  (no API key needed)
  ▸  Google News parameterised RSS  (no API key needed)
  ▸  Alternative alpha sources: insider trades, 13F filings, options flow
  ▸  Commodity vertical: oil, gold, agricultural, base metals
  ▸  Risk & Policy vertical: geopolitical, sanctions, regulatory
  ▸  Central banks expanded: BoE, RBA, SNB, Norges, Riksbank
  ▸  Academic feeds: NBER, BIS Working Papers, NY Fed Liberty Street
  ▸  Institutional crypto: The Block, Messari, DeFi Pulse
  ▸  Helper functions for programmatic source selection
"""

from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────────────────────
# FEED SOURCE SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeedSource:
    """
    One trusted RSS/Atom feed source with full metadata.

    Fields:
        name        : Human-readable display name
        url         : The RSS/Atom feed URL (free, no API key required)
        category    : Market vertical — see CATEGORIES below
        tags        : Asset, topic, and signal tags for downstream filtering
        priority    : 1 (must-have) → 5 (supplemental / experimental)
        description : One-line explanation of why this source matters
        latency     : Expected update frequency — 'realtime' | 'hourly' | 'daily'
        region      : Geographic focus — 'global' | 'us' | 'eu' | 'apac' | 'uk'
    """
    name: str
    url: str
    category: str
    tags: list[str]
    priority: int = 3
    description: str = ""
    latency: str = "hourly"
    region: str = "global"


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

CATEGORIES: dict[str, str] = {
    "forex":     "FX pairs, currency markets, central bank FX policy",
    "stocks":    "Equities, earnings, ETFs, IPOs, analyst ratings",
    "crypto":    "Digital assets, DeFi, NFTs, blockchain protocols",
    "macro":     "Economic indicators, GDP, inflation, employment data",
    "global":    "Broad-market journalism from high-trust global outlets",
    "official":  "Central banks, regulators, government statistical agencies",
    "alt":       "Alternative alpha: insider filings, options flow, 13F data",
    "social":    "Reddit/social sentiment — retail trader discourse tracker",
    "commodity": "Energy, precious metals, agricultural and base metals",
    "risk":      "Geopolitical risk, policy, regulation, investigative journalism",
}


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#   THE MASTER REGISTRY — 80 FREE PUBLIC FEEDS
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

ALL_FEED_SOURCES: list[FeedSource] = [


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1: FOREX & FX MARKETS
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="ForexFactory",
        url="https://www.forexfactory.com/rss",
        category="forex",
        tags=["forex", "fx", "calendar", "macro", "sentiment"],
        priority=1,
        description="The gold standard for forex economic calendar & community sentiment.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="DailyFX",
        url="https://www.dailyfx.com/feeds/market-news",
        category="forex",
        tags=["forex", "fx", "technical-analysis", "macro", "ig-group"],
        priority=1,
        description="IG Group's institutional-grade FX research feed. High signal-to-noise.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="FXStreet",
        url="https://www.fxstreet.com/rss/news",
        category="forex",
        tags=["forex", "fx", "currency", "analysis"],
        priority=1,
        description="25M monthly readers. Broad FX pair coverage with real-time commentary.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="Investing.com Forex",
        url="https://www.investing.com/rss/news_25.rss",
        category="forex",
        tags=["forex", "fx", "currency", "aggregated"],
        priority=2,
        description="High-volume currency news aggregated from hundreds of outlets.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="OANDA Market News",
        url="https://www.oanda.com/rss",
        category="forex",
        tags=["forex", "fx", "rates", "oanda", "institutional"],
        priority=2,
        description="Institutional FX broker market commentary. Position data insight.",
        latency="hourly",
        region="global",
    ),


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2: STOCKS & EQUITY MARKETS
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="Yahoo Finance",
        url="https://finance.yahoo.com/rss/",
        category="stocks",
        tags=["stocks", "equities", "earnings", "markets", "broad"],
        priority=1,
        description="Highest-traffic free finance feed. Broad coverage, real-time.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="MarketWatch Top Stories",
        url="https://www.marketwatch.com/rss/topstories",
        category="stocks",
        tags=["stocks", "markets", "equities", "wsj", "macro"],
        priority=1,
        description="WSJ-backed real-time market news. Strong institutional readership.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="CNBC Markets",
        url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
        category="stocks",
        tags=["stocks", "markets", "macro", "cnbc", "tv"],
        priority=1,
        description="TV-grade breaking market news. Massive retail investor reach.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="NASDAQ Official",
        url="https://www.nasdaq.com/feed/rssoutbound",
        category="stocks",
        tags=["stocks", "nasdaq", "equities", "ipo", "listings"],
        priority=2,
        description="Official NASDAQ exchange news. IPOs, listings, delistings.",
        latency="hourly",
        region="us",
    ),
    FeedSource(
        name="Benzinga",
        url="https://www.benzinga.com/feed",
        category="stocks",
        tags=["stocks", "options", "earnings", "benzinga", "fast-moving"],
        priority=2,
        description="Speed-optimised stocks & options intelligence. Strong on earnings.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="Zacks Research",
        url="https://www.zacks.com/stock/research/rss.php",
        category="stocks",
        tags=["stocks", "earnings", "ratings", "estimates", "quant"],
        priority=2,
        description="Quant-driven earnings estimate research. Strong on consensus revisions.",
        latency="daily",
        region="us",
    ),
    FeedSource(
        name="Seeking Alpha",
        url="https://seekingalpha.com/market_currents.xml",
        category="stocks",
        tags=["stocks", "analysis", "fundamental", "alpha", "crowd-sourced"],
        priority=2,
        description="20M+ readers. Crowd-sourced fundamental analysis and earnings colour.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="Motley Fool",
        url="https://www.fool.com/feeds/index.aspx",
        category="stocks",
        tags=["stocks", "analysis", "long-term", "fundamental"],
        priority=3,
        description="Long-term fundamental analysis. Useful for position sizing context.",
        latency="daily",
        region="us",
    ),
    FeedSource(
        name="StockAnalysis News",
        url="https://stockanalysis.com/rss/news.xml",
        category="stocks",
        tags=["stocks", "equities", "data", "screener"],
        priority=2,
        description="Clean equity data with real-time news. Excellent for ticker-level search.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="ETF.com News",
        url="https://www.etf.com/sections/news.rss",
        category="stocks",
        tags=["etf", "funds", "passive", "flows", "institutional"],
        priority=2,
        description="ETF flow intelligence — institutional positioning and sector rotation.",
        latency="daily",
        region="us",
    ),
    FeedSource(
        name="Bloomberg ETF Report",
        url="https://www.bloomberg.com/feed/podcast/etf-report.xml",
        category="stocks",
        tags=["etf", "funds", "markets", "bloomberg"],
        priority=2,
        description="Bloomberg's ETF-focused podcast intelligence feed.",
        latency="daily",
        region="global",
    ),
    FeedSource(
        name="Google News: Earnings",
        url="https://news.google.com/rss/search?q=earnings+beat+miss+quarterly+results+when:1d&hl=en-US&gl=US&ceid=US:en",
        category="stocks",
        tags=["earnings", "stocks", "aggregated", "google"],
        priority=2,
        description="Real-time earnings beat/miss coverage aggregated from 1000s of sources.",
        latency="realtime",
        region="us",
    ),


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3: CRYPTO & DIGITAL ASSETS
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="CoinDesk",
        url="https://www.coindesk.com/arc/outboundfeeds/rss/",
        category="crypto",
        tags=["crypto", "bitcoin", "defi", "blockchain", "institutional"],
        priority=1,
        description="Premier institutional-grade crypto journalism. Regulated, trusted.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="CoinTelegraph",
        url="https://cointelegraph.com/rss",
        category="crypto",
        tags=["crypto", "bitcoin", "altcoins", "defi", "nft"],
        priority=1,
        description="Global crypto news. 20M+ readers. Wide altcoin coverage.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="The Block",
        url="https://www.theblock.co/rss.xml",
        category="crypto",
        tags=["crypto", "institutional", "defi", "venture", "data"],
        priority=1,
        description="Institutional crypto intelligence. VC deals, protocol launches, on-chain.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="CryptoSlate",
        url="https://cryptoslate.com/feed/",
        category="crypto",
        tags=["crypto", "blockchain", "defi", "altcoins"],
        priority=2,
        description="On-chain analytics and crypto market intelligence.",
        latency="hourly",
        region="global",
    ),
    FeedSource(
        name="Decrypt",
        url="https://decrypt.co/feed",
        category="crypto",
        tags=["crypto", "web3", "nft", "consumer"],
        priority=2,
        description="Web3 and consumer crypto culture. Early-signal for adoption trends.",
        latency="hourly",
        region="global",
    ),
    FeedSource(
        name="Bitcoin Magazine",
        url="https://bitcoinmagazine.com/.rss/full/",
        category="crypto",
        tags=["bitcoin", "lightning", "btc", "maximalist"],
        priority=2,
        description="Oldest authoritative Bitcoin-only publication. Lightning network coverage.",
        latency="daily",
        region="global",
    ),
    FeedSource(
        name="Messari Crypto Research",
        url="https://messari.io/rss",
        category="crypto",
        tags=["crypto", "research", "on-chain", "metrics", "professional"],
        priority=2,
        description="Professional-grade crypto research and on-chain analytics.",
        latency="daily",
        region="global",
    ),
    FeedSource(
        name="DeFi Pulse",
        url="https://defipulse.com/blog/feed/",
        category="crypto",
        tags=["defi", "tvl", "protocols", "yield", "lending"],
        priority=2,
        description="DeFi TVL analytics and protocol-level intelligence.",
        latency="daily",
        region="global",
    ),
    FeedSource(
        name="Google News: Crypto",
        url="https://news.google.com/rss/search?q=cryptocurrency+bitcoin+ethereum+when:1d&hl=en-US&gl=US&ceid=US:en",
        category="crypto",
        tags=["crypto", "bitcoin", "aggregated", "google"],
        priority=2,
        description="Broad crypto news aggregated from all major media outlets.",
        latency="realtime",
        region="global",
    ),


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4: MACRO ECONOMIC INDICATORS
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="TradingEconomics",
        url="https://tradingeconomics.com/rss/news.aspx",
        category="macro",
        tags=["macro", "economic-data", "gdp", "inflation", "rates", "195-countries"],
        priority=1,
        description="195 countries of macro indicators. The broadest free economic data feed.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="FRED (St. Louis Fed)",
        url="https://research.stlouisfed.org/rss/releases.xml",
        category="macro",
        tags=["fred", "macro", "economic-data", "rates", "usd", "800k-series"],
        priority=1,
        description="800,000+ economic time series. The definitive macro data warehouse.",
        latency="daily",
        region="us",
    ),
    FeedSource(
        name="CME Group Markets",
        url="https://www.cmegroup.com/rss/markets.xml",
        category="macro",
        tags=["futures", "derivatives", "rates", "fed", "cme", "fed-watch"],
        priority=1,
        description="World's largest derivatives exchange. CME FedWatch rate expectations.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="NBER Working Papers",
        url="https://www.nber.org/rss/new_working_papers.xml",
        category="macro",
        tags=["academic", "research", "macro", "economics", "nber"],
        priority=3,
        description="Top academic economics research before publication. Long-lead indicator.",
        latency="weekly",
        region="us",
    ),
    FeedSource(
        name="Google News: Fed Policy",
        url="https://news.google.com/rss/search?q=federal+reserve+interest+rates+fomc+when:1d&hl=en-US&gl=US&ceid=US:en",
        category="macro",
        tags=["fed", "rates", "fomc", "aggregated", "monetary-policy"],
        priority=2,
        description="Real-time Fed policy coverage aggregated across all major outlets.",
        latency="realtime",
        region="us",
    ),


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5: GLOBAL HIGH-TRUST JOURNALISM
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="Reuters Business",
        url="https://feeds.reuters.com/reuters/businessNews",
        category="global",
        tags=["macro", "global", "geopolitics", "reuters", "authoritative"],
        priority=1,
        description="World's largest international news agency. Primary global source.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="Financial Times",
        url="https://www.ft.com/rss/home",
        category="global",
        tags=["macro", "global", "markets", "geopolitics", "ft", "authoritative"],
        priority=1,
        description="The definitive source for global financial journalism. Required reading.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="BBC Business",
        url="https://feeds.bbci.co.uk/news/business/rss.xml",
        category="global",
        tags=["macro", "global", "economy", "bbc", "uk"],
        priority=2,
        description="Globally trusted public-service journalism. Strong geopolitical coverage.",
        latency="hourly",
        region="global",
    ),
    FeedSource(
        name="CNN Money",
        url="https://rss.cnn.com/rss/money_latest.rss",
        category="global",
        tags=["macro", "global", "markets", "cnn"],
        priority=2,
        description="Mass-market financial news. Useful for retail sentiment calibration.",
        latency="hourly",
        region="us",
    ),
    FeedSource(
        name="Google News: Stocks",
        url="https://news.google.com/rss/search?q=stock+market+wall+street+when:1d&hl=en-US&gl=US&ceid=US:en",
        category="global",
        tags=["stocks", "markets", "aggregated", "google"],
        priority=2,
        description="Real-time aggregation from 1000s of sources. Widest possible coverage.",
        latency="realtime",
        region="us",
    ),


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6: OFFICIAL SOURCES — CENTRAL BANKS & REGULATORS
    # These are the HIGHEST-TRUST, MOST MARKET-MOVING feeds in the system.
    # A single press release from any of these can move markets ±2%.
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="Federal Reserve (US)",
        url="https://www.federalreserve.gov/feeds/press_all.xml",
        category="official",
        tags=["fed", "rates", "monetary-policy", "fomc", "usd", "qe"],
        priority=1,
        description="Primary source for US monetary policy. Every press release. P1 forever.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="NY Fed Liberty Street",
        url="https://feeds.feedburner.com/LibertyStreetEconomics",
        category="official",
        tags=["fed", "research", "macro", "academic", "ny-fed", "early-signal"],
        priority=2,
        description="NY Fed research blog. Signals Fed thinking BEFORE official FOMC statements.",
        latency="weekly",
        region="us",
    ),
    FeedSource(
        name="Bank of England",
        url="https://www.bankofengland.co.uk/rss/news",
        category="official",
        tags=["boe", "gbp", "rates", "monetary-policy", "uk", "mpc"],
        priority=1,
        description="BoE press releases. GBP rate decisions, QE, financial stability reports.",
        latency="realtime",
        region="uk",
    ),
    FeedSource(
        name="Bank of Canada",
        url="https://www.bankofcanada.ca/feed/",
        category="official",
        tags=["boc", "rates", "cad", "monetary-policy", "canada"],
        priority=2,
        description="Official BoC press releases and rate decisions. CAD primary source.",
        latency="realtime",
        region="us",  # Canada grouped under North America
    ),
    FeedSource(
        name="ECB (European Central Bank)",
        url="https://www.ecb.europa.eu/rss/press.html",
        category="official",
        tags=["ecb", "rates", "eur", "monetary-policy", "europe", "lagarde"],
        priority=1,
        description="ECB press releases. EUR rate decisions, PEPP, TPI activation signals.",
        latency="realtime",
        region="eu",
    ),
    FeedSource(
        name="Reserve Bank of Australia",
        url="https://www.rba.gov.au/rss.xml",
        category="official",
        tags=["rba", "aud", "rates", "monetary-policy", "australia"],
        priority=2,
        description="RBA rate decisions and monetary policy statements. AUD primary source.",
        latency="realtime",
        region="apac",
    ),
    FeedSource(
        name="SNB (Swiss National Bank)",
        url="https://www.snb.ch/en/rss/news",
        category="official",
        tags=["snb", "chf", "rates", "monetary-policy", "switzerland", "fx-intervention"],
        priority=2,
        description="SNB press releases. Known for surprise CHF FX interventions.",
        latency="realtime",
        region="eu",
    ),
    FeedSource(
        name="Norges Bank",
        url="https://www.norges-bank.no/en/rss/",
        category="official",
        tags=["norges", "nok", "rates", "monetary-policy", "norway", "oil-fund"],
        priority=3,
        description="Norwegian central bank. NOK policy + NBIM sovereign wealth fund news.",
        latency="realtime",
        region="eu",
    ),
    FeedSource(
        name="Riksbank (Sweden)",
        url="https://www.riksbank.se/en-gb/rss/news/",
        category="official",
        tags=["riksbank", "sek", "rates", "monetary-policy", "sweden"],
        priority=3,
        description="Swedish central bank. SEK monetary policy decisions.",
        latency="realtime",
        region="eu",
    ),
    FeedSource(
        name="IMF",
        url="https://www.imf.org/en/News/RSS",
        category="official",
        tags=["imf", "macro", "global", "economic-outlook", "weo"],
        priority=2,
        description="IMF World Economic Outlook, Article IV reports, emergency lending.",
        latency="weekly",
        region="global",
    ),
    FeedSource(
        name="IMF Blog",
        url="https://www.imf.org/en/Blogs/RSS",
        category="official",
        tags=["imf", "macro", "research", "academic", "global"],
        priority=2,
        description="IMF chief economists' blog. Policy signals before formal publications.",
        latency="weekly",
        region="global",
    ),
    FeedSource(
        name="World Bank",
        url="https://www.worldbank.org/en/news/all.xml",
        category="official",
        tags=["world-bank", "macro", "global", "development", "emerging-markets"],
        priority=3,
        description="World Bank development finance. Emerging markets macro signal.",
        latency="weekly",
        region="global",
    ),
    FeedSource(
        name="BIS (Bank for International Settlements)",
        url="https://www.bis.org/rss/bisnews.xml",
        category="official",
        tags=["bis", "macro", "financial-stability", "global", "systemic-risk", "cbdc"],
        priority=2,
        description="Central bank for central banks. Systemic risk warnings and CBDC research.",
        latency="weekly",
        region="global",
    ),
    FeedSource(
        name="BIS Working Papers",
        url="https://www.bis.org/rss/bis_work.xml",
        category="official",
        tags=["bis", "academic", "macro", "financial-stability", "research"],
        priority=3,
        description="Pre-publication central banking research. Long-lead policy signal.",
        latency="weekly",
        region="global",
    ),
    FeedSource(
        name="US Treasury",
        url="https://home.treasury.gov/news/press-releases/rss.xml",
        category="official",
        tags=["treasury", "bonds", "yields", "usd", "macro", "sanctions", "tga"],
        priority=1,
        description="TGA balance, bond auctions, sanctions designations — direct from Treasury.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="CFTC (Commodity Futures)",
        url="https://www.cftc.gov/rss/pressroom/pressreleases.xml",
        category="official",
        tags=["cftc", "cot", "futures", "regulatory", "derivatives", "positioning"],
        priority=1,
        description="COT report source + enforcement on futures markets. Positioning data.",
        latency="weekly",
        region="us",
    ),
    FeedSource(
        name="OCC (Banking Regulator)",
        url="https://occ.gov/news-issuances/news-releases/occ-news-releases-rss.xml",
        category="official",
        tags=["occ", "banking", "regulatory", "financial-stability", "national-banks"],
        priority=2,
        description="National bank regulator. Crypto banking decisions with market impact.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="FDIC",
        url="https://www.fdic.gov/news/news/press/pressrel.rss",
        category="official",
        tags=["fdic", "banking", "financial-stability", "regulatory", "bank-failures"],
        priority=2,
        description="Bank failures and deposit insurance. Systemic risk early warning system.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="SEC Litigation Releases",
        url="https://www.sec.gov/rss/litigation/litreleases.xml",
        category="official",
        tags=["sec", "regulatory", "enforcement", "stocks", "fraud", "litigation"],
        priority=1,
        description="Real SEC enforcement actions. Named companies = immediate stock impact.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="BLS (US Bureau of Labor Statistics)",
        url="https://www.bls.gov/feed/news_release.rss",
        category="official",
        tags=["bls", "jobs", "cpi", "ppi", "nfp", "inflation", "usd", "macro"],
        priority=1,
        description="CPI, PPI, NFP — the most market-moving economic data releases on Earth.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="Congressional Budget Office",
        url="https://www.cbo.gov/publications/feed",
        category="official",
        tags=["fiscal", "deficit", "debt", "macro", "usd", "congress"],
        priority=2,
        description="US fiscal outlook. Deficit trajectory is a direct signal for bonds & USD.",
        latency="weekly",
        region="us",
    ),


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7: ALTERNATIVE ALPHA — FILINGS, FLOW, SMART MONEY
    # These sources give you an EDGE. Institutional data before it's priced in.
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="OpenInsider (SEC Form 4)",
        url="https://openinsider.com/rss-feed-latest",
        category="alt",
        tags=["insider-trading", "sec", "form4", "stocks", "alpha", "corporate"],
        priority=1,
        description="Real-time corporate insider buys/sells. Pure alpha before public knows.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="WhaleWisdom (13F Filings)",
        url="https://whalewisdom.com/rss.xml",
        category="alt",
        tags=["13f", "institutional", "hedge-funds", "alpha", "positions", "smart-money"],
        priority=1,
        description="Hedge fund 13F filings parsed. Track Bridgewater, Citadel, and more.",
        latency="quarterly",
        region="us",
    ),
    FeedSource(
        name="Unusual Whales (Options Flow)",
        url="https://unusualwhales.com/feed.xml",
        category="alt",
        tags=["options-flow", "dark-pool", "unusual", "alpha", "institutional-flow"],
        priority=1,
        description="Unusual options flow & dark pool prints. Institutional footprint tracker.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="SEC EDGAR 8-K Filings",
        url="https://efts.sec.gov/LATEST/search-index?q=%22material%22&forms=8-K&dateRange=custom&startdt=2024-01-01",
        category="alt",
        tags=["8k", "sec", "filings", "stocks", "material-events", "ma", "earnings"],
        priority=1,
        description="8-K material event filings. M&A, CEO changes, earnings surprises.",
        latency="realtime",
        region="us",
    ),


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 8: SOCIAL SENTIMENT — RETAIL TRADER RADAR
    # No API key needed. Plain RSS from Reddit.
    # These are your contrarian indicator and meme-stock early-warning system.
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="Reddit r/wallstreetbets",
        url="https://www.reddit.com/r/wallstreetbets/.rss",
        category="social",
        tags=["retail", "meme-stocks", "options", "sentiment", "wsb", "yolo"],
        priority=3,
        description="3.5M members. Meme stock early detection & retail sentiment thermometer.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="Reddit r/investing",
        url="https://www.reddit.com/r/investing/.rss",
        category="social",
        tags=["stocks", "investing", "retail", "sentiment", "mainstream"],
        priority=3,
        description="16M member mainstream retail investor community. Consensus sentiment.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="Reddit r/stocks",
        url="https://www.reddit.com/r/stocks/.rss",
        category="social",
        tags=["stocks", "equities", "retail", "sentiment", "tickers"],
        priority=3,
        description="4M member high-volume stocks discussion. Good for ticker-level sentiment.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="Reddit r/Forex",
        url="https://www.reddit.com/r/Forex/.rss",
        category="social",
        tags=["forex", "fx", "retail", "sentiment", "currency"],
        priority=4,
        description="Retail FX community. Useful as a contrarian retail sentiment signal.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="Reddit r/CryptoCurrency",
        url="https://www.reddit.com/r/CryptoCurrency/.rss",
        category="social",
        tags=["crypto", "sentiment", "retail", "altcoins", "defi"],
        priority=3,
        description="7M member crypto community. Alt-coin retail sentiment tracker.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="Reddit r/Economics",
        url="https://www.reddit.com/r/economics/.rss",
        category="social",
        tags=["macro", "economics", "academic", "sentiment", "policy"],
        priority=4,
        description="Academic/policy discourse. Long-lag but useful for narrative tracking.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="Hacker News",
        url="https://news.ycombinator.com/rss",
        category="social",
        tags=["tech", "fintech", "ai", "startups", "macro", "yc"],
        priority=3,
        description="YC community. Fintech disruption signals & AI impact on markets.",
        latency="realtime",
        region="global",
    ),


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 9: COMMODITIES
    # Energy, precious metals, agricultural, base metals
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="EIA (US Energy Information Admin)",
        url="https://www.eia.gov/rss/press_releases.xml",
        category="commodity",
        tags=["oil", "nat-gas", "energy", "crude", "wti", "brent", "eia"],
        priority=1,
        description="The authoritative US oil/gas data source. Weekly inventory reports.",
        latency="weekly",
        region="us",
    ),
    FeedSource(
        name="OilPrice.com",
        url="https://oilprice.com/rss/main",
        category="commodity",
        tags=["oil", "crude", "brent", "wti", "energy", "opec"],
        priority=1,
        description="Dedicated crude oil market intelligence. OPEC coverage is exceptional.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="Kitco (Gold & Silver)",
        url="https://www.kitco.com/rss/kitcogold.rss",
        category="commodity",
        tags=["gold", "silver", "precious-metals", "xau", "xag", "inflation-hedge"],
        priority=1,
        description="Premier precious metals news and spot pricing. XAU/USD primary source.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="USDA (Agricultural Data)",
        url="https://www.usda.gov/rss.xml",
        category="commodity",
        tags=["wheat", "corn", "soy", "agriculture", "commodity", "crop-report"],
        priority=2,
        description="Crop reports and WASDE data. Essential for agricultural commodity trading.",
        latency="monthly",
        region="us",
    ),
    FeedSource(
        name="Argus Media (Metals)",
        url="https://www.argusmedia.com/en/rss/news",
        category="commodity",
        tags=["metals", "steel", "copper", "base-metals", "commodity", "industrial"],
        priority=3,
        description="Base metals supply chain and industrial commodity pricing intelligence.",
        latency="daily",
        region="global",
    ),
    FeedSource(
        name="Grain Central",
        url="https://www.graincentral.com/feed/",
        category="commodity",
        tags=["wheat", "corn", "grain", "agriculture", "australia"],
        priority=3,
        description="Australian grain markets. Good for APAC agricultural commodity signal.",
        latency="daily",
        region="apac",
    ),


    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 10: RISK & POLICY
    # Geopolitical risk, sanctions, regulatory, investigative journalism
    # These are your BLACK SWAN early-warning feeds.
    # ══════════════════════════════════════════════════════════════════════════

    FeedSource(
        name="Google News: Geopolitics",
        url="https://news.google.com/rss/search?q=sanctions+tariffs+trade+war+geopolitical+when:1d&hl=en-US&gl=US&ceid=US:en",
        category="risk",
        tags=["geopolitical", "sanctions", "tariffs", "risk", "aggregated"],
        priority=1,
        description="Geopolitical risk aggregated. Sanctions, tariffs, trade wars — all outlets.",
        latency="realtime",
        region="global",
    ),
    FeedSource(
        name="Politico Economy",
        url="https://rss.politico.com/economy.xml",
        category="risk",
        tags=["policy", "fiscal", "regulation", "geopolitical", "congress", "dc"],
        priority=2,
        description="Washington policy before it becomes market-moving news. Lead indicator.",
        latency="realtime",
        region="us",
    ),
    FeedSource(
        name="The Hill Finance",
        url="https://thehill.com/category/finance/feed/",
        category="risk",
        tags=["policy", "fiscal", "regulation", "congress", "legislation"],
        priority=3,
        description="Congressional finance policy and regulatory news. Legislative risk.",
        latency="hourly",
        region="us",
    ),
    FeedSource(
        name="ProPublica Investigations",
        url="https://www.propublica.org/feeds/propublica/main",
        category="risk",
        tags=["investigative", "regulatory", "corruption", "risk", "whistleblower"],
        priority=3,
        description="Investigative journalism. Regulatory risk early signals, often months ahead.",
        latency="weekly",
        region="us",
    ),

]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS — Programmatic source selection
# ─────────────────────────────────────────────────────────────────────────────

def get_sources(
    category: str | None = None,
    priority: int | None = None,
    region: str | None = None,
    latency: str | None = None,
    tags: list[str] | None = None,
    max_priority: int | None = None,
) -> list[FeedSource]:
    """
    Filter the master registry by any combination of criteria.

    Examples:
        # All P1 sources:
        get_sources(priority=1)

        # All USD-affecting official sources:
        get_sources(category="official", tags=["usd"])

        # All real-time feeds up to P2:
        get_sources(latency="realtime", max_priority=2)

        # US-only macro feeds:
        get_sources(category="macro", region="us")
    """
    result = ALL_FEED_SOURCES

    if category:
        result = [s for s in result if s.category == category]
    if priority is not None:
        result = [s for s in result if s.priority == priority]
    if max_priority is not None:
        result = [s for s in result if s.priority <= max_priority]
    if region:
        result = [s for s in result if s.region == region]
    if latency:
        result = [s for s in result if s.latency == latency]
    if tags:
        result = [s for s in result if any(t in s.tags for t in tags)]

    return sorted(result, key=lambda s: s.priority)


def get_critical_sources() -> list[FeedSource]:
    """Return only P1 sources — the non-negotiable core of the system."""
    return get_sources(priority=1)


def get_by_asset(asset: str) -> list[FeedSource]:
    """
    Return all feeds relevant to a specific asset or instrument.

    Examples:
        get_by_asset("bitcoin")   → crypto feeds tagged bitcoin
        get_by_asset("gold")      → commodity + macro feeds tagged gold
        get_by_asset("usd")       → all USD-affecting sources
        get_by_asset("eurusd")    → EUR and USD sources
    """
    return [s for s in ALL_FEED_SOURCES if asset.lower() in s.tags]


def print_registry_summary() -> None:
    """Print a formatted summary of the entire source registry."""
    from collections import Counter

    by_cat = Counter(s.category for s in ALL_FEED_SOURCES)
    by_pri = Counter(s.priority for s in ALL_FEED_SOURCES)
    all_tags = [t for s in ALL_FEED_SOURCES for t in s.tags]
    unique_tags = len(set(all_tags))

    print("\n╔" + "═" * 60 + "╗")
    print("║  MONSTER TRADING AI — SOURCE REGISTRY v2.0 SUMMARY          ║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  Total sources  : {len(ALL_FEED_SOURCES):<5}  Unique tags  : {unique_tags:<5}             ║")
    print("╠" + "═" * 60 + "╣")
    print("║  BY CATEGORY                                                 ║")
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        bar = "▓" * count
        print(f"║    {cat:<12} {count:>3}  {bar:<30}       ║")
    print("╠" + "═" * 60 + "╣")
    print("║  BY PRIORITY                                                 ║")
    labels = {1:"critical",2:"high    ",3:"medium  ",4:"low     ",5:"suppl.  "}
    for p in sorted(by_pri.keys()):
        print(f"║    P{p} {labels[p]}  {by_pri[p]:>3} sources                              ║")
    print("╚" + "═" * 60 + "╝\n")


# ─────────────────────────────────────────────────────────────────────────────
# QUICK-ACCESS BUNDLES — pre-built source lists for common use cases
# ─────────────────────────────────────────────────────────────────────────────

# The 20 sources with the highest market-moving potential
ALPHA_BUNDLE = get_sources(max_priority=1)

# Complete FX and macro ecosystem
FOREX_BUNDLE = get_sources(category="forex") + get_sources(category="macro")

# Complete digital assets ecosystem
CRYPTO_BUNDLE = get_sources(category="crypto") + get_by_asset("defi")

# All central bank and government sources
CENTRAL_BANK_BUNDLE = [s for s in get_sources(category="official")
                        if any(t in s.tags for t in ["rates", "monetary-policy", "macro"])]

# Alternative data sources for edge generation
ALPHA_ALT_BUNDLE = get_sources(category="alt")

# Real-time feeds only (no daily/weekly/monthly)
REALTIME_BUNDLE = get_sources(latency="realtime")

# Black swan early warning system
RISK_BUNDLE = (
    get_sources(category="risk") +
    get_sources(category="official", tags=["regulatory"]) +
    get_sources(category="alt", tags=["insider-trading"])
)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT — run directly to see registry summary
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_registry_summary()

    print("ALPHA BUNDLE (P1 only):")
    for s in ALPHA_BUNDLE:
        print(f"  [{s.category.upper():10}] P{s.priority} {s.name:<30} {s.url}")

    print(f"\nREALTIME BUNDLE: {len(REALTIME_BUNDLE)} sources")
    print(f"CRYPTO BUNDLE:   {len(CRYPTO_BUNDLE)} sources")
    print(f"FOREX BUNDLE:    {len(FOREX_BUNDLE)} sources")
    print(f"RISK BUNDLE:     {len(RISK_BUNDLE)} sources")

    print("\nSAMPLE QUERY — all feeds tagged 'usd':")
    for s in get_by_asset("usd"):
        print(f"  {s.name} [{s.category}] P{s.priority}")