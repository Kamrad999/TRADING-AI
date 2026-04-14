"""
rss_sandbox.py
--------------
Sandbox test for fetching and displaying RSS headlines.
Built to be extended into a broader trading AI news engine.

DEPENDENCIES:
  - feedparser (third-party RSS/Atom parser)
  - requests (third-party HTTP client)
  
  These are NOT stdlib and must be installed via:
    pip install -r requirements.txt
"""

try:
    import feedparser
except ImportError:
    raise ImportError(
        "feedparser not installed. Please run:\n"
        "  pip install -r requirements.txt\n"
        "or:\n"
        "  pip install feedparser requests\n"
        "\nWithout feedparser, RSS feed fetching will not work."
    ) from None

# --- Config ---
RSS_URL = "https://finance.yahoo.com/rss/"
MAX_HEADLINES = 5


def fetch_feed(url: str) -> feedparser.FeedParserDict | None:
    """
    Fetch and parse an RSS feed from the given URL.
    Returns the parsed feed, or None if something went wrong.
    """
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        print(f"[ERROR] Failed to parse feed: {e}")
        return None

    # feedparser doesn't raise on HTTP errors — check status manually
    status = getattr(feed, "status", None)
    if status and status >= 400:
        print(f"[ERROR] Feed returned HTTP {status} for URL: {url}")
        return None

    return feed


def print_headlines(feed: feedparser.FeedParserDict, url: str, max_items: int = MAX_HEADLINES) -> None:
    """
    Print the top N headlines from a parsed feed.
    Handles empty feeds gracefully.
    """
    feed_title = feed.feed.get("title", "Unknown Feed")
    entries = feed.get("entries", [])

    print(f"\nSource : {url}")
    print(f"Feed   : {feed_title}")
    print(f"{'─' * 40}")

    if not entries:
        print("[INFO] No entries found in this feed.")
        return

    print(f"Top {min(max_items, len(entries))} Headlines:\n")
    for i, entry in enumerate(entries[:max_items], start=1):
        title = entry.get("title", "[No title]").strip()
        print(f"  {i}. {title}")


def test_single_feed(url: str = RSS_URL) -> None:
    """
    Main test function: fetch a single RSS feed and display its headlines.
    Entry point for future integration with a multi-feed news engine.
    """
    print(f"[INFO] Fetching feed: {url}")

    feed = fetch_feed(url)
    if feed is None:
        print("[INFO] Skipping display — feed could not be loaded.")
        return

    print_headlines(feed, url)


# --- Run ---
if __name__ == "__main__":
    test_single_feed()