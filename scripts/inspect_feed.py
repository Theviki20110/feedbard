import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter


import requests

NS = {
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

# Frasi candidate: quale usa davvero Substack per troncare i post a pagamento?
CANDIDATES = [
    "post is for paid",
    "post is for paying",
    "subscribe to read",
    "this post is only for",
    "keep reading with a 7-day free trial",
    "paid subscribers",
    "become a paid subscriber",
]


def strip_tags(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()




def main() -> None:
    url = sys.argv[1] if len(sys.argv) > 1 else sys.exit("uso: inspect_feed.py <url>")
    r = requests.get(url, headers={"User-Agent": "substack-podcast/1.0"}, timeout=30)

    print(f"=== HTTP ===  {r.status_code}  {len(r.content):,} byte")
    for h in ("ETag", "Last-Modified", "Cache-Control", "Content-Type"):
        print(f"  {h:16} {r.headers.get(h) or '<assente>'}")
    if not (r.headers.get("ETag") or r.headers.get("Last-Modified")):
        print("  !! nessun validatore: il GET condizionale non servira' a nulla")

    fixture = "real_feed.xml"
    with open(fixture, "wb") as fh:
        fh.write(r.content)

    root = ET.fromstring(r.content)
    ch = root.find("channel")
    items = ch.findall("item")
    print(f"\n=== CANALE ===  {(ch.findtext('title') or '?')}   item: {len(items)}")
    print(f"  namespace dichiarati: {sorted(set(re.findall(r'xmlns:(\w+)=', r.text[:2000])))}")

    tags = Counter()
    for it in items:
        for child in it:
            tags[re.sub(r"\{[^}]+\}", lambda m: m.group(0).split('/')[-1].rstrip('}') + ":", child.tag)] += 1
    print(f"  tag negli item: {dict(tags)}")

    print(f"\n=== ITEM ===")
    marker_hits = Counter()
    for it in items:
        title = (it.findtext("title") or "?")[:52]
        guid_el = it.find("guid")
        guid = (guid_el.text or "")[:60] if guid_el is not None else "<assente>"
        perma = guid_el.get("isPermaLink") if guid_el is not None else None
        content = it.findtext("content:encoded", namespaces=NS) or ""
        desc = it.findtext("description") or ""
        body = content or desc
        words = len(strip_tags(body).split())
        low = strip_tags(body).lower()
        hits = [c for c in CANDIDATES if c in low]
        for h in hits:
            marker_hits[h] += 1
        enc = it.find("enclosure")
        audio = "audio" in (enc.get("type") or "") if enc is not None else False
        print(f"  {words:>6} parole  content:encoded={'si' if content else 'NO':2}  "
              f"audio={'si' if audio else 'no'}  {title}")
        if hits:
            print(f"          marcatori: {hits}")

    print(f"\n=== VERDETTO ===")
    print(f"  content:encoded presente: {sum(1 for i in items if i.findtext('content:encoded', namespaces=NS))}/{len(items)}")
    print(f"  guid isPermaLink: {perma!r}   forma: {'URL' if guid.startswith('http') else 'opaco'}")
    print(f"  marcatori paywall trovati: {dict(marker_hits) or 'NESSUNO'}")
    print(f"  enclosure audio: {sum(1 for i in items if (i.find('enclosure') is not None and 'audio' in (i.find('enclosure').get('type') or '')))}/{len(items)}")
    print(f"\n  fixture salvata in {fixture}")


if __name__ == "__main__":
    main()