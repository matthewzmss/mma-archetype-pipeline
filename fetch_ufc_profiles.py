"""
Fetch each active fighter's UFC.com athlete page (full-body image, pro record, ranking tags, bio)
into ref/ufc_profiles.json. Resumable: fighters already in the cache are skipped, and the cache is
saved after every page, so it can be stopped and restarted at any time.

UFC.com's robots.txt asks for a 15 s crawl delay, so a full run over ~740 fighters takes ~3 hours.

Usage:  python fetch_ufc_profiles.py [--delay 15] [--limit N] [--retry-missing]
Slugs that can't be guessed from the name go in ref/ufc_slug_overrides.json: {"Fighter Name": "ufc-com-slug"}
"""
import argparse, html, json, re, sys, time, unicodedata
import urllib.error, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
CACHE = ROOT / "ref" / "ufc_profiles.json"
OVERRIDES = ROOT / "ref" / "ufc_slug_overrides.json"
BASE = "https://www.ufc.com/athlete/"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"


def slugify(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['.]", "", s)
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def candidates(name):
    s = slugify(name)
    out = [s]
    base = re.sub(r"-(jr|sr|ii|iii)$", "", s)
    if base != s:
        out.append(base)
    out.append(s + "-0")
    return out


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""


def text(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def first(pat, page):
    m = re.search(pat, page, re.S)
    return text(m.group(1)) if m else None


def parse(page):
    img = re.search(r'https://(?:www\.)?ufc\.com/images/styles/athlete_bio_full_body/[^"\s]+', page)
    og = re.search(r'og:image" content="([^"]+)', page)
    img = img.group(0) if img else (og.group(1) if og else None)
    if img:
        img = html.unescape(img).replace("https://ufc.com/", "https://www.ufc.com/")
    p = {
        "name": first(r'class="hero-profile__name">(.*?)</', page),
        "nick": first(r'class="hero-profile__nickname">(.*?)</', page),
        "record": first(r'class="hero-profile__division-body">(.*?)</', page),
        "division": first(r'class="hero-profile__division-title">(.*?)</', page),
        "tags": [text(t) for t in re.findall(r'class="hero-profile__tag">(.*?)</', page, re.S)],
        "img": img,
        "hero": {text(t): text(n) for n, t in re.findall(
            r'class="hero-profile__stat-numb">(.*?)</.*?class="hero-profile__stat-text">(.*?)</', page, re.S)},
        "bio": {},
    }
    p["tags"] = [t for t in p["tags"] if t and not t.endswith("Division")]
    for seg in page.split('class="c-bio__field')[1:]:
        m = re.search(r'class="c-bio__label">(.*?)</div>(.*?)</div>\s*</div>', seg, re.S)
        if m:
            p["bio"][text(m.group(1))] = text(m.group(2))
    return p


def norm(s):
    return slugify(s or "").replace("-", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay", type=float, default=15.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry-missing", action="store_true")
    a = ap.parse_args()

    d = pd.read_csv(ROOT / "out" / "fighter_archetypes.csv").sort_values("UFC_FIGHTS", ascending=False)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}

    todo = [r for r in d.itertuples() if r.fighter_id not in cache
            or (a.retry_missing and cache[r.fighter_id].get("status") != "ok")
            or (r.NAME in overrides and cache[r.fighter_id].get("slug") != overrides[r.NAME])]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(cache)} cached, {len(todo)} to fetch (~{len(todo) * a.delay / 60:.0f} min)", flush=True)

    for i, r in enumerate(todo, 1):
        slugs = [overrides[r.NAME]] if r.NAME in overrides else candidates(r.NAME)
        rec = {"status": "not_found", "tried": slugs}
        for slug in slugs:
            status, page = get(BASE + slug)
            time.sleep(a.delay)
            if status == 200 and "hero-profile__name" in page:
                p = parse(page)
                ok = r.NAME in overrides or norm(p["name"]) == norm(r.NAME)
                rec = {"status": "ok" if ok else "name_mismatch", "slug": slug, "url": BASE + slug, **p}
                break
            if status not in (404, 200):
                rec = {"status": "error", "http": status, "tried": slugs}
                break
        cache[r.fighter_id] = rec
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
        tmp.replace(CACHE)
        print(f"[{i}/{len(todo)}] {r.NAME}: {rec['status']}" + (f" ({rec.get('slug')})" if rec.get("slug") else ""), flush=True)

    bad = {k: v for k, v in cache.items() if v.get("status") != "ok"}
    if bad:
        names = d.set_index("fighter_id").NAME
        print(f"\n{len(bad)} without a matched profile (add to {OVERRIDES.name} if needed):")
        for k, v in bad.items():
            print(f"  {names.get(k, k)}: {v['status']} {v.get('slug') or v.get('tried')}")


if __name__ == "__main__":
    main()
