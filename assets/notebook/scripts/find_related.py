#!/usr/bin/env python3
from __future__ import annotations

"""
Build a single JSON mapping for all events:
- event metadata (category, start/end)
- lexicon (include + exclude regex patterns)
- matched subreddits from master list

Matching rule (formal & reproducible):
A subreddit is event-related iff:
  (matches at least one include-pattern) AND (matches no exclude-pattern)

Usage:
    python scripts/find_related.py  --subs-file data/all_subreddits.txt --out data/event_related_subreddits_lexicon.json


Optional:
  --min-token-len 4         ignore tiny tokens from event title
  --case-sensitive          treat regex patterns as case-sensitive (default: insensitive)
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple


# -----------------------------
# 1) Your event list (authoritative)
# -----------------------------
EVENTS: List[Tuple[str, datetime, datetime, str]] = [
    # Political/Governance
    ("Annexation of Crimea", datetime(2014, 2, 22), datetime(2014, 3, 18), "Political/Governance"),
    ("Donald Trump Inauguration", datetime(2017, 1, 20), datetime(2017, 1, 20), "Political/Governance"),
    ("US Presidential Campaign", datetime(2015, 6, 16), datetime(2016, 11, 8), "Political/Governance"),
    ("Brexit Campaign & Referendum", datetime(2016, 2, 20), datetime(2016, 6, 23), "Political/Governance"),

    # Health/Public Health
    ("Ebola Outbreak", datetime(2014, 3, 1), datetime(2016, 1, 14), "Health/Public Health"),

    # Disasters
    ("MH370 Disappearance", datetime(2014, 3, 8), datetime(2014, 3, 8), "Disasters"),
    ("Cyclone Pam (Vanuatu)", datetime(2015, 3, 13), datetime(2015, 3, 14), "Disasters"),
    ("Nepal Earthquake", datetime(2015, 4, 25), datetime(2015, 4, 25), "Disasters"),
    ("Fort McMurray Wildfire", datetime(2016, 5, 1), datetime(2016, 5, 31), "Disasters"),

    # Diplomacy
    ("Iran Nuclear Deal (JCPOA) Signed", datetime(2015, 7, 14), datetime(2015, 7, 14), "Diplomacy"),
    ("Obama Visits Cuba", datetime(2016, 3, 20), datetime(2016, 3, 20), "Diplomacy"),
    ("Colombia–FARC Peace Accord", datetime(2016, 9, 26), datetime(2016, 9, 26), "Diplomacy"),

    # Conflict/Security
    ("El Chapo Arrest", datetime(2014, 2, 22), datetime(2014, 2, 22), "Conflict/Security"),
    ("Turkish Coup Attempt", datetime(2016, 7, 15), datetime(2016, 7, 15), "Conflict/Security"),
    ("Rise of ISIS / Global Terrorism", datetime(2014, 6, 1), datetime(2017, 4, 1), "Conflict/Security"),

    # Climate / Environment
    ("UN Climate Summit (NYC)", datetime(2014, 9, 23), datetime(2014, 9, 23), "Climate/Environment"),
    ("COP21 Climate Conference", datetime(2015, 11, 30), datetime(2015, 12, 12), "Climate/Environment"),
    ("COP22 Marrakech", datetime(2016, 11, 7), datetime(2016, 11, 18), "Climate/Environment"),
]


# -----------------------------
# 2) Manual include/exclude lexicons (high precision)
#    You can extend these lists as you inspect outputs.
# -----------------------------
MANUAL_LEXICONS: Dict[str, Dict[str, List[str]]] = {
    "Annexation of Crimea": {
        "include": [
            "crimea", "sevastopol", "donbas", "maidan",
            "ukraine", "ukrainian", "ukraina",
            "russia", "russian", "russiatoday",
            "russianukrainianwar", "russiawarinukraine",
        ],
        "exclude": [
            "borussia", "brazil", "prussia", "gta", "russianroulette",
        ],
    },
    "Donald Trump Inauguration": {
        "include": ["donaldtrump", "trump", "inauguration", "maga", "whitehouse", "potus"],
        "exclude": [
            "trumpet", "amagami", "bigpharmagame", "magazine", "umagain", "himymagain", "komagatamaru",
            "kravmaga", "magajuana", "michaelwhitehouse", "magazin",
        ],
    },
    "US Presidential Campaign": {
        "include": [
            "uselection", "election", "elections", "president", "presidential",
            "democrats", "republican", "gop",
            "hillary", "clinton", "bernie", "sanders", "trump", "politics",
            "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware",
            "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
            "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri",
            "montana", "nebraska", "nevada", "newhampshire", "newjersey", "newmexico", "newyork",
            "northcarolina", "northdakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhodeisland",
            "southcarolina", "southdakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
            "wisconsin", "wyoming"
        ],
        "exclude": [
            "uk", "asian", "australianpolitics", "pokemon", "ski", "beer", "fishing", "gamers", "golf",
        ],
    },
    "Brexit Campaign & Referendum": {
        "include": ["brexit", "ukpolitics", "unitedkingdom", "referendum", "europeanunion", "eu", "eureferendum"],
        "exclude": ["badukpolitics", "badbadukpolitics"],  
    },
    "Ebola Outbreak": {
        "include": ["ebola", "ebolahoax", "ebolapanic", "ebolawestafrica", "publichealth"],
        "exclude": ["opiate", "computer", "antivirus", "computervirus", "computerviruses", "virusnamed", "clanvirus", "zika_virus"],
    },
    "MH370 Disappearance": {"include": ["mh370", "malaysiaairlines", "findflightmh370", "aircrash", "aviation"], "exclude": []},

    "Cyclone Pam (Vanuatu)": {
        "include": ["cyclonepam", "cyclone", "pam", "vanuatu"],
        "exclude": ["stormfront", "stormlight", "heroesofthestorm", "stormchasing", "stormwater", "sandstorm", "halestorm"],
    },
    "Nepal Earthquake": {
        "include": ["nepal", "nepalearthquake", "kathmandu", "earthquake", "earthquakes"],
        "exclude": ["quake3", "quakecon", "quakelive", "quakeworld", "quakegear", "quakechampions", "quakers", "planetquake"],
    },
    "Fort McMurray Wildfire": {
        "include": ["fortmcmurray", "mcmurray", "ymm", "woodbuffalo", "alberta", "wildfire", "fireseason"],
        "exclude": ["fireside", "firesidegatherings", "forgedinfire", "canadawhisky"], 
    },

    "Iran Nuclear Deal (JCPOA) Signed": {
        "include": ["iran", "iranian", "jcpoa", "nuclear", "sanctions", "iaea"],
        "exclude": ["philadelphiaeagles"],
    },
    "Obama Visits Cuba": {"include": ["obama", "cuba", "havana"], "exclude": ["scuba", "succubae"]},
    "Colombia–FARC Peace Accord": {
        "include": ["colombia", "farc"],
        "exclude": ["farcry", "farcraft", "skiesofarcadia", "accordingtoreddit", "accordion"],
    },
    "El Chapo Arrest": {"include": ["elchapo", "chapo", "sinaloa", "cartel", "drugwar"], "exclude": ["chapolin"]},
    "Turkish Coup Attempt": {
        "include": ["turkey", "turkish", "erdogan", "coup", "coupdetat"],
        "exclude": ["coupon", "coupons", "couponing", "couples", "couple", "coupe"],
    },
    "Rise of ISIS / Global Terrorism": {
        "include": ["isis", "islamicstate", "daesh", "terrorism", "jihad"],
        "exclude": [
            "thisis", "im14andthisis", "im12andthisis", "im16andthisis", "im18andthisis", "im30andthisis",
            "crisis", 
        ],
    },

    "UN Climate Summit (NYC)": {"include": ["climate", "climatechange", "climate_change"], "exclude": ["summitgaming", "leesummit"]},
    "COP21 Climate Conference": {"include": ["cop21", "parisclimate", "climate"], "exclude": ["parish", "sizecomparison", "comparison"]},
    "COP22 Marrakech": {"include": ["cop22", "marrakech", "climate"], "exclude": []},
}

# -----------------------------
# 3) Automatic fallback lexicon from event title
# -----------------------------
TOKEN_RE = re.compile(r"[a-z0-9]+")


DEFAULT_TITLE_STOPWORDS = {
    "of", "the", "and", "or", "signed", "visit", "visits", "global", "event",
    "campaign", "referendum", "conference", "summit", "agreement", "attempt",
    "rise", "un", "nyc", "phase", "out",
    # keep "cop21"/"cop22" out of stopwords because they are meaningful
}


def normalize(s: str) -> str:
    return s.strip().lower()


def load_master_subs(path: Path) -> List[str]:
    subs: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = normalize(line)
        if s:
            subs.append(s)
    # de-duplicate while preserving order
    seen: Set[str] = set()
    out: List[str] = []
    for s in subs:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def title_tokens(event_name: str, min_token_len: int) -> List[str]:
    toks = TOKEN_RE.findall(normalize(event_name))
    return [t for t in toks if len(t) >= min_token_len and t not in DEFAULT_TITLE_STOPWORDS]


def terms_to_patterns(terms: List[str], min_term_len: int = 4) -> List[str]:
    """
    Convert terms into safe regex patterns.
    By default we use substring matching (escaped), which works for concatenated subreddit names.
    """
    patterns: List[str] = []
    seen: Set[str] = set()
    for term in terms:
        term_n = normalize(term)
        if len(term_n) < min_term_len:
            continue
        if term_n in seen:
            continue
        seen.add(term_n)
        patterns.append(re.escape(term_n))
    return patterns


def make_event_lexicon(event_name: str, min_token_len: int) -> Tuple[List[str], List[str]]:
    """
    Build (include_patterns, exclude_patterns) for an event.
    Priority:
      1) manual include/exclude lists for precision
      2) fallback include patterns from title tokens if manual include missing/empty
    """
    manual = MANUAL_LEXICONS.get(event_name, {})
    include_terms = manual.get("include", [])
    exclude_terms_or_patterns = manual.get("exclude", [])

    include_patterns = terms_to_patterns(include_terms, min_term_len=4)

    # If include is empty, fall back to title tokens (keeps script robust)
    if not include_patterns:
        include_patterns = terms_to_patterns(title_tokens(event_name, min_token_len), min_term_len=4)

    # Exclude list can be either raw regex patterns (advanced users) or plain terms.
    exclude_patterns: List[str] = []
    for x in exclude_terms_or_patterns:
        x_n = normalize(x)
        if not x_n:
            continue
        # If user already provided regex metacharacters, accept as-is;
        # otherwise treat as plain term.
        if re.search(r"[\\.^$|?*+()[\]{}]", x_n):
            exclude_patterns.append(x)  # keep original
        else:
            exclude_patterns.append(re.escape(x_n))

    return include_patterns, exclude_patterns


def compile_patterns(patterns: List[str], case_sensitive: bool) -> List[re.Pattern]:
    flags = 0 if case_sensitive else re.IGNORECASE
    return [re.compile(p, flags=flags) for p in patterns]


def matches_any(s: str, pats: List[re.Pattern]) -> bool:
    return any(p.search(s) for p in pats)


def iso_date(d: datetime) -> str:
    return d.strftime("%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-token-len", type=int, default=4)
    ap.add_argument("--case-sensitive", action="store_true")
    args = ap.parse_args()

    master = load_master_subs(args.subs_file)

    out: Dict[str, Dict] = {}
    for name, start, end, category in EVENTS:
        include_patterns, exclude_patterns = make_event_lexicon(name, args.min_token_len)

        include_compiled = compile_patterns(include_patterns, args.case_sensitive)
        exclude_compiled = compile_patterns(exclude_patterns, args.case_sensitive) if exclude_patterns else []

        matched: List[str] = []
        for sub in master:
            if not matches_any(sub, include_compiled):
                continue
            if exclude_compiled and matches_any(sub, exclude_compiled):
                continue
            matched.append(sub)

        out[name] = {
            "category": category,
            "start_date": iso_date(start),
            "end_date": iso_date(end),
            "lexicon": {
                "include_patterns": include_patterns,
                "exclude_patterns": exclude_patterns,
            },
            "matched_subreddits": matched,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
