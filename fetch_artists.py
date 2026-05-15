"""
Fetch artist data from Last.fm API.

Collects top tags, top tracks, listener/play counts, and similar artists
for the 10 target artists. Saves raw data to data/artists.json.
"""

import requests
import json
import time
import os
from pathlib import Path

API_KEY = os.environ.get("LASTFM_API_KEY", "")
BASE_URL = "https://ws.audioscrobbler.com/2.0/"

ARTISTS = [
    "Taylor Swift",
    "Drake",
    "Billie Eilish",
    "The Weeknd",
    "Kendrick Lamar",
    "Olivia Rodrigo",
    "21 Savage",
    "Lorde",
    "Post Malone",
    "J. Cole",
]


def lastfm_get(method: str, params: dict) -> dict:
    params = {**params, "method": method, "api_key": API_KEY, "format": "json"}
    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise ValueError(f"Last.fm error {data['error']}: {data.get('message', '')}")
    return data


def fetch_artist_info(artist_name: str) -> dict:
    raw = lastfm_get("artist.getInfo", {"artist": artist_name})
    info = raw.get("artist", {})
    stats = info.get("stats", {})
    return {
        "name": info.get("name", artist_name),
        "listeners": int(stats.get("listeners", 0)),
        "playcount": int(stats.get("playcount", 0)),
        "url": info.get("url", ""),
    }


def fetch_top_tags(artist_name: str, limit: int = 15) -> list[dict]:
    raw = lastfm_get("artist.getTopTags", {"artist": artist_name})
    tags = raw.get("toptags", {}).get("tag", [])[:limit]
    return [{"name": t["name"].lower().strip(), "count": int(t["count"])} for t in tags]


def fetch_top_tracks(artist_name: str, limit: int = 10) -> list[dict]:
    raw = lastfm_get("artist.getTopTracks", {"artist": artist_name, "limit": limit})
    tracks = raw.get("toptracks", {}).get("track", [])
    return [
        {
            "name": t["name"],
            "playcount": int(t.get("playcount", 0)),
            "listeners": int(t.get("listeners", 0)),
            "rank": int(t.get("@attr", {}).get("rank", idx + 1)),
        }
        for idx, t in enumerate(tracks)
    ]


def fetch_similar_artists(artist_name: str, limit: int = 10) -> list[dict]:
    raw = lastfm_get("artist.getSimilar", {"artist": artist_name, "limit": limit})
    similar = raw.get("similarartists", {}).get("artist", [])
    return [{"name": s["name"], "match": float(s["match"])} for s in similar]


def fetch_artist_data(artist_name: str) -> dict:
    print(f"  Fetching info...")
    info = fetch_artist_info(artist_name)
    time.sleep(0.2)

    print(f"  Fetching tags...")
    tags = fetch_top_tags(artist_name)
    time.sleep(0.2)

    print(f"  Fetching top tracks...")
    tracks = fetch_top_tracks(artist_name)
    time.sleep(0.2)

    print(f"  Fetching similar artists...")
    similar = fetch_similar_artists(artist_name)
    time.sleep(0.2)

    return {
        **info,
        "tags": tags,
        "top_tracks": tracks,
        "similar_artists": similar,
    }


def main():
    Path("data").mkdir(exist_ok=True)
    artists_data = {}

    for artist in ARTISTS:
        print(f"\nFetching {artist}...")
        try:
            artists_data[artist] = fetch_artist_data(artist)
            listeners = artists_data[artist]["listeners"]
            playcount = artists_data[artist]["playcount"]
            tag_names = [t["name"] for t in artists_data[artist]["tags"][:5]]
            print(f"  -> {listeners:,} listeners | {playcount:,} plays | tags: {tag_names}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            artists_data[artist] = {"name": artist, "error": str(exc)}

    out_path = "data/artists.json"
    with open(out_path, "w") as f:
        json.dump(artists_data, f, indent=2)

    success = sum(1 for v in artists_data.values() if "error" not in v)
    print(f"\nSaved {success}/{len(ARTISTS)} artists to {out_path}")


if __name__ == "__main__":
    main()
