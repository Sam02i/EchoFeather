"""
Step 2a: Pull Indian bird calls from the Xeno-Canto API.

As of Oct 2025, Xeno-Canto is on API v3 and requires a free API key
(the old v2 endpoint used here previously now returns 404):
    1. Register at https://xeno-canto.org and verify your email
    2. Get your key from https://xeno-canto.org/account
    3. Pass it as --api_key, or set env var XC_API_KEY

Usage:
    python download_xenocanto.py --species_file species.txt --out_dir data/audio/raw \
        --country India --max_per_species 40 --api_key YOUR_KEY

species.txt: one scientific name per line, e.g.
    Corvus splendens
    Merops orientalis
    Pycnonotus cafer
"""

import argparse
import os
import time
from pathlib import Path

import requests

API_URL = "https://xeno-canto.org/api/3/recordings"


def fetch_species_recordings(species, api_key, country=None, max_results=40):
    # gen:genus sp:species query syntax, optionally scoped to a country
    genus, sp = species.strip().split(" ", 1)
    query = f'gen:"{genus}" sp:"{sp}"'
    if country:
        query += f' cnt:"{country}"'

    recordings = []
    page = 1
    while len(recordings) < max_results:
        resp = requests.get(
            API_URL,
            params={"query": query, "page": page, "per_page": 100, "key": api_key},
            timeout=30,
        )
        if resp.status_code == 401:
            raise SystemExit(
                "401 Unauthorized from Xeno-Canto. Your --api_key is missing or invalid -- "
                "get one at https://xeno-canto.org/account"
            )
        resp.raise_for_status()
        data = resp.json()
        recordings.extend(data.get("recordings", []))
        if page >= int(data.get("numPages", 1)):
            break
        page += 1
    return recordings[:max_results]


def download_recording(rec, out_dir):
    # some recordings have no downloadable file (licensing restrictions) -- skip those
    url = rec.get("file")
    if not url:
        return None
    # file url sometimes comes back protocol-relative ("//...")
    if url.startswith("//"):
        url = "https:" + url
    filename = f"{rec['id']}.mp3"
    out_path = out_dir / filename
    if out_path.exists():
        return out_path

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species_file", required=True)
    ap.add_argument("--out_dir", default="data/audio/raw")
    ap.add_argument("--country", default="India")
    ap.add_argument("--max_per_species", type=int, default=40)
    ap.add_argument("--sleep", type=float, default=1.0, help="politeness delay between downloads")
    ap.add_argument("--api_key", default=os.environ.get("XC_API_KEY"),
                    help="required; get one at https://xeno-canto.org/account, "
                        "or set env var XC_API_KEY")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit(
            "No API key. Xeno-Canto v3 requires one -- register + verify email at "
            "https://xeno-canto.org then get your key at https://xeno-canto.org/account, "
            "then pass --api_key YOUR_KEY (or export XC_API_KEY=YOUR_KEY)."
        )

    species_list = [s.strip() for s in Path(args.species_file).read_text().splitlines() if s.strip()]
    print(f"{len(species_list)} species to fetch")

    for species in species_list:
        species_dir = Path(args.out_dir) / species.replace(" ", "_")
        species_dir.mkdir(parents=True, exist_ok=True)

        try:
            recordings = fetch_species_recordings(species, args.api_key, args.country, args.max_per_species)
        except requests.RequestException as e:
            print(f"  [{species}] query failed: {e}")
            continue

        print(f"[{species}] {len(recordings)} recordings found")
        downloaded, skipped = 0, 0
        for rec in recordings:
            try:
                result = download_recording(rec, species_dir)
                if result is None:
                    skipped += 1
                else:
                    downloaded += 1
            except requests.RequestException as e:
                print(f"    failed to download {rec.get('id')}: {e}")
            except Exception as e:
                print(f"    unexpected error on {rec.get('id')}: {e}")
            time.sleep(args.sleep)
        extra = f" ({skipped} had no downloadable file)" if skipped else ""
        print(f"  -> saved {downloaded} clips to {species_dir}{extra}")


if __name__ == "__main__":
    main()