"""
Step 0b: Pull real, ID-verified bird photos from iNaturalist -- no API key needed.

Uses "research grade" observations only (community-verified IDs), which is
better ground truth than scraping random images off the web.

Usage:
    python download_inaturalist.py --species_file species.txt \
        --out_dir data/images_raw --place_id 6681 --max_per_species 150

species.txt: one scientific name per line (same file used for the audio downloader), e.g.
    Corvus splendens
    Merops orientalis
    Pycnonotus cafer

--place_id restricts to a region via iNaturalist's place IDs (6681 = India).
Look up other places at https://www.inaturalist.org/places -- the ID is in the URL.
Omit --place_id to search worldwide.
"""

import argparse
import time
from pathlib import Path

import requests

API_URL = "https://api.inaturalist.org/v1/observations"
TAXA_URL = "https://api.inaturalist.org/v1/taxa"


def resolve_taxon_id(species):
    """
    Look up the iNaturalist taxon_id for a scientific name.

    Querying observations by taxon_name directly is unreliable -- it only
    matches some species and silently returns zero for others (their
    matching logic doesn't reliably handle every valid scientific name
    string). The documented, robust approach is to resolve the name to a
    numeric taxon_id first via the taxa search endpoint, then query
    observations by that ID instead.
    """
    resp = requests.get(TAXA_URL, params={"q": species, "rank": "species", "per_page": 5}, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    for taxon in results:
        # prefer an exact scientific-name match over the first fuzzy result
        if taxon.get("name", "").lower() == species.lower():
            return taxon["id"], taxon.get("name")
    if results:
        # no exact match -- fall back to the top fuzzy result, but flag it
        return results[0]["id"], results[0].get("name")
    return None, None


def fetch_species_photos(taxon_id, place_id=None, max_results=150, per_page=30):
    photos = []
    page = 1
    while len(photos) < max_results:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": per_page,
            "page": page,
        }
        if place_id:
            params["place_id"] = place_id

        resp = requests.get(API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for obs in results:
            for photo in obs.get("photos", []):
                url = photo.get("url", "")
                if url:
                    # thumbnail URLs look like .../square.jpg -- swap for a larger size
                    large_url = url.replace("square", "large")
                    photos.append((obs["id"], photo["id"], large_url))

        if len(results) < per_page:
            break
        page += 1
        time.sleep(1)  # be polite to the API

    return photos[:max_results]


def download_photo(url, out_path):
    if out_path.exists():
        return
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species_file", required=True)
    ap.add_argument("--out_dir", default="data/images_raw")
    ap.add_argument("--place_id", type=int, default=None, help="6681 = India")
    ap.add_argument("--max_per_species", type=int, default=150)
    ap.add_argument("--sleep", type=float, default=0.5)
    args = ap.parse_args()

    species_list = [s.strip() for s in Path(args.species_file).read_text().splitlines() if s.strip()]
    print(f"{len(species_list)} species to fetch")

    for species in species_list:
        species_dir = Path(args.out_dir) / species.replace(" ", "_")
        species_dir.mkdir(parents=True, exist_ok=True)

        try:
            taxon_id, matched_name = resolve_taxon_id(species)
        except requests.RequestException as e:
            print(f"  [{species}] taxon lookup failed: {e}")
            continue

        if taxon_id is None:
            print(f"  [{species}] no matching taxon found on iNaturalist -- skipping")
            continue
        if matched_name.lower() != species.lower():
            print(f"  [{species}] no exact match -- using closest match '{matched_name}' (taxon_id={taxon_id})")

        try:
            photos = fetch_species_photos(taxon_id, args.place_id, args.max_per_species)
        except requests.RequestException as e:
            print(f"  [{species}] query failed: {e}")
            continue

        print(f"[{species}] {len(photos)} photos found")
        downloaded = 0
        for obs_id, photo_id, url in photos:
            out_path = species_dir / f"{obs_id}_{photo_id}.jpg"
            try:
                download_photo(url, out_path)
                downloaded += 1
            except requests.RequestException as e:
                print(f"    failed to download {url}: {e}")
            time.sleep(args.sleep)
        print(f"  -> saved {downloaded} images to {species_dir}")


if __name__ == "__main__":
    main()