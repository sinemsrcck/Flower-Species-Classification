"""Out-of-distribution (OOD) flower species sourcing for generalization testing.

Downloads a handful of CC-licensed sample images (via the Wikimedia Commons
search API, no API key required) for flower species that do NOT appear among
the 102 classes in data/cat_to_name.json, so we can test how CLIP behaves on
species it was never asked to classify.
"""
import json
import re
from pathlib import Path

import requests

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "FlowerSpeciesClassificationProject/1.0 (SENG481 educational project)"

# Candidate species picked to NOT share a family/name with any of the 102
# known classes (checked programmatically in select_ood_species below).
CANDIDATE_SPECIES = [
    "tulip", "chrysanthemum", "zinnia", "gardenia", "peony",
    "hydrangea", "lavender", "begonia", "jasmine", "freesia", "cosmos flower",
]

NUM_OOD_SPECIES = 8
IMAGES_PER_SPECIES = 4


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower())


def select_ood_species(cat_to_name: dict, candidates=None, n=NUM_OOD_SPECIES) -> list:
    """Pick flower species whose names don't overlap with any of the 102 known classes."""
    candidates = candidates or CANDIDATE_SPECIES
    known_names = [_normalize(name) for name in cat_to_name.values()]
    selected = []
    for species in candidates:
        key = _normalize(species.replace(" flower", ""))
        overlaps = any(key in known or known in key for known in known_names if known)
        if not overlaps:
            selected.append(species)
        if len(selected) >= n:
            break
    return selected


def search_commons_images(query: str, limit: int = 8):
    """Search Wikimedia Commons for CC/public-domain images matching query."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 800,
    }
    r = requests.get(COMMONS_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    pages = r.json().get("query", {}).get("pages", {})
    results = []
    for page in pages.values():
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        url = info.get("thumburl") or info.get("url")
        meta = info.get("extmetadata", {})
        license_name = meta.get("LicenseShortName", {}).get("value", "")
        artist_raw = meta.get("Artist", {}).get("value", "unknown")
        artist = re.sub("<[^<]+?>", "", artist_raw).strip() or "unknown"
        if url and ("cc" in license_name.lower() or "public domain" in license_name.lower()):
            results.append({
                "url": url,
                "title": page.get("title", ""),
                "license": license_name,
                "artist": artist,
            })
    return results


def download_ood_samples(cat_to_name: dict, out_dir="data/ood_samples",
                          n_species=NUM_OOD_SPECIES, per_species=IMAGES_PER_SPECIES):
    """Download sample images for OOD species into out_dir/<species>/, with attribution.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    species_list = select_ood_species(cat_to_name, n=n_species)

    attribution = {}
    for species in species_list:
        species_dir = out_dir / species.replace(" ", "_")
        species_dir.mkdir(exist_ok=True)
        candidates = search_commons_images(f"{species} flower", limit=per_species * 3)
        saved = []
        for item in candidates:
            if len(saved) >= per_species:
                break
            try:
                resp = requests.get(item["url"], headers={"User-Agent": USER_AGENT}, timeout=20)
                resp.raise_for_status()
            except requests.RequestException:
                continue
            ext = ".png" if ".png" in item["url"].lower() else ".jpg"
            fname = f"{len(saved)}{ext}"
            (species_dir / fname).write_bytes(resp.content)
            saved.append({"file": fname, **item})
        attribution[species] = saved
        print(f"{species}: saved {len(saved)} image(s)")

    with open(out_dir / "attribution.json", "w", encoding="utf-8") as f:
        json.dump(attribution, f, indent=2, ensure_ascii=False)
    return species_list, attribution


if __name__ == "__main__":
    with open("data/cat_to_name.json") as f:
        _cat_to_name = json.load(f)
    download_ood_samples(_cat_to_name)
