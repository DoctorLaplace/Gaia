"""
GAIA: Download EnMAP L2A tiles covering the Southern Cape region.

Uses the DLR EOC Geoservice STAC API to search EnMAP L2A products and
optionally download GeoTIFF assets.

Usage:
    python src/download_enmap.py
    python src/download_enmap.py --limit 3
    python src/download_enmap.py --download --limit 1
    python src/download_enmap.py --download --start-date 2024-01-01 --end-date 2024-12-31
    python src/download_enmap.py --download --user YOUR_USER --password YOUR_PASS

Login:
    For downloads you need a DLR EOC/EnMAP account with EnMAP access enabled.
    Credentials can be passed via:
      1. --user / --password flags
      2. ENMAP_USERNAME / ENMAP_PASSWORD environment variables
      3. ENMAP_USER / ENMAP_PASS environment variables, for compatibility
      4. A .env file in the project root or home directory

Recommended .env:
    ENMAP_USERNAME=your_username
    ENMAP_PASSWORD=your_password

Security:
    Avoid passing --password on shared systems because command-line arguments
    can be visible in shell history or process lists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from tqdm import tqdm

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── ANSI colors ───────────────────────────────────────────────────────────────
GREEN = "\033[32m"
PURPLE = "\033[35m"
CYAN = "\033[36m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"

# ── Southern Cape Bounding Box ────────────────────────────────────────────────
# Coordinates are STAC bbox order: west, south, east, north.
BBOX_WEST = 18.0
BBOX_SOUTH = -35.5
BBOX_EAST = 25.0
BBOX_NORTH = -31.5

# ── DLR EOC Geoservice STAC API ───────────────────────────────────────────────
STAC_ROOT = "https://geoservice.dlr.de/eoc/ogc/stac/v1"
STAC_SEARCH = f"{STAC_ROOT}/search"
COLLECTION_ID = "ENMAP_HSI_L2A"  # EnMAP L2A surface reflectance / CEOS-ARD

# ── EnMAP band info, kept for downstream compatibility if needed ──────────────
INVALID_BAND_IDXS = [
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
    139, 140, 160, 161, 162, 163, 164, 165, 166,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search and download EnMAP L2A GeoTIFF tiles for the Southern Cape."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download files. Without this, only runs discovery/dry-run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of STAC items after filtering. Useful for testing.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=500,
        help="Maximum number of STAC items to retrieve before local filtering.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Parallel download workers. Only used with --download.",
    )
    parser.add_argument(
        "--out-dir",
        "--out_dir",
        dest="out_dir",
        type=str,
        default=None,
        help="Output directory.",
    )
    parser.add_argument("--user", type=str, default=None, help="DLR EOC/EnMAP username.")
    parser.add_argument("--password", type=str, default=None, help="DLR EOC/EnMAP password.")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2022-04-27",
        help="Start date in YYYY-MM-DD format. Default is EnMAP L2A archive start.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format. Default searches from start date to latest.",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=30.0,
        help="Maximum cloud cover percentage. Items with unknown cloud are kept.",
    )
    parser.add_argument(
        "--max-quality",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help=(
            "Maximum enmap:overallQuality to keep if present. "
            "0=nominal, 1=reduced, 2=low. Default keeps all."
        ),
    )
    parser.add_argument(
        "--asset-keys",
        type=str,
        default="image",
        help=(
            "Comma-separated STAC asset keys to download. Default: image. "
            "Use 'auto' to download all GeoTIFF-like assets found."
        ),
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Save catalog JSON and exit, even if --download is provided.",
    )
    return parser.parse_args()


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_credentials(args: argparse.Namespace) -> tuple[str | None, str | None]:
    """Load DLR EOC/EnMAP credentials from args, env vars, or .env files."""
    env_values: dict[str, str] = {}
    for env_path in (PROJECT_ROOT / ".env", Path.home() / ".env"):
        env_values.update(read_env_file(env_path))

    user = (
        args.user
        or os.environ.get("ENMAP_USERNAME")
        or os.environ.get("ENMAP_USER")
        or env_values.get("ENMAP_USERNAME")
        or env_values.get("ENMAP_USER")
    )
    password = (
        args.password
        or os.environ.get("ENMAP_PASSWORD")
        or os.environ.get("ENMAP_PASS")
        or env_values.get("ENMAP_PASSWORD")
        or env_values.get("ENMAP_PASS")
    )
    return user, password


def validate_date(date_str: str | None, label: str) -> None:
    if not date_str:
        return
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(f"{RED}[ERROR] {label} must be YYYY-MM-DD, got: {date_str}{RESET}") from exc


def stac_datetime_range(start_date: str | None, end_date: str | None) -> str | None:
    validate_date(start_date, "--start-date")
    validate_date(end_date, "--end-date")

    if start_date and end_date:
        return f"{start_date}T00:00:00Z/{end_date}T23:59:59Z"
    if start_date:
        return f"{start_date}T00:00:00Z/.."
    if end_date:
        return f"../{end_date}T23:59:59Z"
    return None


def short_response_text(resp: requests.Response, max_chars: int = 500) -> str:
    try:
        return resp.text[:max_chars].replace("\n", " ")
    except Exception:
        return ""


def list_enmap_collections() -> None:
    """Print EnMAP-like collections to help debug API/collection issues."""
    print(f"\n{CYAN}[*] Listing EnMAP-like STAC collections...{RESET}")
    try:
        resp = requests.get(f"{STAC_ROOT}/collections", timeout=30)
        resp.raise_for_status()
        collections = resp.json().get("collections", [])
        enmap_collections = [
            c for c in collections
            if "enmap" in c.get("id", "").lower()
            or "enmap" in c.get("title", "").lower()
        ]
        if not enmap_collections:
            print(f"{YELLOW}    No EnMAP-like collections found at this endpoint.{RESET}")
            return
        for collection in enmap_collections:
            print(f"    - {collection.get('id')}: {collection.get('title', 'N/A')}")
    except Exception as exc:
        print(f"{RED}    Failed to list collections: {exc}{RESET}")


def next_stac_page(session: requests.Session, data: dict[str, Any]) -> requests.Response | None:
    """Follow a STAC next link if present."""
    for link in data.get("links", []):
        if link.get("rel") != "next":
            continue

        href = link.get("href") or STAC_SEARCH
        method = str(link.get("method", "GET")).upper()
        body = link.get("body")
        headers = link.get("headers") or {}

        if method == "POST" or body:
            return session.post(href, json=body, headers=headers, timeout=60)
        return session.get(href, headers=headers, timeout=60)

    return None


def search_stac(
    bbox: tuple[float, float, float, float],
    max_items: int,
    start_date: str | None,
    end_date: str | None,
    collection: str = COLLECTION_ID,
) -> list[dict[str, Any]]:
    """Search the DLR EOC STAC API for EnMAP L2A items in the bounding box."""
    datetime_filter = stac_datetime_range(start_date, end_date)

    print(f"{CYAN}[*] Searching DLR EOC STAC API for EnMAP L2A tiles...{RESET}")
    print(f"{CYAN}    Endpoint:   {STAC_SEARCH}{RESET}")
    print(f"{CYAN}    Collection: {collection}{RESET}")
    print(f"{CYAN}    BBox:       [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]{RESET}")
    if datetime_filter:
        print(f"{CYAN}    Datetime:   {datetime_filter}{RESET}")

    payload: dict[str, Any] = {
        "collections": [collection],
        "bbox": list(bbox),
        "limit": min(max_items, 100),
    }
    if datetime_filter:
        payload["datetime"] = datetime_filter

    items: list[dict[str, Any]] = []
    session = requests.Session()

    try:
        resp = session.post(STAC_SEARCH, json=payload, timeout=60)
        if resp.status_code != 200:
            print(f"{RED}[ERROR] STAC search failed: HTTP {resp.status_code}{RESET}")
            print(f"{YELLOW}        {short_response_text(resp)}{RESET}")
            list_enmap_collections()
            return []

        data = resp.json()
        page_count = 1

        while True:
            features = data.get("features", [])
            items.extend(features)

            if len(items) >= max_items:
                break

            resp = next_stac_page(session, data)
            if resp is None:
                break
            if resp.status_code != 200:
                print(f"{YELLOW}[WARN] Pagination stopped: HTTP {resp.status_code}{RESET}")
                break

            data = resp.json()
            page_count += 1
            if page_count > 200:
                print(f"{YELLOW}[WARN] Pagination stopped after 200 pages to avoid an infinite loop.{RESET}")
                break

    except requests.RequestException as exc:
        print(f"{RED}[ERROR] STAC request failed: {exc}{RESET}")
        list_enmap_collections()
        return []
    except json.JSONDecodeError as exc:
        print(f"{RED}[ERROR] STAC returned non-JSON response: {exc}{RESET}")
        return []

    return items[:max_items]


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def item_datetime(item: dict[str, Any]) -> str:
    props = item.get("properties", {})
    date_value = (
        props.get("datetime")
        or props.get("start_datetime")
        or props.get("end_datetime")
        or "?"
    )
    return str(date_value)[:10]


def item_cloud_cover(item: dict[str, Any]) -> float | None:
    props = item.get("properties", {})
    for key in ("eo:cloud_cover", "cloud_cover", "cloudCover", "enmap:cloudCover"):
        value = to_float(props.get(key))
        if value is not None:
            return value
    return None


def item_quality(item: dict[str, Any]) -> int | None:
    props = item.get("properties", {})
    for key in ("enmap:overallQuality", "overallQuality", "quality"):
        value = to_int(props.get(key))
        if value is not None:
            return value
    return None


def filter_items(
    items: list[dict[str, Any]],
    max_cloud: float,
    max_quality: int,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []

    for item in items:
        cloud = item_cloud_cover(item)
        quality = item_quality(item)

        cloud_ok = cloud is None or cloud <= max_cloud
        quality_ok = quality is None or quality <= max_quality

        if cloud_ok and quality_ok:
            filtered.append(item)

    return filtered


def is_geotiff_asset(key: str, asset: dict[str, Any]) -> bool:
    href = str(asset.get("href", ""))
    media_type = str(asset.get("type", "")).lower()
    roles = [str(role).lower() for role in asset.get("roles", [])]
    key_lower = key.lower()
    href_path = urlparse(href).path.lower()

    has_tif_extension = href_path.endswith((".tif", ".tiff"))
    has_tif_media_type = "tiff" in media_type or "geotiff" in media_type or "cog" in media_type
    looks_like_image_key = any(
        token in key_lower
        for token in ("image", "data", "spectral", "reflectance", "enmap", "hsi")
    )
    has_data_role = "data" in roles

    return has_tif_extension or has_tif_media_type or has_data_role or looks_like_image_key


def get_download_assets(item: dict[str, Any], requested_keys: list[str]) -> dict[str, str]:
    """Extract downloadable GeoTIFF-like asset URLs from a STAC item."""
    assets = item.get("assets", {}) or {}
    if not isinstance(assets, dict):
        return {}

    requested_lower = [key.strip().lower() for key in requested_keys if key.strip()]
    auto_mode = not requested_lower or "auto" in requested_lower

    selected: dict[str, str] = {}

    if not auto_mode:
        # First: exact key match, because DLR/GFZ examples commonly use asset key "image".
        for key, asset in assets.items():
            if key.lower() in requested_lower and asset.get("href"):
                selected[key] = asset["href"]

        if selected:
            return selected

    # Fallback / auto: find GeoTIFF-like data assets.
    for key, asset in assets.items():
        href = asset.get("href")
        if href and is_geotiff_asset(key, asset):
            selected[key] = href

    return selected


def safe_filename_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "unknown"


def extension_from_url(url: str, default: str = ".tif") -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix in {".tif", ".tiff", ".json", ".xml", ".txt"}:
        return suffix
    return default


def make_output_path(out_dir: Path, item_id: str, asset_key: str, url: str) -> Path:
    safe_item = safe_filename_part(item_id)
    safe_key = safe_filename_part(asset_key)
    ext = extension_from_url(url, default=".tif")
    return out_dir / f"{safe_item}_{safe_key}{ext}"


def download_tile(url: str, output_path: Path, auth: tuple[str, str] | None = None) -> str:
    """Download a single tile. Returns ok, skip, or an error string."""
    if output_path.exists() and output_path.stat().st_size > 0:
        return "skip"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with requests.get(url, stream=True, timeout=(10, 180), auth=auth) as resp:
            if resp.status_code in (401, 403):
                if auth:
                    return (
                        f"error: HTTP {resp.status_code}. Credentials were sent, but access was denied. "
                        "Check username/password and EnMAP Access Service permission."
                    )
                return f"error: HTTP {resp.status_code}. Login credentials are probably required."

            resp.raise_for_status()

            with tmp_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        if tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            return "error: downloaded file is empty"

        tmp_path.replace(output_path)
        return "ok"

    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        return f"error: {exc}"


def print_manual_download_hint(out_dir: Path) -> None:
    print(f"\n{YELLOW}{'=' * 60}")
    print("  NO TILES FOUND VIA STAC API")
    print(f"{'=' * 60}{RESET}")
    print("\n  Try these checks:")
    print("  1. Confirm your bbox is correct for the Southern Cape.")
    print("  2. Widen the date range, for example --start-date 2022-04-27.")
    print("  3. Raise --max-cloud or --max-quality if filtering is too strict.")
    print("  4. Check the EOC Geoservice EnMAP L2A catalog manually.")
    print("  5. If downloads fail with 401/403, enable EnMAP Access Service for your account.")
    print(f"\n  Expected output folder: {out_dir}")
    print("\n  Then run:")
    print(f"  python src/infer_heatmap.py --enmap_dir {out_dir}\n")


def print_dry_run(items: list[dict[str, Any]], out_dir: Path, requested_asset_keys: list[str]) -> None:
    print(f"\n{BOLD}{'─' * 60}")
    print("  DRY RUN — Tiles that would be downloaded:")
    print(f"{'─' * 60}{RESET}")

    for i, item in enumerate(items[:20], start=1):
        item_id = item.get("id", "?")
        date = item_datetime(item)
        cloud = item_cloud_cover(item)
        quality = item_quality(item)
        assets = get_download_assets(item, requested_asset_keys)
        cloud_label = "?" if cloud is None else f"{cloud:.1f}%"
        quality_label = "?" if quality is None else str(quality)
        print(
            f"  [{i:3d}] {item_id} | {date} | "
            f"cloud: {cloud_label} | quality: {quality_label} | assets: {len(assets)}"
        )
        if assets:
            print(f"        asset keys: {', '.join(sorted(assets.keys()))}")

    if len(items) > 20:
        print(f"  ... and {len(items) - 20} more")

    print(f"\n  {BOLD}To download, re-run with --download{RESET}")
    print(f"  Catalog saved to: {out_dir / 'stac_catalog.json'}\n")


def save_catalog(items: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = out_dir / "stac_catalog.json"
    catalog_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return catalog_path


def build_download_jobs(
    items: list[dict[str, Any]],
    out_dir: Path,
    requested_asset_keys: list[str],
) -> list[tuple[str, str, str, Path]]:
    jobs: list[tuple[str, str, str, Path]] = []

    for i, item in enumerate(items):
        item_id = str(item.get("id") or f"tile_{i}")
        assets = get_download_assets(item, requested_asset_keys)

        if not assets:
            print(f"{YELLOW}[SKIP] {item_id}: no matching downloadable assets{RESET}")
            continue

        for key, url in assets.items():
            output_path = make_output_path(out_dir, item_id, key, url)
            jobs.append((item_id, key, url, output_path))

    return jobs


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "data" / "enmap" / "southern_cape"
    requested_asset_keys = [key.strip() for key in args.asset_keys.split(",") if key.strip()]

    print(f"\n{BOLD}{PURPLE}{'=' * 60}")
    print("  GAIA — EnMAP L2A Download (Southern Cape)")
    print(f"{'=' * 60}{RESET}\n")
    print(f"  Bounding Box: ({BBOX_WEST}°E, {BBOX_SOUTH}°S) → ({BBOX_EAST}°E, {BBOX_NORTH}°S)")
    print(f"  Date Range:   {args.start_date} → {args.end_date or 'latest'}")
    print(f"  Max Cloud:    {args.max_cloud}%")
    print(f"  Max Quality:  {args.max_quality}  (0 nominal, 1 reduced, 2 low)")
    print(f"  Asset Keys:   {', '.join(requested_asset_keys) if requested_asset_keys else 'auto'}")
    print(f"  Output:       {out_dir}")
    print()

    bbox = (BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH)
    items = search_stac(
        bbox=bbox,
        max_items=args.max_items,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if not items:
        print_manual_download_hint(out_dir)
        return

    filtered_items = filter_items(items, max_cloud=args.max_cloud, max_quality=args.max_quality)

    print(
        f"\n{GREEN}[✔] Found {len(items)} total STAC items, "
        f"{len(filtered_items)} after cloud/quality filtering{RESET}"
    )

    if args.limit is not None:
        filtered_items = filtered_items[: args.limit]
        print(f"{CYAN}[*] Limited to {len(filtered_items)} item(s){RESET}")

    if not filtered_items:
        print(f"{YELLOW}[!] Nothing left after filters. Try raising --max-cloud or --max-quality.{RESET}")
        save_catalog([], out_dir)
        return

    catalog_path = save_catalog(filtered_items, out_dir)
    print(f"{GREEN}[✔] Catalog saved to: {catalog_path}{RESET}")

    if not args.download or args.catalog_only:
        print_dry_run(filtered_items, out_dir, requested_asset_keys)
        return

    user, password = load_credentials(args)
    auth = (user, password) if user and password else None

    if not auth:
        print(f"{YELLOW}[!] No credentials found. Downloads may fail if auth is required.{RESET}")
        print("    Use --user/--password, ENMAP_USERNAME/ENMAP_PASSWORD, or a .env file.\n")

    jobs = build_download_jobs(filtered_items, out_dir, requested_asset_keys)
    if not jobs:
        print(f"{YELLOW}[!] No downloadable asset URLs found in the filtered STAC items.{RESET}")
        return

    print(f"\n{CYAN}[*] Downloading {len(jobs)} asset(s) with {args.workers} worker(s)...{RESET}")

    success = 0
    skipped = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_tile, url, output_path, auth): (item_id, key, output_path)
            for item_id, key, url, output_path in jobs
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            item_id, key, output_path = futures[future]
            result = future.result()

            if result == "ok":
                tqdm.write(f"{GREEN}[OK] {output_path.name}{RESET}")
                success += 1
            elif result == "skip":
                tqdm.write(f"{YELLOW}[SKIP] {output_path.name} already exists{RESET}")
                skipped += 1
            else:
                tqdm.write(f"{RED}[FAIL] {item_id}/{key}: {result}{RESET}")
                failed += 1

    print(f"\n{BOLD}{GREEN}{'=' * 60}")
    print("  DOWNLOAD COMPLETE")
    print(f"  Success: {success} | Skipped: {skipped} | Failed: {failed}")
    print(f"  Files in: {out_dir}")
    print(f"{'=' * 60}{RESET}\n")

    print(f"  {BOLD}Next: Generate heatmap with:{RESET}")
    print(f"  python src/infer_heatmap.py --enmap_dir {out_dir}\n")


if __name__ == "__main__":
    main()
