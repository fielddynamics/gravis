"""
Bulk-import all SPARC galaxies from data/sparc_sources/ into sparc/.

Reads every JSON file in data/sparc_sources/, validates it against both
the schema validator (validate_galaxy) and the quality validator
(validate_galaxy_quality), then writes valid galaxies to sparc/<id>.json
using an atomic write.

Usage:
    python data/import_all_sparc.py
    python data/import_all_sparc.py --dry-run
    python data/import_all_sparc.py --force --verbose

Options:
    --dry-run   Validate only; do not write files.
    --force     Overwrite galaxies that already exist in sparc/.
    --verbose   Print per-galaxy detail (warnings, errors).

IMPORTANT: No unicode in code or messages (Windows charmap).
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import tempfile

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from physics.services.sparc.sparc_parser import (
    validate_galaxy,
    validate_galaxy_quality,
)

SPARC_SOURCES_DIR = os.path.join(_REPO_ROOT, "data", "sparc_sources")
SPARC_OUTPUT_DIR = os.path.join(_REPO_ROOT, "sparc")


def _atomic_write(data, out_dir):
    """Write galaxy dict to out_dir/<id>.json atomically."""
    gid = data["id"]
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        final_path = os.path.join(out_dir, gid + ".json")
        os.replace(tmp_path, final_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def build_catalog(sparc_dir):
    """Scan sparc_dir for galaxy JSONs and write catalog.json with {id, name} per galaxy."""
    entries = []
    for fname in sorted(os.listdir(sparc_dir)):
        if not fname.endswith(".json") or fname == "catalog.json":
            continue
        path = os.path.join(sparc_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("id") and data.get("name"):
                entries.append({"id": data["id"], "name": data["name"]})
        except (json.JSONDecodeError, OSError):
            continue
    entries.sort(key=lambda e: e["id"].lower())
    catalog_path = os.path.join(sparc_dir, "catalog.json")
    fd, tmp_path = tempfile.mkstemp(dir=sparc_dir, prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp_path, catalog_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return entries


def _collect_source_files(src_dir):
    """Return sorted list of (galaxy_id, filepath) tuples from src_dir."""
    if not os.path.isdir(src_dir):
        return []
    results = []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".json"):
            continue
        gid = name[:-5]
        results.append((gid, os.path.join(src_dir, name)))
    return results


def main():
    ap = argparse.ArgumentParser(
        description="Bulk-import all SPARC galaxies into sparc/."
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Validate only, do not write files",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files in sparc/",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print per-galaxy detail",
    )
    args = ap.parse_args()

    sources = _collect_source_files(SPARC_SOURCES_DIR)
    if not sources:
        print("No source files found in {}".format(SPARC_SOURCES_DIR))
        return 1

    existing = set()
    if os.path.isdir(SPARC_OUTPUT_DIR):
        for name in os.listdir(SPARC_OUTPUT_DIR):
            if name.endswith(".json"):
                existing.add(name[:-5])

    total = len(sources)
    imported = 0
    skipped_exists = 0
    skipped_schema = 0
    skipped_quality = 0
    warned = 0
    schema_errors = []
    quality_errors = []
    quality_warnings = []

    for gid, path in sources:
        # Load raw JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            schema_errors.append((gid, "could not read/parse: {}".format(e)))
            skipped_schema += 1
            continue

        # Schema validation
        try:
            validate_galaxy(data)
        except ValueError as e:
            schema_errors.append((gid, str(e)))
            skipped_schema += 1
            continue

        # Quality validation
        qr = validate_galaxy_quality(data)

        if qr["errors"]:
            quality_errors.append((gid, qr["errors"]))
            skipped_quality += 1
            if args.verbose:
                print("FAIL  {}: {}".format(gid, "; ".join(qr["errors"])))
            continue

        if qr["warnings"]:
            quality_warnings.append((gid, qr["warnings"]))
            warned += 1
            if args.verbose:
                print("WARN  {}: {}".format(gid, "; ".join(qr["warnings"])))

        # Skip if already exists and not forcing
        if gid in existing and not args.force:
            skipped_exists += 1
            if args.verbose:
                print("SKIP  {}: already imported".format(gid))
            continue

        # Write (unless dry run)
        if args.dry_run:
            if args.verbose:
                print("OK    {} (dry-run, not written)".format(gid))
            imported += 1
            continue

        try:
            _atomic_write(data, SPARC_OUTPUT_DIR)
            imported += 1
            if args.verbose:
                print("OK    {}".format(gid))
        except Exception as e:
            schema_errors.append((gid, "write failed: {}".format(e)))
            skipped_schema += 1

    # -- Summary report -----------------------------------------------------
    print("")
    print("=" * 60)
    print("SPARC Bulk Import Summary")
    print("=" * 60)
    print("Source files found:      {}".format(total))
    print("Successfully imported:   {}".format(imported))
    print("Skipped (already exist): {}".format(skipped_exists))
    print("Skipped (schema error):  {}".format(skipped_schema))
    print("Skipped (quality error): {}".format(skipped_quality))
    print("Imported with warnings:  {}".format(warned))

    if args.dry_run:
        print("")
        print("(dry-run mode: no files were written)")

    if schema_errors:
        print("")
        print("Schema errors:")
        for gid, msg in schema_errors[:30]:
            print("  {}: {}".format(gid, msg))
        if len(schema_errors) > 30:
            print("  ... and {} more".format(len(schema_errors) - 30))

    if quality_errors:
        print("")
        print("Quality errors:")
        for gid, errs in quality_errors[:30]:
            print("  {}: {}".format(gid, "; ".join(errs)))
        if len(quality_errors) > 30:
            print("  ... and {} more".format(len(quality_errors) - 30))

    if quality_warnings and args.verbose:
        print("")
        print("Quality warnings:")
        for gid, warns in quality_warnings[:50]:
            print("  {}: {}".format(gid, "; ".join(warns)))
        if len(quality_warnings) > 50:
            print("  ... and {} more".format(len(quality_warnings) - 50))

    # -- Build catalog index --------------------------------------------------
    if not args.dry_run and os.path.isdir(SPARC_OUTPUT_DIR):
        catalog = build_catalog(SPARC_OUTPUT_DIR)
        print("Wrote sparc/catalog.json ({} entries)".format(len(catalog)))
    elif args.dry_run:
        print("(dry-run mode: catalog.json not written)")

    print("")
    ok = imported + skipped_exists
    print("Total galaxies in sparc/: ~{}".format(ok + len(existing) if not args.force else imported))
    return 0


if __name__ == "__main__":
    sys.exit(main())
