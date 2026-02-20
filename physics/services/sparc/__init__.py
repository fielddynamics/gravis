"""
SPARC import service: catalog of importable galaxies and import endpoint.

GET /api/sparc/catalog  - list galaxies available for import + already_imported
POST /api/sparc/import - import by galaxy_id (from bundled source) or file upload

Catalog authority: Only the galaxy id from sparc_catalog.json is reliable. Names and
other fields in the catalog are best-effort and must not be used for import or logic.
Import is always by galaxy_id; data is loaded from data/sparc_sources/<id>.json.

IMPORTANT: No unicode in code or messages (Windows charmap).
"""

import json
import logging
import os
import re
import tempfile

from flask import request, jsonify

from physics.services import GravisService
from data.galaxies import _sparc_dir, _load_sparc_galaxies
from physics.services.sparc.sparc_parser import (
    validate_galaxy,
    parse_galaxy_json,
    load_and_validate_galaxy_file,
)

log = logging.getLogger(__name__)

# Sanitize galaxy_id: only alphanumeric, underscore, hyphen
GALAXY_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

# Paths relative to repo root (this file: physics/services/sparc/__init__.py)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CATALOG_PATH = os.path.join(_REPO_ROOT, "data", "sparc_catalog.json")
SPARC_SOURCES_DIR = os.path.join(_REPO_ROOT, "data", "sparc_sources")


class SparcService(GravisService):

    id = "sparc"
    name = "SPARC Import"
    description = "Import galaxy observations from SPARC catalog or file"
    category = "galactic"
    status = "live"
    route = "/analysis"

    def validate(self, config):
        """Stub: no computation config for this service."""
        return config if isinstance(config, dict) else {}

    def compute(self, config):
        """Stub: no computation; import is handled in routes."""
        raise NotImplementedError("SPARC service has no compute endpoint")

    def _get_catalog(self):
        """Load catalog JSON; return list of {id} only. Only id is authoritative; names are not relied upon. On error return []."""
        if not os.path.isfile(CATALOG_PATH):
            return []
        try:
            with open(CATALOG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(raw, list):
            return []
        out = []
        seen = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            gid = item.get("id")
            if not gid or not isinstance(gid, str) or gid in seen:
                continue
            seen.add(gid)
            out.append({"id": gid.strip()})
        out.sort(key=lambda x: x["id"].lower())
        return out

    def _importable_ids(self):
        """Return set of galaxy ids for which we have data in sparc_sources/. Only these can be imported by id."""
        out = set()
        if not os.path.isdir(SPARC_SOURCES_DIR):
            return out
        for name in os.listdir(SPARC_SOURCES_DIR):
            if name.endswith(".json") and GALAXY_ID_PATTERN.match(name[:-5]):
                out.add(name[:-5])
        return out

    def _already_imported_ids(self):
        """Return list of galaxy ids that exist in sparc/ (user-imported only). We do not exclude built-in examples, so the full SPARC catalog is shown; only duplicates from prior imports are hidden."""
        galaxies = _load_sparc_galaxies()
        return [g["id"] for g in galaxies if isinstance(g, dict) and g.get("id")]

    def _import_by_id(self, galaxy_id):
        """
        Import by galaxy id only. Load from data/sparc_sources/<id>.json, validate, write to sparc/.
        Returns (galaxy_dict, None) on success; (None, error_dict) on failure.
        """
        if not GALAXY_ID_PATTERN.match(galaxy_id):
            return None, {"error": "Import failed: invalid galaxy identifier.", "detail": None}
        path = os.path.join(SPARC_SOURCES_DIR, galaxy_id + ".json")
        if not os.path.isfile(path):
            return None, {"error": "No SPARC data file for this galaxy.", "error_code": "not_found", "detail": "No data file for this id."}
        try:
            data = load_and_validate_galaxy_file(path)
        except ValueError as e:
            return None, {"error": str(e), "error_code": "parse_error", "detail": None}
        if data.get("id") != galaxy_id:
            return None, {"error": "Import failed: invalid galaxy identifier.", "error_code": "invalid_id", "detail": None}
        return data, None

    def _import_from_bytes(self, raw):
        """
        Parse raw bytes as JSON, validate, return (galaxy_dict, None) or (None, error_dict).
        """
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, {"error": "Import failed: file encoding must be UTF-8.", "detail": None}
        try:
            data = parse_galaxy_json(text)
        except ValueError as e:
            return None, {"error": str(e), "error_code": "parse_error", "detail": None}
        gid = data.get("id")
        if not gid or not GALAXY_ID_PATTERN.match(gid):
            return None, {"error": "Import failed: invalid galaxy identifier.", "error_code": "invalid_id", "detail": None}
        return data, None

    def _write_galaxy(self, data):
        """
        Write validated galaxy dict to sparc/<id>.json (atomic).
        Returns None on success; raises OSError on failure.
        """
        gid = data["id"]
        sparc_dir = _sparc_dir()
        os.makedirs(sparc_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=sparc_dir, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            final_path = os.path.join(sparc_dir, gid + ".json")
            os.replace(tmp_path, final_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def register_routes(self, bp):
        """Mount SPARC catalog and import endpoints."""

        @bp.route("/sparc/catalog", methods=["GET"])
        def sparc_catalog():
            try:
                galaxies = self._get_catalog()
                already_imported = self._already_imported_ids()
                importable_ids = list(self._importable_ids())
            except Exception as e:
                log.error("SPARC catalog load failed: %s", e)
                return jsonify({"error": "Could not load catalog"}), 500
            return jsonify({
                "galaxies": galaxies,
                "already_imported": already_imported,
                "importable_ids": importable_ids,
            })

        @bp.route("/sparc/import", methods=["POST"])
        def sparc_import():
            content_type = request.content_type or ""
            # File upload
            if "multipart/form-data" in content_type:
                if "file" not in request.files:
                    return jsonify({"error": "Import failed: no file provided.", "detail": None}), 400
                f = request.files["file"]
                if not f.filename:
                    return jsonify({"error": "Import failed: no file provided.", "detail": None}), 400
                try:
                    raw = f.read()
                except OSError:
                    return jsonify({"error": "Import failed: could not read file.", "detail": None}), 400
                data, err = self._import_from_bytes(raw)
            else:
                # JSON body: { galaxy_id: "..." }
                body = request.get_json(silent=True)
                if not body or not isinstance(body, dict):
                    return jsonify({"error": "Import failed: request body must be JSON with galaxy_id or upload a file.", "detail": None}), 400
                galaxy_id = body.get("galaxy_id")
                if not galaxy_id or not isinstance(galaxy_id, str):
                    return jsonify({"error": "Import failed: galaxy_id is required.", "detail": None}), 400
                galaxy_id = galaxy_id.strip()
                data, err = self._import_by_id(galaxy_id)
            if err:
                status = 404 if err.get("error_code") == "not_found" else 400
                return jsonify(err), status
            try:
                self._write_galaxy(data)
            except OSError as e:
                log.warning("SPARC import write failed: galaxy_id=%s, err=%s", data.get("id"), e)
                return jsonify({
                    "error": "Import failed: could not save galaxy data. Try again or check permissions.",
                    "error_code": "save_failed",
                    "detail": str(e)
                }), 500
            log.info("SPARC import succeeded galaxy_id=%s name=%s", data.get("id"), data.get("name"))
            return jsonify({"galaxy_id": data["id"], "name": data.get("name") or data["id"]})
