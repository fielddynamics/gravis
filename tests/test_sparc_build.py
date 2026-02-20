"""
Gate tests: verify SPARC table1+table2 were correctly processed to JSON for selected galaxies.

Run after build_sparc_sources.py. Ensures we do not assume the build worked;
we assert schema validity and consistency with source table data for three galaxies.
ASCII only.
"""

import json
import os
import sys

import pytest

# Repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
SPARC_SOURCES_DIR = os.path.join(DATA_DIR, "sparc_sources")
TABLE1_PATH = os.path.join(DATA_DIR, "sparc_table1.dat")
TABLE2_PATH = os.path.join(DATA_DIR, "table2.dat")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from physics.services.sparc.sparc_parser import validate_galaxy


# Three galaxies we validate end-to-end (table -> JSON)
GATE_GALAXY_IDS = ["d631_7", "ngc3198", "ddo154"]


def _normalize_id(raw_name):
    s = (raw_name or "").strip()
    if not s:
        return ""
    lower = s.lower().replace(" ", "")
    if lower.startswith("ngc") and lower[3:].isdigit():
        return "ngc" + str(int(lower[3:]))
    if lower.startswith("ugc") and lower[3:].isdigit():
        return "ugc" + str(int(lower[3:]))
    if lower.startswith("ddo") and lower[3:].replace("0", "").isdigit():
        return "ddo" + lower[3:].lstrip("0") or "0"
    lower = lower.replace("-", "_")
    return lower


def _load_table2_rows_by_id():
    """Load table2 and group rows by galaxy id (first 11 chars -> normalize_id)."""
    if not os.path.isfile(TABLE2_PATH):
        return {}
    by_id = {}
    with open(TABLE2_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if len(line) < 30:
                continue
            name = line[0:11].strip()
            rest = line[11:].strip().split()
            if len(rest) < 3:
                continue
            gid = _normalize_id(name)
            rad = _float(rest[1])
            vobs = _float(rest[2])
            if gid and rad is not None and vobs is not None:
                if gid not in by_id:
                    by_id[gid] = []
                by_id[gid].append({"rad": rad, "vobs": vobs})
    return by_id


def _float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


@pytest.fixture(scope="module")
def table2_by_id():
    return _load_table2_rows_by_id()


def test_sparc_sources_dir_exists():
    assert os.path.isdir(SPARC_SOURCES_DIR), "data/sparc_sources/ must exist (run build_sparc_sources.py)"


def test_gate_galaxy_json_files_exist():
    for gid in GATE_GALAXY_IDS:
        path = os.path.join(SPARC_SOURCES_DIR, gid + ".json")
        assert os.path.isfile(path), "Missing JSON for gate galaxy: %s" % gid


def test_gate_galaxy_json_schema_and_invariants(table2_by_id):
    for gid in GATE_GALAXY_IDS:
        path = os.path.join(SPARC_SOURCES_DIR, gid + ".json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        validate_galaxy(data)
        assert data.get("id") == gid
        assert data.get("distance") and data["distance"] > 0
        assert data.get("mass") and data["mass"] > 0
        assert data.get("galactic_radius") is not None and data["galactic_radius"] > 0
        mm = data.get("mass_model") or {}
        for comp in ("bulge", "disk", "gas"):
            assert comp in mm and isinstance(mm[comp], dict)
            if comp == "bulge":
                assert mm[comp].get("a") and float(mm[comp]["a"]) > 0
            else:
                assert mm[comp].get("Rd") is not None and float(mm[comp]["Rd"]) > 0
        obs = data.get("observations") or []
        assert len(obs) >= 1
        for i, pt in enumerate(obs):
            assert "r" in pt and "v" in pt
            assert float(pt["r"]) >= 0 and float(pt["v"]) >= 0


def test_gate_galaxy_observations_match_table2(table2_by_id):
    """Assert each gate galaxy JSON observations match table2 (count, first and last r,v)."""
    for gid in GATE_GALAXY_IDS:
        path = os.path.join(SPARC_SOURCES_DIR, gid + ".json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obs = data.get("observations") or []
        rows = table2_by_id.get(gid, [])
        assert len(rows) >= 1, "Table2 has no rows for %s" % gid
        assert len(obs) == len(rows), "Observation count mismatch for %s: JSON %d vs table2 %d" % (
            gid, len(obs), len(rows))
        # First row
        assert abs(obs[0]["r"] - rows[0]["rad"]) < 0.01
        assert abs(obs[0]["v"] - rows[0]["vobs"]) < 0.02
        # Last row
        assert abs(obs[-1]["r"] - rows[-1]["rad"]) < 0.01
        assert abs(obs[-1]["v"] - rows[-1]["vobs"]) < 0.02


def test_gate_d631_7_no_bulge():
    """D631-7 has Vbulge=0 in table2; JSON must have bulge M=0."""
    path = os.path.join(SPARC_SOURCES_DIR, "d631_7.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["mass_model"]["bulge"]["M"] == 0, "D631-7 must have bulge M=0 (no-bulge galaxy)"


def test_gate_ngc3198_distance_and_radius():
    """NGC3198: distance and galactic_radius from table1 (sanity)."""
    path = os.path.join(SPARC_SOURCES_DIR, "ngc3198.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert 10 <= data["distance"] <= 20
    assert data["galactic_radius"] >= 10
    assert len(data["observations"]) >= 10


def test_gate_ddo154_scale_lengths_positive():
    """DDO154: all scale lengths must be positive (backend requirement)."""
    path = os.path.join(SPARC_SOURCES_DIR, "ddo154.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mm = data["mass_model"]
    assert mm["bulge"]["a"] > 0
    assert mm["disk"]["Rd"] > 0
    assert mm["gas"]["Rd"] > 0
