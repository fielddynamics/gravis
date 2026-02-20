"""
Build data/sparc_sources/*.json for all 175 SPARC galaxies from VizieR table1 + table2.

Use: place table1.dat and table2.dat in data/ (or pass --zip with a zip that
contains them), then run:
  python data/build_sparc_sources.py

If a zip path is given, the script unzips to a temp dir and looks for table1.dat
and table2.dat inside. Otherwise uses data/table1.dat and data/table2.dat (or
sparc_table1.dat for table1 if table2 is from VizieR as table2.dat).

Each galaxy is built from table1 (distance, RHI, L3.6, Rdisk, MHI, Ref) and
table2 (Rad, Vobs, e_Vobs, Vgas, Vdisk, Vbulge, SBdisk, SBbulge). We require
photometric data: at least one table2 row with valid Vobs and (Vdisk != 0 or
Vbulge != 0 or SBdisk > 0 or SBbulge > 0). Every output JSON is validated
with the app parser before write. ASCII only.
"""

from __future__ import print_function

import argparse
import json
import os
import re
import sys
import tempfile
import zipfile

TABLE2_URL = "https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/157/table2.dat"

# Add repo root for imports
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from physics.services.sparc.sparc_parser import validate_galaxy


def _normalize_id(raw_name):
    s = (raw_name or "").strip()
    if not s:
        return ""
    lower = s.lower().replace(" ", "")
    m = re.match(r"(ngc|ugc|ddo|ic|pgc)(0+)(\d+)$", lower)
    if m:
        return m.group(1) + m.group(3)
    m = re.match(r"(ugca)(0*)(\d+)$", lower)
    if m:
        return m.group(1) + (m.group(3) or "")
    lower = lower.replace("-", "_")
    return lower


def _normalize_name(raw_name):
    s = (raw_name or "").strip()
    if not s:
        return ""
    m = re.match(r"(NGC|UGC|DDO|IC|PGC)(0+)(\d+)$", s, re.IGNORECASE)
    if m:
        prefix = m.group(1).upper() if m.group(1).upper() in ("NGC", "UGC", "DDO", "IC", "PGC") else m.group(1)
        num = m.group(3).lstrip("0") or "0"
        return "{} {}".format(prefix, num)
    m = re.match(r"(UGCA)(0*)(\d+)$", s, re.IGNORECASE)
    if m:
        return "{} {}".format("UGCA", m.group(3).lstrip("0") or "0")
    if re.match(r"ESO\d", s, re.IGNORECASE):
        return s[:3] + " " + s[3:]
    if s.startswith("CamB"):
        return "Cam B"
    return s


# VizieR table1.dat: Name 1-11, then numeric columns
def _parse_table1_line(line):
    if len(line) < 30:
        return None
    name = line[0:11].strip()
    rest = line[11:].strip().split()
    if len(rest) < 14:
        return None
    dist = _float(rest[1])   # Dist
    l36 = _float(rest[6])    # L3.6
    rdisk = _float(rest[10]) # Rdisk
    mhi = _float(rest[12])   # MHI
    rhi = _float(rest[13])   # RHI
    ref = rest[17] if len(rest) > 17 else ""
    if name and dist is not None and dist > 0:
        return {
            "name": name,
            "id": _normalize_id(name),
            "distance": dist,
            "rdisk": rdisk if rdisk is not None and rdisk >= 0 else 1.0,
            "mhi": mhi if mhi is not None and mhi >= 0 else 0.0,
            "rhi": rhi if rhi is not None and rhi >= 0 else 1.0,
            "l36": l36 if l36 is not None and l36 >= 0 else 0.0,
            "ref": ref,
        }
    return None


# VizieR table2.dat: Name 1-11, then Dist, Rad, Vobs, e_Vobs, Vgas, Vdisk, Vbulge, SBdisk, SBbulge
def _parse_table2_line(line):
    if len(line) < 30:
        return None
    name = line[0:11].strip()
    rest = line[11:].strip().split()
    if len(rest) < 3:
        return None
    rad = _float(rest[1])
    vobs = _float(rest[2])
    e_vobs = _float(rest[3]) if len(rest) > 3 else None
    vgas = _float(rest[4]) if len(rest) > 4 else None
    vdisk = _float(rest[5]) if len(rest) > 5 else None
    vbulge = _float(rest[6]) if len(rest) > 6 else None
    sbdisk = _float(rest[7]) if len(rest) > 7 else None
    sbbulge = _float(rest[8]) if len(rest) > 8 else None
    if name and rad is not None and vobs is not None:
        return {
            "name": name,
            "rad": rad,
            "vobs": vobs,
            "e_vobs": e_vobs if e_vobs is not None else 0.0,
            "vgas": vgas if vgas is not None else 0.0,
            "vdisk": vdisk if vdisk is not None else 0.0,
            "vbulge": vbulge if vbulge is not None else 0.0,
            "sbdisk": sbdisk if sbdisk is not None else 0.0,
            "sbbulge": sbbulge if sbbulge is not None else 0.0,
        }
    return None


def _float(s):
    try:
        return float(s.strip())
    except (TypeError, ValueError):
        return None


def _load_table1(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            row = _parse_table1_line(line)
            if row and row["id"]:
                out[row["id"]] = row
    return out


def _load_table2(path):
    by_id = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            row = _parse_table2_line(line)
            if not row:
                continue
            gid = _normalize_id(row["name"])
            if not gid:
                continue
            if gid not in by_id:
                by_id[gid] = []
            by_id[gid].append(row)
    return by_id


def _has_photometric(rows):
    """Require at least one row with Vobs and (Vdisk or Vbulge or SBdisk or SBbulge)."""
    for r in rows:
        vobs = r.get("vobs")
        if vobs is None or vobs <= 0:
            continue
        if (r.get("vdisk") or 0) != 0 or (r.get("vbulge") or 0) != 0:
            return True
        if (r.get("sbdisk") or 0) > 0 or (r.get("sbbulge") or 0) > 0:
            return True
    return False


def _build_galaxy_json(t1, t2_rows):
    """Build one galaxy dict matching app schema. Raises ValueError if invalid."""
    if not t2_rows:
        raise ValueError("No table2 rows for galaxy")
    if not _has_photometric(t2_rows):
        raise ValueError("No photometric data (need Vdisk/Vbulge/SBdisk/SBbulge)")
    dist = t1["distance"]
    rhi = t1["rhi"]
    rdisk = max(t1["rdisk"], 0.1)
    l36 = max(t1["l36"], 0.0)
    mhi = max(t1["mhi"], 0.0)
    # Stellar mass ~ 0.5 M/L * L3.6 in 10^9 Lsun -> 0.5 * L3.6 * 1e9 Msun
    m_disk = 0.5 * l36 * 1e9
    m_gas = mhi * 1.33 * 1e9
    max_vbulge = max((r.get("vbulge") or 0) for r in t2_rows)
    if max_vbulge <= 0:
        m_bulge = 0
    else:
        max_vdisk = max((r.get("vdisk") or 0) for r in t2_rows) or 1.0
        m_bulge = int(0.15 * m_disk * (max_vbulge / max_vdisk) ** 2)
        if m_bulge <= 0:
            m_bulge = 0
    mass_bary = (0.5 * l36 + mhi * 1.33)
    observations = []
    for r in t2_rows:
        observations.append({
            "r": round(r["rad"], 3),
            "v": round(r["vobs"], 2),
            "err": round(r["e_vobs"], 2) if r.get("e_vobs") else 0.0,
        })
    if not observations:
        raise ValueError("No valid observations")
    max_obs_r = max(o["r"] for o in observations)
    galactic_radius = round(rhi, 2) if rhi and rhi > 0 else round(max_obs_r * 1.2, 2)
    if galactic_radius <= 0:
        galactic_radius = round(max_obs_r, 2) if max_obs_r > 0 else 1.0
    disk_rd = round(rdisk, 2) if rdisk and rdisk > 0 else 0.5
    gas_rd = round(rhi / 3.0, 2) if rhi and rhi > 0 else round(max_obs_r / 3.0, 2)
    if gas_rd <= 0:
        gas_rd = 0.5
    ref_str = t1.get("ref") or ""
    references = ["SPARC VizieR J/AJ/152/157 (Lelli+2016)"]
    if ref_str:
        references.append("Ref: " + ref_str)
    data = {
        "id": t1["id"],
        "name": _normalize_name(t1["name"]),
        "distance": round(dist, 2),
        "galactic_radius": galactic_radius,
        "mass": round(mass_bary, 4),
        "accel": 1,
        "mass_model": {
            "bulge": {"M": int(m_bulge), "a": 0.3},
            "disk": {"M": int(m_disk), "Rd": disk_rd},
            "gas": {"M": int(m_gas), "Rd": gas_rd},
        },
        "observations": observations,
        "references": references,
    }
    validate_galaxy(data)
    return data


def main():
    ap = argparse.ArgumentParser(description="Build sparc_sources JSON from SPARC table1+table2.")
    ap.add_argument("--zip", default=None, help="Path to zip containing table1.dat and table2.dat")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: data/sparc_sources)")
    args = ap.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = script_dir
    out_dir = args.out_dir or os.path.join(script_dir, "sparc_sources")
    table1_path = os.path.join(data_dir, "sparc_table1.dat")
    table2_path = os.path.join(data_dir, "table2.dat")

    if args.zip:
        if not os.path.isfile(args.zip):
            print("Error: zip file not found:", args.zip)
            sys.exit(1)
        tmpdir = tempfile.mkdtemp(prefix="sparc_build_")
        try:
            with zipfile.ZipFile(args.zip, "r") as z:
                names = z.namelist()
                t1_name = None
                t2_name = None
                for n in names:
                    base = os.path.basename(n).lower()
                    if base == "table1.dat":
                        t1_name = n
                    elif base == "table2.dat":
                        t2_name = n
                if not t1_name or not t2_name:
                    print("Error: zip must contain table1.dat and table2.dat")
                    sys.exit(1)
                z.extract(t1_name, tmpdir)
                z.extract(t2_name, tmpdir)
            table1_path = os.path.join(tmpdir, t1_name)
            table2_path = os.path.join(tmpdir, t2_name)
            if not os.path.isfile(table1_path):
                table1_path = os.path.join(tmpdir, os.path.basename(t1_name))
            if not os.path.isfile(table2_path):
                table2_path = os.path.join(tmpdir, os.path.basename(t2_name))
        except Exception as e:
            print("Error unzipping:", e)
            sys.exit(1)
    else:
        if not os.path.isfile(table1_path):
            table1_path = os.path.join(data_dir, "table1.dat")
        if not os.path.isfile(table1_path):
            print("Error: table1 not found. Place sparc_table1.dat or table1.dat in data/, or use --zip")
            sys.exit(1)
        if not os.path.isfile(table2_path):
            try:
                import urllib.request
                print("Downloading table2.dat from VizieR...")
                req = urllib.request.Request(TABLE2_URL)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(table2_path, "wb") as f:
                        f.write(resp.read())
                print("Saved to", table2_path)
            except Exception as e:
                print("Error: table2.dat not found and download failed:", e)
                print("Download from", TABLE2_URL, "and save as data/table2.dat, or use --zip")
                sys.exit(1)

    t1_by_id = _load_table1(table1_path)
    t2_by_id = _load_table2(table2_path)
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    errors = []
    for gid, t1 in sorted(t1_by_id.items()):
        t2_rows = t2_by_id.get(gid, [])
        try:
            data = _build_galaxy_json(t1, t2_rows)
            out_path = os.path.join(out_dir, gid + ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            written += 1
        except ValueError as e:
            errors.append((gid, str(e)))
        except Exception as e:
            errors.append((gid, str(e)))

    if errors:
        print("Skipped or failed ({}):".format(len(errors)))
        for gid, msg in errors[:20]:
            print("  {}: {}".format(gid, msg))
        if len(errors) > 20:
            print("  ... and {} more".format(len(errors) - 20))
    print("Wrote {} galaxy JSONs to {}".format(written, out_dir))
    if written < 175:
        print("Expected 175; missing table2 rows or photometric check for some galaxies.")
    return 0 if written >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
