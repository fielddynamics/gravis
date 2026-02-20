"""
One-off script: build data/sparc_catalog.json from SPARC table1.dat (175 galaxies).
Output: JSON array of {id, name}. Only id is authoritative; name is best-effort for
display only (do not rely on it for import or logic). Reads table from local
sparc_table1.dat or VizieR FTP. ASCII only, no unicode.
"""
import json
import os
import re
import urllib.request

TABLE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/157/table1.dat"
OUT_PATH = "sparc_catalog.json"


def normalize_id(raw_name):
    """Convert table name (first 11 chars) to id: lowercase, strip leading zeros in numbers."""
    s = raw_name.strip()
    if not s:
        return ""
    lower = s.lower().replace(" ", "")
    # Strip leading zeros from numeric part for NGC, UGC, DDO, IC, PGC
    m = re.match(r"(ngc|ugc|ddo|ic|pgc)(0+)(\d+)$", lower)
    if m:
        return m.group(1) + m.group(3)
    m = re.match(r"(ugca)(0*)(\d+)$", lower)
    if m:
        return m.group(1) + (m.group(3) or "")
    # ESO079-G014 -> eso079_g014, F563-V1 -> f563_v1
    lower = lower.replace("-", "_")
    return lower


def normalize_name(raw_name):
    """Display name: add space after prefix, strip leading zeros in number for NGC/UGC etc."""
    s = raw_name.strip()
    if not s:
        return ""
    # NGC0024 -> NGC 24, UGC00128 -> UGC 128
    m = re.match(r"(NGC|UGC|DDO|IC|PGC)(0+)(\d+)$", s, re.IGNORECASE)
    if m:
        prefix = m.group(1).upper() if m.group(1).upper() in ("NGC", "UGC", "DDO", "IC", "PGC") else m.group(1)
        num = m.group(3).lstrip("0") or "0"
        return "{} {}".format(prefix, num)
    m = re.match(r"(UGCA)(0*)(\d+)$", s, re.IGNORECASE)
    if m:
        return "{} {}".format("UGCA", m.group(3).lstrip("0") or "0")
    # ESO079-G014 -> ESO 079-G014, F563-V1 -> F563-V1, CamB -> Cam B
    if re.match(r"ESO\d", s, re.IGNORECASE):
        return s[:3] + " " + s[3:]
    if s.startswith("CamB"):
        return "Cam B"
    return s


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "sparc_table1.dat")
    if os.path.isfile(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        req = urllib.request.Request(TABLE_URL)
        with urllib.request.urlopen(req, timeout=15) as f:
            text = f.read().decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    out = []
    seen_ids = set()
    for ln in lines:
        raw_name = ln[:11].strip()
        if not raw_name:
            continue
        gid = normalize_id(raw_name)
        name = normalize_name(raw_name)
        if not gid or gid in seen_ids:
            continue
        seen_ids.add(gid)
        out.append({"id": gid, "name": name})
    out.sort(key=lambda x: (x["name"].upper(), x["id"]))
    out_path_abs = os.path.join(script_dir, OUT_PATH)
    with open(out_path_abs, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote {} galaxies to {}".format(len(out), out_path_abs))


if __name__ == "__main__":
    main()
