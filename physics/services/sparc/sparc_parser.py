"""
SPARC galaxy data parser and validator.

Validates galaxy dicts against the app schema (id, name, distance,
galactic_radius, mass, accel, mass_model, observations, references).
Use for import by id (from bundled JSON) or file upload (JSON).

Also provides quality validation (validate_galaxy_quality) that checks
observation depth, photometric completeness, mass consistency, and
numeric sanity. Quality validation returns a result dict rather than
raising, so callers can collect warnings across many galaxies.

IMPORTANT: No unicode in code or error messages (Windows charmap).
"""

import json
import math
import re


REQUIRED_TOP_LEVEL = (
    "id", "name", "distance", "galactic_radius", "mass", "accel",
    "mass_model", "observations", "references"
)
REQUIRED_MASS_MODEL_KEYS = ("bulge", "disk", "gas")
REQUIRED_OBS_KEYS = ("r", "v")


def _check_mass_model(obj):
    """Return None if valid; else error string."""
    if not isinstance(obj, dict):
        return "mass_model must be a dict"
    for k in REQUIRED_MASS_MODEL_KEYS:
        if k not in obj or not isinstance(obj[k], dict):
            return "mass_model must have bulge, disk, gas as dicts"
    return None


def _check_observations(obj):
    """Return None if valid; else error string."""
    if not isinstance(obj, list):
        return "observations must be a list"
    if len(obj) == 0:
        return "observations must have at least one point"
    for i, pt in enumerate(obj):
        if not isinstance(pt, dict):
            return "observations[{}] must be a dict".format(i)
        if "r" not in pt or "v" not in pt:
            return "observations[{}] must have r and v".format(i)
        try:
            float(pt["r"])
            float(pt["v"])
        except (TypeError, ValueError):
            return "observations[{}] r and v must be numeric".format(i)
    return None


def validate_galaxy(data):
    """
    Validate a galaxy dict. Raises ValueError with a user-facing message
    if invalid. Returns the same dict if valid (no copy).
    """
    if not isinstance(data, dict):
        raise ValueError("Import failed: file format is not recognized.")
    for key in REQUIRED_TOP_LEVEL:
        if key not in data:
            raise ValueError("Import failed: missing required field '{}'.".format(key))
    err = _check_mass_model(data.get("mass_model"))
    if err:
        raise ValueError("Import failed: {}.".format(err))
    err = _check_observations(data.get("observations"))
    if err:
        raise ValueError("Import failed: {}.".format(err))
    try:
        d = float(data["distance"])
        m = float(data["mass"])
        if d <= 0:
            raise ValueError("Import failed: distance must be positive.")
        if m <= 0:
            raise ValueError("Import failed: mass must be positive.")
    except (TypeError, ValueError) as e:
        if isinstance(e, ValueError) and "Import failed" in str(e):
            raise
        raise ValueError("Import failed: distance and mass must be positive numbers.")
    return data


def parse_galaxy_json(text):
    """
    Parse JSON string into a galaxy dict and validate. Returns validated dict.
    Raises ValueError with user-facing message on parse or validation error.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError("Import failed: file format is not recognized.")
    return validate_galaxy(data)


def load_and_validate_galaxy_file(path):
    """
    Read file at path (UTF-8), parse as JSON, validate. Returns galaxy dict.
    Raises ValueError with user-facing message on error.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        raise ValueError("Import failed: could not read file.")
    return parse_galaxy_json(text)


# ---------------------------------------------------------------------------
# Quality validation (non-raising, returns structured result)
# ---------------------------------------------------------------------------

_GALAXY_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

MIN_OBSERVATION_POINTS = 3
MIN_RADIAL_SPAN_KPC = 2.0
MASS_CONSISTENCY_TOLERANCE = 0.5  # dex (log10 scale)


def _is_finite(val):
    """Return True if val is a finite number."""
    try:
        f = float(val)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def _safe_float(val, default=None):
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def validate_galaxy_quality(data):
    """
    Run quality checks on a galaxy dict that already passed validate_galaxy().

    Returns a dict with:
        valid    (bool)  : True if no fatal errors
        errors   (list)  : fatal problems that should block import
        warnings (list)  : non-fatal issues worth reporting
    """
    errors = []
    warnings = []

    # -- ID and metadata ----------------------------------------------------
    gid = data.get("id", "")
    if not isinstance(gid, str) or not _GALAXY_ID_RE.match(gid):
        errors.append("id is missing or contains invalid characters")

    name = data.get("name", "")
    if not isinstance(name, str) or not name.strip():
        errors.append("name is empty or missing")

    refs = data.get("references")
    if not isinstance(refs, list) or len(refs) == 0:
        warnings.append("references list is empty")
    elif not all(isinstance(r, str) and r.strip() for r in refs):
        warnings.append("references contains non-string or empty entries")

    # -- Distance and scale -------------------------------------------------
    dist = _safe_float(data.get("distance"))
    if dist is None or dist <= 0:
        errors.append("distance is not a positive finite number")

    galactic_radius = _safe_float(data.get("galactic_radius"))
    if galactic_radius is None or galactic_radius <= 0:
        errors.append("galactic_radius is not a positive finite number")

    # -- Velocity observations ----------------------------------------------
    obs = data.get("observations", [])
    if not isinstance(obs, list):
        errors.append("observations is not a list")
        obs = []

    if len(obs) < MIN_OBSERVATION_POINTS:
        errors.append(
            "only {} observation points (minimum {})".format(
                len(obs), MIN_OBSERVATION_POINTS
            )
        )

    radii = []
    for i, pt in enumerate(obs):
        if not isinstance(pt, dict):
            continue
        r = _safe_float(pt.get("r"))
        v = _safe_float(pt.get("v"))
        err = _safe_float(pt.get("err"))

        if r is None or r <= 0:
            errors.append("observations[{}]: r is not positive".format(i))
        else:
            radii.append(r)

        if v is None or v <= 0:
            errors.append("observations[{}]: v is not positive".format(i))

        if err is not None and err < 0:
            warnings.append("observations[{}]: err is negative".format(i))

    if radii:
        if radii != sorted(radii):
            warnings.append("observation radii are not sorted")

        if len(radii) != len(set(radii)):
            warnings.append("duplicate radii in observations")

        radial_span = max(radii) - min(radii)
        if radial_span < MIN_RADIAL_SPAN_KPC:
            warnings.append(
                "radial span is only {:.2f} kpc (minimum {})".format(
                    radial_span, MIN_RADIAL_SPAN_KPC
                )
            )

        max_obs_r = max(radii)
        if galactic_radius and galactic_radius > 0:
            if max_obs_r > galactic_radius * 1.2:
                warnings.append(
                    "max observed radius ({:.2f}) exceeds galactic_radius "
                    "({:.2f}) by more than 20%".format(max_obs_r, galactic_radius)
                )

    # -- Photometric data (mass model) --------------------------------------
    mm = data.get("mass_model", {})
    if not isinstance(mm, dict):
        errors.append("mass_model is not a dict")
        mm = {}

    disk = mm.get("disk", {})
    gas = mm.get("gas", {})
    bulge = mm.get("bulge", {})

    disk_m = _safe_float(disk.get("M") if isinstance(disk, dict) else None)
    disk_rd = _safe_float(disk.get("Rd") if isinstance(disk, dict) else None)
    gas_m = _safe_float(gas.get("M") if isinstance(gas, dict) else None)
    gas_rd = _safe_float(gas.get("Rd") if isinstance(gas, dict) else None)
    bulge_m = _safe_float(bulge.get("M") if isinstance(bulge, dict) else None)
    bulge_a = _safe_float(bulge.get("a") if isinstance(bulge, dict) else None)

    if disk_m is None or disk_m < 0:
        errors.append("disk.M is missing or negative")
    elif disk_m == 0:
        warnings.append("disk.M is zero (no stellar disk mass)")

    if disk_rd is None or disk_rd <= 0:
        errors.append("disk.Rd is missing or not positive")

    if gas_m is None or gas_m < 0:
        errors.append("gas.M is missing or negative")

    if gas_rd is None or gas_rd <= 0:
        errors.append("gas.Rd is missing or not positive")

    if bulge_m is None or bulge_m < 0:
        errors.append("bulge.M is missing or negative")

    if bulge_a is None or bulge_a < 0:
        errors.append("bulge.a is missing or negative")

    total_baryonic = (disk_m or 0) + (gas_m or 0) + (bulge_m or 0)
    if total_baryonic <= 0:
        errors.append("total baryonic mass is zero or negative")

    # Mass consistency: compare sum of components to stated mass field.
    # The "mass" field in SPARC source JSONs is in units of 10^9 M_sun,
    # while mass_model components are in M_sun.
    stated_mass = _safe_float(data.get("mass"))
    if stated_mass and stated_mass > 0 and total_baryonic > 0:
        stated_msun = stated_mass * 1e9
        log_diff = abs(math.log10(total_baryonic) - math.log10(stated_msun))
        if log_diff > MASS_CONSISTENCY_TOLERANCE:
            warnings.append(
                "mass_model total ({:.3e} M_sun) differs from stated mass "
                "({:.4g} x10^9 M_sun) by {:.2f} dex".format(
                    total_baryonic, stated_mass, log_diff
                )
            )

    # -- NaN / Inf sweep on all numeric top-level fields --------------------
    for key in ("distance", "galactic_radius", "mass", "accel"):
        val = data.get(key)
        if val is not None and not _is_finite(val):
            errors.append("{} is NaN or infinite".format(key))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
