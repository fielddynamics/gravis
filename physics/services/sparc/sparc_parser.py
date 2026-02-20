"""
SPARC galaxy data parser and validator.

Validates galaxy dicts against the app schema (id, name, distance,
galactic_radius, mass, accel, mass_model, observations, references).
Use for import by id (from bundled JSON) or file upload (JSON).

IMPORTANT: No unicode in code or error messages (Windows charmap).
"""

import json


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
