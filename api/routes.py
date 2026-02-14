"""
Flask API routes for GRAVIS rotation curve analysis.

Endpoints:
  GET  /api/galaxies             - list all galaxies by mode
  GET  /api/galaxies/<id>        - get single galaxy details
  POST /api/rotation-curve       - compute rotation curves for given parameters
  POST /api/infer-mass           - infer mass from observed velocity
  POST /api/infer-mass-model     - infer scaled mass model from observation + shape
"""

import math

from flask import Blueprint, jsonify, request

from physics import constants
from physics.mass_model import enclosed_mass, total_mass
from physics.newtonian import velocity as newtonian_velocity
from physics.aqual import velocity as dtg_velocity
from physics.mond import velocity as mond_velocity
from physics.inference import infer_mass
from data.galaxies import get_all_galaxies, get_galaxy_by_id

api = Blueprint("api", __name__, url_prefix="/api")


@api.route("/galaxies", methods=["GET"])
def list_galaxies():
    """Return all galaxies grouped by mode (prediction / inference)."""
    return jsonify(get_all_galaxies())


@api.route("/galaxies/<galaxy_id>", methods=["GET"])
def get_galaxy(galaxy_id):
    """Return a single galaxy by id."""
    galaxy = get_galaxy_by_id(galaxy_id)
    if galaxy is None:
        return jsonify({"error": "Galaxy not found"}), 404
    return jsonify(galaxy)


@api.route("/rotation-curve", methods=["POST"])
def compute_rotation_curve():
    """
    Compute Newtonian, DTG, and MOND rotation curves.

    Request JSON:
    {
        "max_radius": 30,          // kpc
        "num_points": 100,         // number of radial points
        "accel_ratio": 1.0,        // a/a0 multiplier
        "mass_model": {            // distributed mass model
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 2.5},
            "gas":   {"M": 1.0e10, "Rd": 5.0}
        }
    }

    Response JSON:
    {
        "radii": [...],            // kpc
        "newtonian": [...],        // km/s
        "dtg": [...],              // km/s
        "mond": [...],             // km/s
        "enclosed_mass": [...]     // M_sun at each radius
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    max_radius = data.get("max_radius", 30)
    num_points = data.get("num_points", 100)
    accel_ratio = data.get("accel_ratio", 1.0)
    mass_model = data.get("mass_model")

    if not mass_model:
        return jsonify({"error": "mass_model is required"}), 400

    # Validate num_points (cap at 500 for safety)
    num_points = min(int(num_points), 500)
    num_points = max(num_points, 10)

    radii = []
    newtonian = []
    dtg = []
    mond = []
    enc_mass = []

    for i in range(num_points):
        r = (max_radius / num_points) * (i + 1)
        m_at_r = enclosed_mass(r, mass_model)

        radii.append(round(r, 6))
        enc_mass.append(round(m_at_r, 2))
        newtonian.append(round(newtonian_velocity(r, m_at_r), 4))
        dtg.append(round(dtg_velocity(r, m_at_r, accel_ratio), 4))
        mond.append(round(mond_velocity(r, m_at_r, accel_ratio), 4))

    return jsonify({
        "radii": radii,
        "newtonian": newtonian,
        "dtg": dtg,
        "mond": mond,
        "enclosed_mass": enc_mass,
    })


@api.route("/infer-mass", methods=["POST"])
def infer_mass_endpoint():
    """
    Infer enclosed baryonic mass from observed velocity.

    Request JSON:
    {
        "r_kpc": 8.0,             // galactocentric radius
        "v_km_s": 230.0,          // observed circular velocity
        "accel_ratio": 1.0        // a/a0 multiplier
    }

    Response JSON:
    {
        "inferred_mass_solar": 1.23e11,
        "log10_mass": 11.09
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    r_kpc = data.get("r_kpc", 8.0)
    v_km_s = data.get("v_km_s", 230.0)
    accel_ratio = data.get("accel_ratio", 1.0)

    mass = infer_mass(r_kpc, v_km_s, accel_ratio)

    return jsonify({
        "inferred_mass_solar": mass,
        "log10_mass": round(math.log10(max(mass, 1.0)), 4),
    })


@api.route("/infer-mass-model", methods=["POST"])
def infer_mass_model_endpoint():
    """
    Infer a scaled mass model from an observation and a distribution shape.

    Given an observed (r, v) and a mass model "shape" (scale lengths +
    relative proportions), compute the total mass that makes DTG predict
    exactly v at r, then scale all components proportionally.

    Also returns the BTFR (Baryonic Tully-Fisher) mass estimate:
      M_BTFR = v^4 / (G * a0)

    Request JSON:
    {
        "r_kpc": 8.0,
        "v_km_s": 230.0,
        "accel_ratio": 1.0,
        "mass_model": {
            "bulge": {"M": 1.5e10, "a": 0.6},
            "disk":  {"M": 5.0e10, "Rd": 2.5},
            "gas":   {"M": 1.0e10, "Rd": 5.0}
        }
    }

    Response JSON:
    {
        "inferred_enclosed": 5.5e10,
        "scale_factor": 0.95,
        "inferred_mass_model": {
            "bulge": {"M": 1.425e10, "a": 0.6},
            "disk":  {"M": 4.75e10, "Rd": 2.5},
            "gas":   {"M": 0.95e10, "Rd": 5.0}
        },
        "inferred_total": 7.125e10,
        "log10_total": 10.853,
        "btfr_mass": 7.0e10,
        "log10_btfr": 10.845
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    r_kpc = data.get("r_kpc", 8.0)
    v_km_s = data.get("v_km_s", 230.0)
    accel_ratio = data.get("accel_ratio", 1.0)
    mass_model = data.get("mass_model")

    if not mass_model:
        return jsonify({"error": "mass_model is required"}), 400

    # Step 1: Infer M(<r) from the observation using DTG inverse
    m_enclosed_needed = infer_mass(r_kpc, v_km_s, accel_ratio)

    # Step 2: Compute M(<r) for the current mass model shape
    m_enclosed_current = enclosed_mass(r_kpc, mass_model)
    m_total_current = total_mass(mass_model)

    if m_enclosed_current <= 0 or m_total_current <= 0:
        return jsonify({"error": "Mass model has zero enclosed mass at observation radius"}), 400

    # Step 3: Scale factor to match observation
    scale = m_enclosed_needed / m_enclosed_current

    # Step 4: Build scaled mass model (preserve scale lengths, scale masses)
    scaled_model = {}
    for comp in ("bulge", "disk", "gas"):
        if comp in mass_model and mass_model[comp].get("M", 0) > 0:
            entry = dict(mass_model[comp])
            entry["M"] = entry["M"] * scale
            scaled_model[comp] = entry
        elif comp in mass_model:
            scaled_model[comp] = dict(mass_model[comp])

    inferred_total = total_mass(scaled_model)

    # Step 5: BTFR mass estimate: M = v^4 / (G * a0)
    v_ms = v_km_s * 1000.0
    a0_eff = constants.A0 * accel_ratio
    btfr_mass = (v_ms ** 4) / (constants.G * a0_eff) / constants.M_SUN

    return jsonify({
        "inferred_enclosed": round(m_enclosed_needed, 2),
        "scale_factor": round(scale, 6),
        "inferred_mass_model": scaled_model,
        "inferred_total": round(inferred_total, 2),
        "log10_total": round(math.log10(max(inferred_total, 1.0)), 4),
        "btfr_mass": round(btfr_mass, 2),
        "log10_btfr": round(math.log10(max(btfr_mass, 1.0)), 4),
    })


@api.route("/constants", methods=["GET"])
def get_constants():
    """Return the physical constants used by the engine."""
    return jsonify({
        "G": constants.G,
        "M_SUN": constants.M_SUN,
        "KPC_TO_M": constants.KPC_TO_M,
        "M_E": constants.M_E,
        "R_E": constants.R_E,
        "K_SIMPLEX": constants.K_SIMPLEX,
        "A0": constants.A0,
    })
