"""
Redshift Dynamics Service.

Provides Tully-Fisher velocity evolution predictions using the GFD
covariant completion, cosmological distance calculations, and
comparison with Lambda-CDM.

Core prediction: v(z)/v(0) = [H(z)/H0]^0.17

Derivation chain (zero free parameters):
  1. The dual tetrad topology links the acceleration scale a0 to the
     Hubble parameter: a0 = c*H/(2*pi) * sqrt(k/pi), where k = 4.
     Because H evolves with redshift as H(z), so does a0.
  2. In the deep partial-coupling regime, field dynamics give the
     baryonic Tully-Fisher relation: v_flat^4 = G * M * a0.
     Rotation velocity scales as a0^(1/4).
  3. The scalar-tensor action of the covariant completion (Lagrangian
     F(y) = y - 2*sqrt(y) + 2*ln(1+sqrt(y)) and disformal coupling
     e^(-2*Phi/c^2)) modifies the effective TF slope, reducing the
     naive exponent from 0.25 to 0.17.

The exponent 0.17 is derived from the covariant completion's
scalar-tensor action, not fitted to TF observations.
"""

import math
from flask import jsonify, request
from physics.services import GravisService


# ---------------------------------------------------------------------------
# Cosmological helper functions
# ---------------------------------------------------------------------------

def hubble_parameter(z, H0, omega_m, omega_l):
    """Compute H(z) for a flat Lambda-CDM cosmology.

    H(z) = H0 * sqrt(Omega_m * (1+z)^3 + Omega_Lambda)

    Parameters
    ----------
    z : float
        Redshift
    H0 : float
        Hubble constant at z=0 (km/s/Mpc)
    omega_m : float
        Matter density parameter
    omega_l : float
        Dark energy density parameter

    Returns
    -------
    float
        H(z) in km/s/Mpc
    """
    return H0 * math.sqrt(omega_m * (1 + z) ** 3 + omega_l)


def comoving_distance(z, H0, omega_m, omega_l, n_steps=1000):
    """Comoving distance via numerical integration of 1/E(z').

    d_C = (c/H0) * integral_0^z dz' / E(z')
    where E(z) = H(z)/H0

    Returns distance in Mpc.
    """
    c_km_s = 299792.458  # speed of light in km/s
    dz = z / n_steps
    integral = 0.0
    for i in range(n_steps):
        z_mid = (i + 0.5) * dz
        e_z = math.sqrt(omega_m * (1 + z_mid) ** 3 + omega_l)
        integral += dz / e_z
    return (c_km_s / H0) * integral


def luminosity_distance(z, H0, omega_m, omega_l):
    """Luminosity distance: d_L = (1+z) * d_C. Returns Mpc."""
    dc = comoving_distance(z, H0, omega_m, omega_l)
    return (1 + z) * dc


def angular_diameter_distance(z, H0, omega_m, omega_l):
    """Angular diameter distance: d_A = d_C / (1+z). Returns Mpc."""
    dc = comoving_distance(z, H0, omega_m, omega_l)
    return dc / (1 + z)


def lookback_time(z, H0, omega_m, omega_l, n_steps=1000):
    """Lookback time via numerical integration.

    t_lb = (1/H0) * integral_0^z dz' / [(1+z') * E(z')]

    Returns time in Gyr.
    """
    h0_per_sec = H0 / 3.0857e19  # convert km/s/Mpc to 1/s
    dz = z / n_steps
    integral = 0.0
    for i in range(n_steps):
        z_mid = (i + 0.5) * dz
        e_z = math.sqrt(omega_m * (1 + z_mid) ** 3 + omega_l)
        integral += dz / ((1 + z_mid) * e_z)
    t_seconds = integral / h0_per_sec
    return t_seconds / (3600 * 24 * 365.25 * 1e9)  # convert to Gyr


def gfd_velocity_ratio(z, H0, omega_m, omega_l, exponent=0.17):
    """GFD Tully-Fisher velocity evolution prediction.

    v(z)/v(0) = [H(z)/H0]^exponent

    The exponent 0.17 is derived from the scalar-tensor action of
    the dual tetrad covariant completion. The derivation chain:
    a0 proportional to H(z) (topology) -> v_flat proportional to
    a0^(1/4) (deep partial coupling) -> disformal coupling and
    the three-level Lagrangian F(y) reduce 0.25 to 0.17.

    This is a zero-parameter prediction: no fitting to TF data.
    """
    hz = hubble_parameter(z, H0, omega_m, omega_l)
    return (hz / H0) ** exponent


# ---------------------------------------------------------------------------
# Service implementation
# ---------------------------------------------------------------------------

class RedshiftService(GravisService):

    id = "redshift"
    name = "Redshift Dynamics"
    description = "TF evolution, H(z), and the distance ladder"
    category = "cosmological"
    status = "live"
    route = "/redshift"

    def validate(self, config):
        """Validate and normalize redshift computation config."""
        z = float(config.get("z", 0))
        if z < 0:
            raise ValueError("Redshift z must be >= 0")
        if z > 20:
            raise ValueError("Redshift z must be <= 20")

        H0 = float(config.get("H0", 70.0))
        if not (50 <= H0 <= 100):
            raise ValueError("H0 must be between 50 and 100 km/s/Mpc")

        omega_m = float(config.get("omega_m", 0.30))
        if not (0.01 <= omega_m <= 0.99):
            raise ValueError("omega_m must be between 0.01 and 0.99")

        omega_l = 1.0 - omega_m  # flat universe

        v0 = float(config.get("v0", 220))
        if v0 <= 0:
            raise ValueError("Local velocity v0 must be positive")

        lens_pct = float(config.get("lens_pct", 6.2))
        if not (0 <= lens_pct <= 50):
            raise ValueError("Lens throughput must be between 0 and 50%")

        return {
            "z": z,
            "H0": H0,
            "omega_m": omega_m,
            "omega_l": omega_l,
            "v0": v0,
            "lens_pct": lens_pct,
        }

    def compute(self, config):
        """Compute redshift dynamics for a single z value."""
        z = config["z"]
        H0 = config["H0"]
        om = config["omega_m"]
        ol = config["omega_l"]
        v0 = config["v0"]

        hz = hubble_parameter(z, H0, om, ol)
        ratio = gfd_velocity_ratio(z, H0, om, ol)
        vz = v0 * ratio
        delta_pct = (ratio - 1.0) * 100

        result = {
            "z": z,
            "H0": H0,
            "omega_m": om,
            "omega_l": ol,
            "v0": v0,
            "hz": round(hz, 2),
            "gfd_ratio": round(ratio, 6),
            "gfd_vz": round(vz, 2),
            "delta_pct": round(delta_pct, 2),
            "lcdm_ratio": 1.0,
        }

        # Distance calculations (skip if z=0)
        if z > 0:
            result["comoving_distance_mpc"] = round(
                comoving_distance(z, H0, om, ol), 2
            )
            result["luminosity_distance_mpc"] = round(
                luminosity_distance(z, H0, om, ol), 2
            )
            result["angular_diameter_distance_mpc"] = round(
                angular_diameter_distance(z, H0, om, ol), 2
            )
            result["lookback_time_gyr"] = round(
                lookback_time(z, H0, om, ol), 3
            )
        else:
            result["comoving_distance_mpc"] = 0
            result["luminosity_distance_mpc"] = 0
            result["angular_diameter_distance_mpc"] = 0
            result["lookback_time_gyr"] = 0

        return result

    def compute_curve(self, config, z_max=4.0, n_points=200):
        """Compute GFD and LCDM curves over a redshift range."""
        H0 = config["H0"]
        om = config["omega_m"]
        ol = config["omega_l"]

        z_values = []
        gfd_ratios = []
        gfd_upper = []
        gfd_lower = []
        lcdm_ratios = []
        hz_values = []

        cage_pct = config.get("lens_pct", 6.2) / 100.0

        for i in range(n_points + 1):
            z = z_max * i / n_points
            z_values.append(round(z, 4))

            ratio = gfd_velocity_ratio(z, H0, om, ol)
            gfd_ratios.append(round(ratio, 6))
            gfd_upper.append(round(ratio * (1 + cage_pct), 6))
            gfd_lower.append(round(ratio * (1 - cage_pct), 6))

            lcdm_ratios.append(1.0)

            hz = hubble_parameter(z, H0, om, ol)
            hz_values.append(round(hz, 2))

        return {
            "z": z_values,
            "gfd": gfd_ratios,
            "gfd_upper": gfd_upper,
            "gfd_lower": gfd_lower,
            "lcdm": lcdm_ratios,
            "hz": hz_values,
        }

    def register_routes(self, bp):
        """Register redshift API endpoints on the given blueprint."""
        service = self

        @bp.route("/redshift/compute", methods=["POST"])
        def redshift_compute():
            data = request.get_json()
            try:
                config = service.validate(data)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            result = service.compute(config)
            return jsonify(result)

        @bp.route("/redshift/curve", methods=["POST"])
        def redshift_curve():
            data = request.get_json()
            try:
                config = service.validate(data)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400
            z_max = float(data.get("z_max", 4.0))
            n_points = int(data.get("n_points", 200))
            curves = service.compute_curve(config, z_max=z_max, n_points=n_points)
            return jsonify(curves)

        @bp.route("/redshift/examples", methods=["GET"])
        def redshift_examples():
            from data.redshift_data import REDSHIFT_EXAMPLES
            return jsonify(REDSHIFT_EXAMPLES)

        @bp.route("/redshift/observations", methods=["GET"])
        def redshift_observations():
            from data.redshift_data import TF_OBSERVATIONS, SINS_HIGHLIGHT
            return jsonify({
                "observations": TF_OBSERVATIONS,
                "sins_highlight": SINS_HIGHLIGHT,
            })

        @bp.route("/redshift/h0-measurements", methods=["GET"])
        def redshift_h0_measurements():
            from data.redshift_data import (
                H0_MEASUREMENTS,
                GFD_H0_TREE,
                GFD_H0_ONE_LOOP,
                MOND_H0_PREDICTED,
                H0_PREDICTED_RANGE_PCT,
            )
            return jsonify({
                "measurements": H0_MEASUREMENTS,
                "gfd_tree": GFD_H0_TREE,
                "gfd_one_loop": GFD_H0_ONE_LOOP,
                "mond_predicted": MOND_H0_PREDICTED,
                "range_pct": H0_PREDICTED_RANGE_PCT,
            })
