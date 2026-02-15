"""
Flask API routes for GRAVIS.

The routes file is thin: it owns only the shared endpoints (registry
listing, physical constants). All domain-specific endpoints are mounted
by their respective GravisService via register_routes().

Shared Endpoints:
    GET  /api/registry    - list all registered services
    GET  /api/constants   - physical constants used by the engine

Service-owned endpoints are mounted at startup via create_api_blueprint().
Each live service calls register_routes(blueprint) to add its own
namespaced endpoints (e.g. /api/rotation/curve, /api/rotation/galaxies).
"""

from flask import Blueprint, jsonify

from physics import constants


def create_api_blueprint(registry):
    """
    Create and return the API blueprint with all routes.

    Creates a fresh blueprint each time to avoid Flask's restriction
    on adding routes after registration. Each live service mounts
    its own named endpoints onto this blueprint.

    Parameters
    ----------
    registry : GravisRegistry
        The application's service registry.

    Returns
    -------
    flask.Blueprint
        The fully configured API blueprint.
    """
    api = Blueprint("api", __name__, url_prefix="/api")

    @api.route("/registry", methods=["GET"])
    def list_services():
        """Return metadata for all registered services."""
        return jsonify(registry.list_all())

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

    # SRP: each live service mounts its own /api/<id>/* endpoints
    for service in registry.live():
        service.register_routes(api)

    return api
