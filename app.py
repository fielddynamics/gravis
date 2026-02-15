"""
GRAVIS - Gravity Field Dynamics Research Platform
Flask application factory.

Serves the frontend (Jinja2 templates + static assets) and the REST API
for physics computations via registered GravisService instances.

Usage:
    python app.py              # Development server on http://localhost:5000
    flask run                  # Same, via Flask CLI
"""

__version__ = "0.1.0"

from flask import Flask, render_template

from physics.services import GravisRegistry
from physics.services.rotation import RotationService
from physics.services.rar import RARService
from physics.services.redshift import RedshiftService
from physics.services.solar import SolarService
from physics.services.valence import ValenceService
from physics.services.a0_derivation import A0DerivationService
from physics.services.nuclear_decay import NuclearDecayService
from physics.services.architecture import ArchitectureService


def create_registry():
    """Build and populate the service registry."""
    registry = GravisRegistry()
    registry.register(RotationService())
    registry.register(RARService())
    registry.register(RedshiftService())
    registry.register(SolarService())
    registry.register(ValenceService())
    registry.register(A0DerivationService())
    registry.register(NuclearDecayService())
    registry.register(ArchitectureService())
    return registry


def create_app():
    """Application factory for GRAVIS Flask app."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )

    # Make app version available to all templates
    @app.context_processor
    def inject_version():
        return {"version": __version__}

    # Build service registry
    registry = create_registry()

    # Create and register API blueprint (shared + service-owned routes)
    from api.routes import create_api_blueprint
    api = create_api_blueprint(registry)
    app.register_blueprint(api)

    # Lightweight splash screen (loads instantly, no external scripts)
    @app.route("/splash")
    def splash():
        return render_template("splash.html")

    # Home dashboard (auto-populated from registry)
    @app.route("/")
    def home():
        return render_template(
            "home.html",
            active_page="home",
            services=registry.list_all(),
        )

    # Analysis page (main app with Chart.js and heavy scripts)
    @app.route("/analysis")
    def analysis():
        return render_template("analysis.html", active_page="analysis")

    # About page (prose only, no heavy JS)
    @app.route("/about")
    def about():
        return render_template("about.html", active_page="about")

    # Field page (prose + inline Three.js stellated octahedron)
    @app.route("/field")
    def field():
        return render_template("field.html", active_page="field")

    # FAQ page (lightweight JS for search/filter)
    @app.route("/faq")
    def faq():
        return render_template("faq.html", active_page="faq")

    # Architecture overview page
    @app.route("/architecture")
    def architecture():
        return render_template("architecture.html", active_page="architecture")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=5000)
