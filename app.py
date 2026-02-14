"""
GRAVIS - GRAvity VISion
Flask application factory.

Serves the frontend (Jinja2 templates + static assets) and the REST API
for computing rotation curves with Dual Tetrad Gravity.

Usage:
    python app.py              # Development server on http://localhost:5000
    flask run                  # Same, via Flask CLI
"""

from flask import Flask, render_template


def create_app():
    """Application factory for GRAVIS Flask app."""
    app = Flask(
        __name__,
        static_folder="static",
        template_folder="templates",
    )

    # Register API blueprint
    from api.routes import api
    app.register_blueprint(api)

    # Lightweight splash screen (loads instantly, no external scripts)
    @app.route("/splash")
    def splash():
        return render_template("splash.html")

    # Analysis page (main app with Chart.js and heavy scripts)
    @app.route("/")
    def index():
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

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=5000)
