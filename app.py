"""
GRAVIS - GRAvity VISion
Flask application factory.

Serves the frontend (Jinja2 template + static assets) and the REST API
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

    # Serve the main page
    @app.route("/")
    def index():
        return render_template("index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=5000)
