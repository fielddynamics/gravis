"""
Architecture Overview Service.

Provides the platform architecture documentation page. This is a
read-only service with no computation endpoints; it exists in the
registry so its card appears on the home dashboard.

Status: live (documentation page, no API computation)
"""

from physics.services import GravisService


class ArchitectureService(GravisService):

    id = "architecture"
    name = "Architecture"
    description = "Platform design, service registry, and pipeline overview"
    category = "platform"
    status = "live"
    route = "/architecture"

    def validate(self, config):
        raise NotImplementedError(
            "Architecture service has no computation endpoint")

    def compute(self, config):
        raise NotImplementedError(
            "Architecture service has no computation endpoint")

    # No register_routes: this is a documentation page with no API endpoints.
    # The page route is registered directly in app.py.
