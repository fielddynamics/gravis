"""
GRAVIS Service Layer: GravisService ABC and GravisRegistry.

Each physics domain (rotation curves, RAR, redshift, etc.) is a
GravisService registered with the GravisRegistry. The registry provides
lightweight dependency injection: services are looked up by ID at
runtime, and each service owns its own API endpoints, config schema,
and result format.

Classes:
    GravisService  - Abstract base class for all physics services
    GravisRegistry - Central lookup container for registered services

IMPORTANT: No unicode characters allowed (Windows charmap constraint).
"""

from abc import ABC, abstractmethod


class GravisService(ABC):
    """
    Abstract base class for a GRAVIS physics service.

    Each service represents a self-contained physics domain (e.g.
    galactic rotation, radial acceleration relation, redshift dynamics).
    Services define their own config validation, computation logic,
    and API endpoint registration.

    Class Attributes
    ----------------
    id : str
        Unique service identifier (e.g. "rotation", "rar").
    name : str
        Human-readable display name (e.g. "Rotation Curves").
    description : str
        One-liner for dashboard cards.
    category : str
        Grouping for the home dashboard. One of:
        "galactic", "cosmological", "quantum_nuclear".
    status : str
        "live" or "coming_soon".
    route : str
        Frontend page route (e.g. "/analysis", "/rar").
    """

    id = ""
    name = ""
    description = ""
    category = ""
    status = "coming_soon"
    route = ""

    @abstractmethod
    def validate(self, config):
        """
        Validate raw input and return normalized config dict.

        Parameters
        ----------
        config : dict
            Raw request payload.

        Returns
        -------
        dict
            Normalized, validated configuration.

        Raises
        ------
        ValueError
            If the config is invalid.
        """

    @abstractmethod
    def compute(self, config):
        """
        Run the service computation and return results.

        Parameters
        ----------
        config : dict
            Validated configuration from validate().

        Returns
        -------
        dict
            JSON-serializable result with domain-specific keys.
        """

    def register_routes(self, blueprint):
        """
        Mount service-specific API endpoints onto a Flask blueprint.

        Live services override this to register their namespaced
        endpoints (e.g. /api/rotation/curve). Coming-soon stubs
        inherit this no-op.

        Parameters
        ----------
        blueprint : flask.Blueprint
            The API blueprint to mount routes on.
        """
        pass

    def metadata(self):
        """
        Return service metadata for the registry and dashboard.

        Returns
        -------
        dict
            Service info: id, name, description, category, status, route.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "status": self.status,
            "route": self.route,
        }


class GravisRegistry:
    """
    Central lookup container for registered GravisService instances.

    Services register themselves at app startup. The registry provides
    lookup by ID, listing for the home dashboard, and iteration over
    live services for API route mounting.
    """

    def __init__(self):
        self._services = {}

    def register(self, service):
        """
        Register a service instance.

        Parameters
        ----------
        service : GravisService
            The service to register. Must have a unique id.

        Raises
        ------
        ValueError
            If a service with the same id is already registered.
        """
        if service.id in self._services:
            raise ValueError(
                "Service '{}' is already registered".format(service.id)
            )
        self._services[service.id] = service

    def get(self, service_id):
        """
        Look up a service by id.

        Parameters
        ----------
        service_id : str
            The service identifier.

        Returns
        -------
        GravisService or None
            The service, or None if not found.
        """
        return self._services.get(service_id)

    def list_all(self):
        """
        Return metadata for all registered services.

        Returns
        -------
        list of dict
            One metadata dict per service, in registration order.
        """
        return [s.metadata() for s in self._services.values()]

    def live(self):
        """
        Return all services with status 'live'.

        Returns
        -------
        list of GravisService
            Live service instances.
        """
        return [s for s in self._services.values()
                if s.status == "live"]
