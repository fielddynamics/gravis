"""
Nuclear Decay Service stub.

Will predict half-lives and decay modes from tetrahedral geometry
constraints, comparing DTG predictions against measured nuclear
data for selected isotopes.

Status: coming_soon
"""

from physics.services import GravisService


class NuclearDecayService(GravisService):

    id = "nuclear_decay"
    name = "Nuclear Decay"
    description = "Half-life predictions from tetrahedral geometry"
    category = "quantum_nuclear"
    status = "coming_soon"
    route = "/nuclear-decay"

    def validate(self, config):
        raise NotImplementedError(
            "Nuclear decay service is not yet implemented")

    def compute(self, config):
        raise NotImplementedError(
            "Nuclear decay service is not yet implemented")
