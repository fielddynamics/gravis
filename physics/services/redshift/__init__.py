"""
Redshift Dynamics Service stub.

Will provide Tully-Fisher evolution with redshift, H(z) distance
ladder calculations, and Hubble tension analysis using DTG
predictions.

Status: coming_soon
"""

from physics.services import GravisService


class RedshiftService(GravisService):

    id = "redshift"
    name = "Redshift Dynamics"
    description = "TF evolution, H(z), and the distance ladder"
    category = "cosmological"
    status = "coming_soon"
    route = "/redshift"

    def validate(self, config):
        raise NotImplementedError("Redshift service is not yet implemented")

    def compute(self, config):
        raise NotImplementedError("Redshift service is not yet implemented")
