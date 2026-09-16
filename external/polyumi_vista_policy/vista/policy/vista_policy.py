"""Deprecated config-assembled VistaPolicy — use BaseVistaPolicy subclasses."""

from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class VistaPolicy(BaseImagePolicy):
    """
    Removed: architecture-registry assembly.

    Use one of:
      - vista.models.see_hear_feel.SeeHearFeelPolicy
      - vista.models.sparsh_x.SparshXPolicy
      - vista.models.polytouch.PolyTouchPolicy
      - vista.models.qformer.QformerPolicy
      - vista.models.mitas.MitasPolicy
      - vista.models.vta_diffusion.VTADiffusionPolicy
    """

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "VistaPolicy (config-assembled conditioner) was removed. "
            "Point Hydra _target_ at SeeHearFeelPolicy, SparshXPolicy, "
            "PolyTouchPolicy, QformerPolicy, MitasPolicy, or VTADiffusionPolicy "
            "(all subclass BaseVistaPolicy)."
        )
