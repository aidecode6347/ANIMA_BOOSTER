"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

nodes/node_teacache.py
AnimaTeaCache — Adaptive Timestep-aware Cache for Anima DiT (BSS).

Caches transformer block outputs between denoising steps when the
timestep embedding changes very little. Skips expensive recomputation
when it's not needed.

Key innovation: ADAPTIVE threshold — varies by position in denoising:
  • Early steps (structure forming) → low threshold (rarely skip)
  • Late steps  (details stable)    → high threshold (skip aggressively)
"""

import logging
from ..core.teacache_engine import patch_model_with_teacache

logger = logging.getLogger("ANIMA_BOOSTER.teacache_node")


class AnimaTeaCache:
    """
    Adaptive TeaCache for Anima DiT models.

    Speeds up inference by skipping redundant transformer computations
    between similar denoising steps. Typically provides 1.5–2.0× speedup.

    The ADAPTIVE mode is the key innovation:
      - Uses a higher threshold (skips more) for late denoising steps
        where image details are already stable
      - Uses a lower threshold (computes more) for early steps
        where global structure is still being established

    Place this node AFTER AnimaBoosterLoader (or any model loader)
    and BEFORE the KSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Connect from AnimaBoosterLoader or Load Diffusion Model"
                }),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.15,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": (
                            "Base cache threshold. Higher = more skipping = faster but may "
                            "reduce quality. Start with 0.15. "
                            "0.10 = conservative | 0.15 = balanced | 0.25 = aggressive"
                        ),
                    },
                ),
                "adaptive_mode": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Adaptive threshold adjusts dynamically by timestep. "
                            "RECOMMENDED: ON. Gives better quality/speed trade-off "
                            "than fixed threshold."
                        ),
                    },
                ),
                "early_steps_factor": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": (
                            "Threshold multiplier for early denoising steps (high noise). "
                            "Lower = more accurate structure. Range: 0.2–0.6"
                        ),
                    },
                ),
                "late_steps_factor": (
                    "FLOAT",
                    {
                        "default": 1.8,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "display": "slider",
                        "tooltip": (
                            "Threshold multiplier for late denoising steps (low noise). "
                            "Higher = more aggressive caching of detail steps. Range: 1.5–2.5"
                        ),
                    },
                ),
            },
            "optional": {
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "% of denoising steps from which to enable TeaCache (0.0 = from start)",
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "% of denoising steps until which TeaCache is active (1.0 = until end)",
                    },
                ),
                "cache_device": (
                    ["cuda", "cpu"],
                    {
                        "default": "cuda",
                        "tooltip": (
                            "cuda: Store cached residuals on GPU (faster, uses more VRAM). "
                            "cpu: Store on RAM (slower transfer but saves VRAM)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "BSS/AnimaBooster"
    DESCRIPTION = (
        "Adaptive TeaCache: accelerates Anima by skipping redundant computations "
        "between similar denoising steps. Uses adaptive thresholds for better "
        "quality/speed balance than fixed-threshold implementations."
    )

    def apply(
        self,
        model,
        threshold: float,
        adaptive_mode: bool,
        early_steps_factor: float,
        late_steps_factor: float,
        start_percent: float = 0.0,
        end_percent: float = 1.0,
        cache_device: str = "cuda",
    ):
        patched = patch_model_with_teacache(
            model=model,
            threshold=threshold,
            adaptive=adaptive_mode,
            early_factor=early_steps_factor,
            late_factor=late_steps_factor,
            start_percent=start_percent,
            end_percent=end_percent,
            cache_device=cache_device,
        )
        return (patched,)
