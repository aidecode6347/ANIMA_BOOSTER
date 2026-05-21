"""
© 2026 blacksnowskill (BSS). All rights reserved.
Developed by: blacksnowskill (BSS)

nodes/node_loader.py
AnimaBoosterLoader — Optimized Anima model loader (BSS).

IMPORTANT: Anima (anima_baseV10) is saved in bfloat16.
Forcing fp16 causes black images due to value range mismatch.
We match exactly what ComfyUI's built-in UNETLoader does:
  - "default" / "bf16": no dtype override (auto-detection, uses bf16)

Speedup comes from SageAttention and TeaCache nodes.
"""

import logging
import torch
import folder_paths
import comfy.sd

from ..core.attention_sage import SAGE_MODES, apply_sage_attention_to_model
from ..core.compile_utils import detect_and_compile_blocks

logger = logging.getLogger("ANIMA_BOOSTER.loader")


class AnimaBoosterLoader:
    """
    Optimized loader for Anima DiT models.
    
    Loads the model using ComfyUI's standard pipeline (identical to UNETLoader),
    then optionally applies SageAttention and torch.compile.
    
    NOTE: Anima is stored in bfloat16 — do NOT force fp16.
    Use 'default' weight_dtype for correct output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    folder_paths.get_filename_list("diffusion_models"),
                    {"tooltip": "Select your Anima .safetensors model file"},
                ),
                "sage_attention": (
                    SAGE_MODES,
                    {
                        "default": "auto",
                        "tooltip": (
                            "SageAttention: quantized attention kernel for faster inference. "
                            "auto: let sageattention pick best kernel. "
                            "disabled: no patch."
                        ),
                    },
                ),
                "torch_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Apply torch.compile to transformer blocks. "
                            "First 2-3 generations are very slow (compilation warmup). "
                            "Subsequent generations: ~20-40% faster."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "BSS/AnimaBooster"

    def load(
        self,
        model_name: str,
        sage_attention: str,
        torch_compile: bool,
    ):
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)
        logger.info(f"[AnimaBoosterLoader] Loading: {model_name} | dtype=default (bfloat16)")

        # Load via ComfyUI's official function (identical to UNETLoader with default dtype)
        model = comfy.sd.load_diffusion_model(model_path, model_options={})

        dm = model.get_model_object("diffusion_model")
        logger.info(
            f"[AnimaBoosterLoader] Loaded: {type(dm).__name__} | "
            f"dtype={model.model.get_dtype()} | "
            f"blocks={len(dm.blocks) if hasattr(dm, 'blocks') else '?'}"
        )

        # Apply SageAttention patch
        if sage_attention != "disabled":
            model, success = apply_sage_attention_to_model(model, sage_attention)
            if success:
                logger.info(f"[AnimaBoosterLoader] SageAttention successfully applied (mode: {sage_attention})")
            else:
                logger.info(
                    "[AnimaBoosterLoader] SageAttention is not installed or failed to load. "
                    "Model will run perfectly with native PyTorch Scaled Dot Product Attention (SDPA) fallback."
                )

        # Apply torch.compile
        if torch_compile:
            try:
                model = detect_and_compile_blocks(model, mode="default")
                logger.info("[AnimaBoosterLoader] torch.compile applied: default")
            except Exception as e:
                logger.error(f"[AnimaBoosterLoader] torch.compile failed: {e} — continuing without it")

        return (model,)


class AnimaBoosterCheckpointLoader:
    """
    Optimized loader for Anima DiT models packaged as full checkpoints (Model + CLIP + VAE).
    
    Loads the checkpoint using ComfyUI's standard pipeline (identical to CheckpointLoaderSimple),
    then optionally applies SageAttention and torch.compile to the model component.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Select your Anima .safetensors checkpoint file"},
                ),
                "sage_attention": (
                    SAGE_MODES,
                    {
                        "default": "auto",
                        "tooltip": (
                            "SageAttention: quantized attention kernel for faster inference. "
                            "auto: let sageattention pick best kernel. "
                            "disabled: no patch."
                        ),
                    },
                ),
                "torch_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Apply torch.compile to transformer blocks. "
                            "First 2-3 generations are very slow (compilation warmup). "
                            "Subsequent generations: ~20-40% faster."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "BSS/AnimaBooster"

    def load_checkpoint(
        self,
        ckpt_name: str,
        sage_attention: str,
        torch_compile: bool,
    ):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        logger.info(f"[AnimaBoosterCheckpointLoader] Loading: {ckpt_name}")

        # Load via ComfyUI's official function (identical to CheckpointLoaderSimple)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        model, clip, vae, clipvision = out

        dm = model.get_model_object("diffusion_model")
        logger.info(
            f"[AnimaBoosterCheckpointLoader] Loaded: {type(dm).__name__} | "
            f"dtype={model.model.get_dtype()} | "
            f"blocks={len(dm.blocks) if hasattr(dm, 'blocks') else '?'}"
        )

        # Apply SageAttention patch
        if sage_attention != "disabled":
            model, success = apply_sage_attention_to_model(model, sage_attention)
            if success:
                logger.info(f"[AnimaBoosterCheckpointLoader] SageAttention successfully applied (mode: {sage_attention})")
            else:
                logger.info(
                    "[AnimaBoosterCheckpointLoader] SageAttention is not installed or failed to load. "
                    "Model will run perfectly with native PyTorch Scaled Dot Product Attention (SDPA) fallback."
                )

        # Apply torch.compile
        if torch_compile:
            try:
                model = detect_and_compile_blocks(model, mode="default")
                logger.info("[AnimaBoosterCheckpointLoader] torch.compile applied: default")
            except Exception as e:
                logger.error(f"[AnimaBoosterCheckpointLoader] torch.compile failed: {e} — continuing without it")

        return (model, clip, vae)


class SdxlBoosterCheckpointLoader:
    """
    Optimized loader for SDXL checkpoints.
    
    Loads standard SDXL checkpoints (Model + CLIP + VAE), and optionally applies torch.compile to the model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (
                    folder_paths.get_filename_list("checkpoints"),
                    {"tooltip": "Select your SDXL .safetensors checkpoint file"},
                ),
                "torch_compile": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Apply torch.compile to the full diffusion UNet model. "
                            "First 2-3 generations are very slow (compilation warmup). "
                            "Subsequent generations: ~20-35% faster."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_checkpoint"
    CATEGORY = "BSS/AnimaBooster"

    def load_checkpoint(
        self,
        ckpt_name: str,
        torch_compile: bool,
    ):
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        logger.info(f"[SdxlBoosterCheckpointLoader] Loading: {ckpt_name}")

        # Load via ComfyUI's official function (identical to CheckpointLoaderSimple)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path, 
            output_vae=True, 
            output_clip=True, 
            embedding_directory=folder_paths.get_folder_paths("embeddings")
        )
        model, clip, vae, clipvision = out

        dm = model.get_model_object("diffusion_model")
        logger.info(
            f"[SdxlBoosterCheckpointLoader] Loaded: {type(dm).__name__} | "
            f"dtype={model.model.get_dtype()}"
        )

        # Apply torch.compile
        if torch_compile:
            try:
                # Compile the full UNet since UNets do not have standard block lists like DiTs
                model = detect_and_compile_blocks(model, mode="default")
                logger.info("[SdxlBoosterCheckpointLoader] torch.compile applied: default")
            except Exception as e:
                logger.error(f"[SdxlBoosterCheckpointLoader] torch.compile failed: {e} — continuing without it")

        return (model, clip, vae)


