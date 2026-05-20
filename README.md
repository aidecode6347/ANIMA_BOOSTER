# ANIMA_BOOSTER (BSS) | Version v1.2.0

🇷🇺 [Читать на русском языке](README_RU.md)

**ANIMA_BOOSTER (BSS)** is a high-performance optimization suite for the **Anima DiT 2B** model in ComfyUI. It is designed to deliver maximum performance, reduce VRAM usage, and accelerate generation speeds on NVIDIA graphics cards.

> [!IMPORTANT]
> **Author and Developer:** **blacksnowskill (BSS)**
> **© 2026 blacksnowskill (BSS). All rights reserved.**
> This project is protected by copyright. Any unauthorized copying, modification without attribution, or representing this code as your own product is strictly prohibited.

This package allows you to achieve a **total acceleration of 3.5× to 5.0×** compared to the default Anima workflow in ComfyUI, with no noticeable loss in visual quality.

---

## 🆕 What's New in Version v1.2 (Changelog)

### 💎 Global Refactoring and Stabilization:
1. **Removed Redundant Nodes (Codebase Cleanup):**
   - **The experimental `AnimaSparseAttention` node has been completely removed**. Using local sparse attention on 28 blocks of a model trained on Full Attention disrupted the global image structure and deformed geometry (causing severe artifacts).
   - **The complex `AnimaTorchCompile` node has been removed**. All of its fine-tuning options were redundant and caused critical PyTorch crashes with CUDA Graphs tensor overwrite errors when handling dynamic latent dimensions.
2. **Safe One-Click JIT Compilation:**
   - Instead of an unstable external node, **stable JIT compilation (`torch.compile`) is now integrated directly into the loaders (`AnimaBoosterLoader` / `CheckpointLoader`) as a simple, single toggle**.
   - Compilation runs on the safe default `inductor` backend without CUDA Graphs, giving the same **+20% to +40% speed boost** while guaranteeing **100% stability** without crashes or failures.
3. **Fixed TeaCache Scaling Bug (Bulletproof Adaptive Mode):**
   - **The Issue:** For SDE-family samplers (e.g., `er_sde`, `sde gpu`) that operate on a sigma scale (`[14.6 .. 0.0]`), TeaCache mistakenly treated the very first step as a 98% denoising progress. This triggered aggressive caching right from the first frame — resulting in fast generation but completely breaking the image with heavy artifacts.
   - **The Solution:** Implemented **dynamic timestep scale auto-detection (`st.max_t`)**. Now, TeaCache mathematically determines the start and end of generation for **any sampler and schedule** (sigmas, 1000..0, or 1..0 scales). The adaptive mode works perfectly: early structural steps are protected from caching, while late-stage detailing is safely accelerated.

---

## ⚡ Performance Quick Overview

| Configuration | Average Acceleration | Description |
|---|---|---|
| fp16 Only | **1.4×** | Baseline precision optimization |
| fp16 + **SageAttention (BSS)** | **1.8–2.5×** | Ultra-fast 8-bit attention for DiT |
| fp16 + **Adaptive TeaCache (BSS)** | **1.5–2.0×** | Intelligent step caching |
| **fp16 + SageAttn + TeaCache** | 🚀 **2.5–3.5×** | Perfect balance of speed and quality |
| **+ Integrated torch.compile** | 💎 **3.5–5.0×** | Maximum performance boost (after a 2-3 step warm-up phase) |

> [!TIP]
> All attention and compilation optimizations are applied in isolation (`model.clone()`). This prevents corruption of ComfyUI's global cache and guarantees stable operation of other models.

---

## 🧩 `BSS/AnimaBooster` Node List

All nodes are registered under the `BSS/AnimaBooster` category and feature a unique brand identity:

1. 📥 **Anima Booster Loader (BSS)** (class `AnimaBoosterLoader`)
   - Loads the Anima DiT model in the optimized fp16 format.
   - **SageAttention**: Automatically applies accelerated 8-bit attention if installed in the system. If unavailable, it seamlessly falls back to built-in PyTorch SDPA without import errors.
   - **Torch Compile**: An integrated toggle for safe compilation of DiT blocks. Runs on the stable `inductor` backend (mode: `default`), boosting speeds by 30-40% without CUDA Graphs crashes.
2. 🎛️ **Anima TeaCache (BSS)** (class `AnimaTeaCache`)
   - Implements adaptive latent state caching based on denoising steps (TeaCache).
   - Saves up to 50-60% of computation during "stable" denoising phases.
3. 🖼️ **Anima Latent Image (BSS)** (class `AnimaLatentImage`)
   - A utility for generating empty latents with automatic size alignment to the Anima DiT patch grid (2x2), preventing tensor dimension mismatch errors.

---

## 📈 Key Innovations

### 1. Adaptive TeaCache Threshold
Unlike standard TeaCache implementations with a fixed threshold, the **BSS** version uses a **dynamic adaptive threshold** that evolves during the denoising process:
- **In early steps** (high noise, image structure formation), the threshold is automatically lowered to ensure maximum rendering precision and geometric accuracy.
- **In later steps** (details have stabilized, micro-texturing takes place), the threshold is raised, allowing up to 80% of block computations to be safely skipped without quality loss.

---

## 📋 Recommended Workflow

To achieve the best results, connect the nodes in the following sequence:

```
[ Anima Booster Loader (BSS) ] ── (Enable sage_attention: auto and torch_compile: True)
             ↓ (MODEL)
[ Anima TeaCache (BSS) ] (Recommended: threshold: 0.15, adaptive: ON)
             ↓ (MODEL)
        [ KSampler ]
```

---

## 🛠️ Installation & Dependencies on Windows

All resource-heavy optimization libraries (**SageAttention**) are **entirely optional**. The package is designed with *Graceful Degradation* in mind: if the libraries are not installed, the nodes will automatically disable patches or fallback to standard PyTorch mechanisms, guaranteeing crash-free execution in ComfyUI.

### 📦 Resolving SageAttention Compilation Issues (for Windows & Portable Builds)

Portable ComfyUI builds (which use an isolated `python_embeded` environment on Python 3.12 or 3.13) do not have C++ compilation tools (MSVC / Build Tools) installed. As a result, the standard `pip install sageattention` command will fail with a compilation error.

You can easily bypass this issue using one of the following methods:

#### Method 1: Automatic Installation via ComfyUI Manager (Recommended)
1. In ComfyUI Manager, search for and install the **ComfyUI-Sage-EasyInstall** node.
2. Restart ComfyUI.
3. This node will automatically analyze your system, install the correct version of `triton-windows`, and download the precompiled binary package (`.whl` wheel) of SageAttention tailored specifically for your Python and CUDA versions.

#### Method 2: Manual Installation of Precompiled Wheel Files
If you prefer to install everything manually in a portable setup:
1. Open a command prompt (CMD or PowerShell) in your main ComfyUI folder.
2. Install Triton for Windows:
   ```bash
   .\python_embeded\python.exe -m pip install triton-windows
   ```
3. Download the precompiled `.whl` file for SageAttention that matches your Python version (e.g., `cp313` for Python 3.13) and CUDA version (e.g., `cu124` / `cu128`):
   - Precompiled builds can be found in the releases of this repository: **[sdbds/SageAttention-for-windows](https://github.com/sdbds/SageAttention-for-windows)**.
   - Precompiled wheels are also published in the project: **[wildminder/AI-windows-whl](https://github.com/wildminder/AI-windows-whl)**.
4. Install the downloaded file into the portable environment:
   ```bash
   .\python_embeded\python.exe -m pip install <path_to_downloaded_file.whl>
   ```

---

## 🧑‍💻 Technical Implementation Details for Developers

- **Model Base**: Anima DiT is based on the `MiniTrainDIT` architecture with the `LLMAdapter` wrapper.
- **SageAttention Integration Point**: The patch is applied via the standard `transformer_options["optimized_attention_override"]` parameter dictionary.
- **Torch Compile Integration Point**: Invokes the built-in `set_torch_compile_wrapper` function from `comfy_api.torch_helpers` at the level of individual transformer blocks (ensuring LoRA compatibility and reducing compilation overhead).
- **Isolation**: All graph and weight modifications are performed strictly on a cloned copy of the model (`model.clone()`). This prevents conflicts and leaves the original model in the ComfyUI cache untouched.
