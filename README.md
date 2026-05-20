# ANIMA_BOOSTER (BSS)

🇷🇺 [Читать на русском языке](README_RU.md)

**ANIMA_BOOSTER (BSS)** is a high-performance optimization suite for the **Anima DiT 2B** model in ComfyUI. It is designed to deliver maximum performance, reduce VRAM usage, and accelerate generation speeds on NVIDIA graphics cards.

> [!IMPORTANT]
> **Author and Developer:** **blacksnowskill (BSS)**
> **© 2026 blacksnowskill (BSS). All rights reserved.**
> This project is protected by copyright. Any unauthorized copying, modification without attribution, or representing this code as your own product is strictly prohibited.

This package allows you to achieve a **total acceleration of 3.5× to 5.0×** compared to the default Anima workflow in ComfyUI, with no noticeable loss in visual quality.

---

## ⚡ Key Features

* **Integrated JIT Compilation (`torch.compile`):** Safe, one-click compilation of DiT blocks built directly into the loaders. Runs on the stable `inductor` backend without CUDA Graphs, ensuring 100% stability and a speed boost of **20% to 40%**.
* **SageAttention:** Built-in support for ultra-fast 8-bit attention tailored for DiT models, significantly accelerating computations while reducing memory consumption.
* **Adaptive TeaCache:** Intelligent latent state caching. Dynamically adjusts the caching threshold: automatically lowering it in early steps to preserve geometry, and raising it in later steps for maximum acceleration.
* **BSS Premium UI:** An integrated, high-contrast dark theme for the package's nodes. Features a fully redesigned full-width slider control, adaptive visibility for inactive preset inputs, and complete suppression of intrusive tooltips for a distraction-free experience.

---

## ⚡ Performance Quick Overview

| Configuration | Average Acceleration | Description |
|---|---|---|
| fp16 Only | **1.4×** | Baseline precision optimization |
| fp16 + **SageAttention** | **1.8–2.5×** | Ultra-fast 8-bit attention for DiT |
| fp16 + **Adaptive TeaCache** | **1.5–2.0×** | Intelligent step caching |
| **fp16 + SageAttn + TeaCache** | 🚀 **2.5–3.5×** | Perfect balance of speed and quality |
| **+ Integrated torch.compile** | 💎 **3.5–5.0×** | Maximum performance boost (after a 2-3 step warm-up phase) |

---

## 📥 Installation

### Method 1: Via ComfyUI Manager (Recommended)
1. Open ComfyUI and click on the **Manager** button.
2. Click **Install via Git URL**.
3. Paste this repository URL: `https://github.com/BlackSnowSkill/ANIMA_BOOSTER`
4. Click **Install**, wait for the process to complete, and restart ComfyUI.

### Method 2: Manual Installation (Git Clone)
1. Open your terminal and navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/BlackSnowSkill/ANIMA_BOOSTER.git
   ```
3. Restart ComfyUI.

---

## 🧩 `BSS/AnimaBooster` Node List

All nodes are registered under the `BSS/AnimaBooster` category:

1. 📥 **Anima Booster Loader (BSS)** (class `AnimaBoosterLoader`)
   - Loads the Anima DiT model in the optimized fp16 format.
   - **SageAttention**: Automatically applies accelerated 8-bit attention if installed in the system. If unavailable, it seamlessly falls back to built-in PyTorch SDPA.
   - **Torch Compile**: An integrated toggle for safe JIT compilation of individual transformer blocks.
2. 🎛️ **Anima TeaCache (BSS)** (class `AnimaTeaCache`)
   - Implements adaptive latent state caching based on denoising steps (TeaCache).
   - Automatically calculates the timestep scale for all sampler types (including SDE).
3. 🖼️ **Anima Latent Image (BSS)** (class `AnimaLatentImage`)
   - A utility for generating empty latents with automatic size alignment to the Anima DiT patch grid (2x2), preventing tensor dimension mismatch errors. Provides predefined aspect ratio presets.

---

## 📈 Adaptive TeaCache Threshold

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

All resource-heavy optimization libraries (**SageAttention** and JIT **Triton**) are **entirely optional**. The package is designed with *Graceful Degradation* in mind: if the libraries are not installed, the nodes will automatically disable patches or compile features, transitioning seamlessly to standard PyTorch mechanisms and guaranteeing crash-free execution in ComfyUI.

> [!IMPORTANT]
> **Triton is required for both SageAttention and JIT Compilation (`torch.compile`)**!
> If you plan to enable `torch_compile` (which yields up to a 40% speed boost) or use SageAttention on Windows, you **must** install `triton-windows`. Without Triton, `torch.compile` will be safely disabled with a warning in the console to avoid crashes.

### 📦 Installing Triton and SageAttention (for Windows & Portable Builds)

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
