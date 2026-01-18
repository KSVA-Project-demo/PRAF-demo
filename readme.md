# Stealthy and Robust: A Physically-Realizable Black-Box Attack Framework for Embodied Semantic Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Conference](https://img.shields.io/badge/IJCAI-2026-blue)](https://ijcai.org/)

This repository contains the official implementation of **PRAF** (Physically-Realizable Attack Framework), a novel adversarial attack framework designed for embodied AI semantic segmentation systems.

## 📖 Introduction

Semantic segmentation serves as the perceptual foundation for embodied AI agents, such as autonomous vehicles and humanoid robots. However, current physical adversarial attacks often fail to bridge the **Sim-to-Real gap** or produce conspicuous patterns that are easily detected.

We propose **PRAF**, a closed-loop framework that generates stealthy and robust adversarial patches in a black-box setting. By incorporating a **Differentiable Physical Simulation Layer** into the optimization loop, PRAF mimics real-world distortions (e.g., motion blur, environmental noise), ensuring the generated patches remain effective against dynamic kinematics and diverse viewing angles.

**Key Contributions:**
* **🛡️ Sim-to-Real Robustness:** Patches are trained with differentiable physical simulations, ensuring resilience against camera jitter, motion blur, and affine transformations.
* **🔄 Cross-Architecture Transferability:** Patches generated using a ResNet-50 substitute effectively compromise Transformer-based models (e.g., SegFormer, Mask2Former) by exploiting shared low-frequency structural vulnerabilities.
* **🤖 Dual-Platform Validation:** Validated on two distinct physical platforms: an **Intelligent Robotic Vehicle** (Road scenarios) and a **Humanoid Robot** (Dynamic corridors).
* **👁️ Visual Stealthiness:** optimized with stealthiness constraints (TV loss + LPIPS) to blend naturally with environmental textures.

## 🧠 Methodology

The PRAF framework operates as an iterative optimization loop consisting of three coupled components:

1.  **Latent-Driven Generator ($G$):** Synthesizes the initial patch from a mixed latent noise distribution to prevent overfitting to the substitute model.
2.  **Differentiable Physical Simulation Layer ($\Phi$):**
    * Mathematically models environmental degradations (Motion Blur, Environmental Noise).
    * **Mechanism:** Acts as a **spectral low-pass filter** during backpropagation, suppressing high-frequency noise and forcing the generator to learn robust, low-frequency structural perturbations.
3.  **Substitute-Guided Optimization:** Uses a white-box substitute (e.g., ResNet) to estimate gradients for black-box targets, maximizing attack loss while maintaining visual naturalness.

![Framework Overview](docs/framework_overview.png)
*(Figure 2: The architecture of PRAF)*

## 📊 Experimental Results

### 1. Digital Cross-Architecture Transferability

We evaluated the transferability of patches generated from a ResNet-50 substitute to various black-box target models and defense mechanisms. PRAF achieves state-of-the-art transferability, especially on Transformer-based architectures and against defense models.

**Metric:** Mean Intersection over Union (mIoU) - *Lower is better for attacks.*

| Attack Method | DeepLabV3+ (CNN) | SegFormer (ViT) | Mask2Former (ViT) | DDCAT (Defense) |
| :--- | :---: | :---: | :---: | :---: |
| **SASS** (2024) | 38.9% | 18.7% | 17.6% | 11.8% |
| **GenSeg** (2024) | 48.5% | 32.4% | 22.8% | 16.2% |
| **SEA** (2024) | 64.2% | 21.5% | 19.4% | 24.1% |
| **FSPGD** (2025) | **66.8%** | 30.2% | 28.5% | 22.1% |
| **PRAF (Ours)** | 63.5% | **48.5%** | **45.1%** | **30.2%** |

> **Insight:** While FSPGD performs slightly better on the homogeneous CNN target (DeepLabV3+), **PRAF dominates on heterogeneous Transformer targets**, achieving a **48.5% mIoU drop on SegFormer** compared to FSPGD's 30.2%. Furthermore, PRAF exhibits the highest resilience against the DDCAT defense model (30.2% mIoU drop).

### 2. Physical World Evaluation

We conducted extensive real-world experiments on two robotic platforms.

| Platform | Scenario | Attack Success Rate (ASR) |
| :--- | :--- | :---: |
| **Intelligent Vehicle** | Static / Low-Speed | **95.4%** |
| **Intelligent Vehicle** | Moving (Motion Blur) | **88.2%** |
| **Humanoid Robot** | Indoor Corridor | **91.2%** |
| **Humanoid Robot** | Outdoor (Dynamic Light) | **85.6%** |

### 3. Robustness Analysis

* **Viewing Angle:** At a steep **60° viewing angle**, standard Adv-Patch ASR drops to 12.5%, whereas PRAF maintains **62.1%**.
* **Defense Evasion:** PRAF effectively bypasses input preprocessing defenses such as **Total Variance Minimization (TVM)** and **Denoising Autoencoders (DAE)**, as its low-frequency perturbations are resistant to smoothing.

![Qualitative Results](docs/qualitative_results.png)
*(Figure 3: Qualitative comparison of physical attack effectiveness)*

