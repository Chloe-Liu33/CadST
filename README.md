
# CadST
**Causality-driven Alignment and Disentanglement in Multi-modal Spatio-Temporal Traffic Prediction**

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()

*Under Review for PAKDD 2026*

</div>

## 📌 Overview

**CadST** is an **end-to-end multi-modal AI framework** engineered to address Out-of-Distribution (OOD) generalization challenges in Intelligent Transportation Systems (ITS). 

In dynamic real-world scenarios, traffic anomalies often coincide with external textual events (e.g., incident reports, weather alerts). However, conventional spatio-temporal models tend to rely on mere statistical co-occurrences between these events and traffic states. This leads to severe performance degradation under distribution shifts, as the actual causal impact is often modulated by latent confounders—such as precise spatial localization and shifting temporal contexts—which the models do not directly observe.

CadST fundamentally shifts this paradigm by explicitly **disentangling robust causal dependencies from spurious environmental correlations**. By systematically aligning multi-modal signals (textual event indicators and spatial-temporal graphs), this framework ensures highly reliable and interpretable zero-shot cross-region forecasting.

## 🚀 Key Features

- **Causality-Driven Disentanglement:** Isolates the true causal impacts of multi-modal events from spurious statistical dependencies, effectively mitigating the influence of unobserved latent confounders.
- **Multi-Modal Representation Alignment:** Seamlessly fuses unstructured external knowledge (textual indicators) with structured spatial-temporal traffic graphs.
- **OOD Generalization Excellence:** Demonstrates state-of-the-art resilience against severe spatial and temporal distribution shifts.
- **Builder-Friendly Architecture:** A modular, end-to-end PyTorch pipeline designed for high engineering agility, rapid experimentation, and scalable deployment.

## ⚙️ Quick Start

### 1. Environment Setup

It is recommended to use a virtual environment.

```bash
git clone [https://github.com/Chloe-Liu33/CadST.git](https://github.com/Chloe-Liu33/CadST.git)
cd CadST
conda create -n cadst python=3.9
conda activate cadst
pip install -r requirements.txt
