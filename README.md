# Weakly Supervised Land Cover Segmentation

##  Project Overview
This repository provides a modular, production-ready deep learning framework for semantic segmentation of remote sensing imagery (DeepGlobe). It directly addresses the bottleneck of dense pixel annotations by combining **highly sparse point-level weak supervision** with advanced **semi-supervised** pipelines, inspired by the Landvisor architecture.

##  Architectural Pillars
Instead of a monolithic script, this project is structured using Separation of Concerns (SoC) to ensure scalability:

1. **Transfer Learning Backbone:** Utilized a pre-trained ResNet34 integrated into both U-Net and DeepLabV3 structures, leveraging strong feature extraction for sparse downstream targets.
2. **Partial Focal Cross-Entropy Loss:** A custom formulation (`src/loss.py`) that strictly isolates backpropagation to labeled coordinates while applying a focal penalty to prioritize hard-to-predict minor classes.
3. **Advanced ML Pipeline Integrations:**
   * **Histogram Matching:** Stabilizes picture style across diverse lighting conditions.
   * **Ensemble Learning:** Integrates predictions from multiple architectures (U-Net + DeepLabV3) to mitigate bias.
   * **Inference Amplification (TTA):** Uses Test-Time Augmentation to improve final mask robustness.
   * **Semi-Supervised (Teacher-Student):** Iteratively expands the supervision signal by generating pseudo-labels on highly confident unannotated pixels.

##  Repository Structure
*   `src/dataset.py`: Data ingestion, point-simulation logic, and Albumentations-based data augmentation.
*   `src/loss.py`: Mathematical implementation of Partial Focal CE Loss.
*   `src/model.py`: Network instantiations and ensemble logic.
*   `Technical_Report.ipynb`: The main orchestration notebook addressing the assessment tasks (Methodology, Experiments, Visualizations).
*   `requirements.txt`: Environment dependencies.

##  Setup & Execution
1.  **Environment Setup:** Create a virtual environment and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Dataset:** Download the DeepGlobe Land Cover Classification dataset and place it within a `data/train` and `data/valid` directory at the project root.
3.  **Run:** Open and execute all cells in `Technical_Report.ipynb` to view the methodology, experimental results, and visual verifications.

---
*Developed by Mehmet İlhan for the Machine Learning Engineer Assessment.*