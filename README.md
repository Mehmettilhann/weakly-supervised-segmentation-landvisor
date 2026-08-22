# Robust Weakly Supervised Semantic Segmentation Framework

## The Engineering Challenge
In traditional remote sensing applications, semantic segmentation models require densely annotated, pixel-perfect masks. However, acquiring such data is highly expensive and time-consuming. 

This project presents a scalable **Weakly Supervised Deep Learning Pipeline** designed to achieve high-accuracy land cover segmentation using only a fraction of the data: **highly sparse, point-level annotations (simulated from the DeepGlobe Land Cover Classification dataset).**

## System Architecture & ML Pipeline
To overcome the limitations of sparse labels and environmental inconsistencies, the pipeline is engineered with a multi-stage architecture. Each module is designed to naturally address a specific bottleneck in remote sensing computer vision:

### 1. Feature Extraction Baseline (Transfer Learning)
When supervision is extremely limited (e.g., 20 points per image), the model must rely on strong prior knowledge. 
*   **Implementation:** The pipeline utilizes robust encoders (ResNet34) pre-trained on ImageNet. This ensures that the network already understands fundamental spatial features, edges, and textures before it even sees the sparse satellite imagery.

### 2. Optimization under Sparsity (Partial Focal Cross-Entropy)
Standard loss functions fail when 99% of the mask is unannotated. 
*   **Implementation:** A custom `PartialCELoss` module was engineered to compute gradients *strictly* on valid points, ignoring unlabeled pixels (assigned `255`). To prevent the model from ignoring hard-to-predict minority classes (like small buildings) in favor of dominant classes (like forests), a Focal Loss modifier (`gamma=2.0`) is mathematically integrated into the optimization step.

### 3. Environmental Robustness (Histogram Matching)
Satellite imagery suffers from heavy domain shifts due to varying sunlight angles, cloud cover, and sensor qualities.
*   **Implementation:** Before inference, images pass through a Preprocessing module utilizing **Histogram Matching**. This standardizes the color distribution of incoming images against a reference, stabilizing the input space and preventing the model from confusing lighting changes with class changes.

### 4. Consensus & Reliability (Ensemble & TTA)
To deploy models in production, inference must be highly reliable. Single-model predictions can be biased.
*   **Implementation:** 
    *   **Test-Time Amplification (TTA):** The input tensor is fed into the model from multiple perspectives (e.g., flipped views), and the predictions are averaged.
    *   **Ensemble Learning:** The pipeline dynamically averages the logits of two structurally different architectures (U-Net and DeepLabV3) to produce a robust final mask, mitigating the structural biases of individual networks.

### 5. Self-Correction (Semi-Supervised Pseudo-Labeling)
The ultimate goal of the pipeline is to utilize the unannotated `255` pixels.
*   **Implementation:** A Teacher-Student pseudo-labeling mechanism is deployed. Once the model achieves a high confidence threshold (e.g., >90%) on unannotated regions, those predictions are converted into pseudo-ground truths. This allows the model to iteratively expand its own training dataset, transforming a weakly supervised task into a semi-supervised learning loop.

## Codebase Structure (Separation of Concerns)
The repository is modularized to ensure maintainability and production readiness:
*   `src/dataset.py`: Handles data ingestion, point-simulation logic, and spatial augmentations (Albumentations).
*   `src/loss.py`: Pure mathematical implementation of the Partial Focal CE logic.
*   `src/model.py`: Network instantiations, pre-trained weight loading, and ensemble functions.
*   `Technical_Report.ipynb`: The orchestration layer demonstrating the entire pipeline flow, experiments, and visualizations.
*   `requirements.txt`: Environment dependencies.

## Quick Start & Execution Guide (For Reviewers & HR)

This section is designed to help anyone (regardless of deep technical background) run the project smoothly in a few simple steps.

### Step 1: Clone the Repository
Open your terminal (Command Prompt, PowerShell, or macOS Terminal) and clone this project to your local machine:
```bash
git clone https://github.com/Mehmettilhann/weakly-supervised-segmentation-landvisor.git
cd weakly-supervised-segmentation-landvisor
```

### Step 2: Set Up the Environment
To ensure the project runs without interfering with your system, create and activate a virtual environment, then install the required tools.
* **For Windows:**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt
  ```
* **For macOS/Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

### Step 3: Download the Dataset
Since satellite images are very large, they are not included in this repository. 
1. Go to the dataset page: [DeepGlobe Land Cover Classification Challenge](https://www.kaggle.com/datasets/balraj98/deepglobe-land-cover-classification-dataset) *(Note: You may need a free Kaggle account to download).*
2. Download the ZIP file and extract it.
3. Move the extracted `train` and `valid` folders directly into this project's `data/` folder. Your folder structure should look exactly like this:
   ```text
   weakly-supervised-segmentation-landvisor/
   ├── data/
   │   ├── train/  <-- (Images and masks go here)
   │   └── valid/
   ├── src/
   ├── Technical_Report.ipynb
   ...
   ```

### Step 4: Run the Interactive Report
This project is presented as an interactive technical report.
1. Open your code editor (like Visual Studio Code).
2. Open the `Technical_Report.ipynb` file.
3. Make sure your Python environment (the `.venv` you created in Step 2) is selected at the top right of your screen.
4. Click the **"Run All"** button at the top of the notebook. The system will automatically process the data, run the AI models, and generate all visual outputs step by step.

---
*Engineered by Mehmet İlhan for the Machine Learning Engineer Assessment.*