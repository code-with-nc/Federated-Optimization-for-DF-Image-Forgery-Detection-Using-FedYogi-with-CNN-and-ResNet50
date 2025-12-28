# Federated-Optimization-for-DF-Image-Forgery-Detection-Using-FedYogi-with-CNN-and-ResNet50

A privacy-preserving image forgery detection framework using **Federated Learning**. Combines **FedYogi optimizer** with **CNN** and **ResNet50** to handle non-IID forensic data. Models are trained collaboratively without sharing raw images, improving stability and accuracy on **CASIA 1.0** dataset.

---

## Introduction
Digital forensics faces challenges in detecting manipulated images, especially when aggregating sensitive data from multiple sources. Centralized models risk privacy and security breaches.  
**Federated Learning (FL)** provides a solution by aggregating model updates instead of raw data.  
**FedYogi** is an adaptive optimizer that improves convergence and stability under non-IID data.  
This repository implements CNN and ResNet50 architectures combined with FedYogi to detect forged images while preserving data privacy.

---

## CASIA 1.0 Dataset
Download link: [CASIA 1.0 Dataset](https://drive.google.com/file/d/14f3jU2VsxTYopgSE1Vvv4hMlFvpAKHUY/view)

- **Authentic Images:** 800 images (8 categories, 100 images each)
  - Format: `Au_<category>_xxxx.jpg`
  - Categories: `ani` (animal), `arc` (architecture), `art`, `cha` (characters), `nat` (nature), `pla` (plants), `sec`, `txt` (texture)

- **Tampered Images:**  
  - **Spliced Image**
    ```
    Sp_D_CND_A_pla0005_pla0023_0281.jpg
    ```
    - Sp: Splicing
    - D: Different (tampered region copied from a different image)
    - `pla0005`: source image, `pla0023`: target image, `0281`: tampered ID

  - **Copy-Move Image**
    ```
    Sp_S_CND_A_pla0016_pla0016_0196.jpg
    ```
    - S: Same (tampered region copied from the same image)

---

## Folder Structure
Federated_Optimization_for_DF/
├── Model/ # CNN and ResNet50 model architectures
├── Model_Image/ # Sample images for testing
├── Results/ # Evaluation metrics and visualizations
├── 596.pdf # Paper / research PDF
└── README.md


## Folder Structure
Federated_Optimization_for_DF/
├── Model/ # CNN and ResNet50 model architectures
├── Model_Image/ # Sample images for testing
├── Results/ # Evaluation metrics and visualizations
├── 596.pdf # Paper / research PDF
└── README.md


---

## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/Federated-Optimization-for-DF.git
cd Federated-Optimization-for-DF

#Create a Python virtual environment (optional but recommended):
python3 -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows

Install dependencies:
pip install -r requirements.txt

python train_federated.py

---
## Methodology

Data Split: CASIA 1.0 dataset split among multiple clients to simulate federated setup.

Local Training: Each client trains CNN or ResNet50 on its local data.

Model Aggregation: FedYogi optimizer aggregates model updates at the central server.

Evaluation Metrics: Accuracy, Precision, Recall, F1-Score.

Privacy Preservation: Raw images never leave client machines.

---

## Methodology
1. **Data Split:** CASIA 1.0 dataset split among multiple clients to simulate federated setup.
2. **Local Training:** Each client trains CNN or ResNet50 on its local data.
3. **Model Aggregation:** FedYogi optimizer aggregates model updates at the central server.
4. **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score to assess performance.
5. **Privacy Preservation:** Raw images never leave client machines.

**FedYogi Optimizer Equation:**
mt = β1*mt-1 + (1-β1)*gt
vt = vt-1 - (1-β2) * sign(vt-1 - g^2_t)
wt+1 = wt - η * mt / (√vt + ε)

- `gt`: aggregated gradient from clients  
- `β1, β2`: momentum coefficients  
- `η`: learning rate  
- `ε`: numerical stability constant

---

## Experiments and Results
- **Centralized Learning:** CNN (57.27% acc), ResNet50 (58.43% acc)
- **Federated Learning + FedYogi:**  
  - CNN (peak ~53.49% acc)  
  - ResNet50 (peak ~55.52% acc)
    
- **Observation:** FedYogi improves stability and convergence under non-IID client data, maintaining privacy while achieving competitive accuracy.

---

## Citation
If you use the dataset or methods in this repository, please cite:

**CASIA dataset**
```bibtex
@inproceedings{Dong2013,
  doi = {10.1109/chinasip.2013.6625374},
  url = {https://doi.org/10.1109/chinasip.2013.6625374},
  year = {2013},
  month = jul,
  publisher = {{IEEE}},
  author = {Jing Dong and Wei Wang and Tieniu Tan},
  title = {{CASIA} Image Tampering Detection Evaluation Database},
  booktitle = {2013 {IEEE} China Summit and International Conference on Signal and Information Processing}
}
CASIA groundtruth dataset

@article{pham2019hybrid,
  title={Hybrid Image-Retrieval Method for Image-Splicing Validation},
  author={Pham, Nam Thanh and Lee, Jong-Weon and Kwon, Goo-Rak and Park, Chun-Su},
  journal={Symmetry},
  volume={11},
  number={1},
  pages={83},
  year={2019},
  publisher={MDPI}
}
License
MIT License (or as per your institution/project requirement)


