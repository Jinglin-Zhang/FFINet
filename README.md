# Fine-tuning Feature Interaction Network for Unsupervised Domain Adaptive Low-Light Object Detection (FFINet)

## 📘 Overview

Low-light object detection remains a challenging task because detectors trained on well-lit datasets often experience severe performance degradation under poor illumination. The primary obstacles include the **absence of labeled low-light data** and the **difficulty of transferring knowledge** directly from well-lit domains.

To address these challenges, we introduce the **Fine-tuning Feature Interaction Network (FFINet)** — a novel framework designed for **unsupervised domain adaptation (UDA)** in low-light object detection.  
FFINet integrates illumination-aware augmentation, federated fine-tuning feature interaction, and causal cross-modal alignment to enable effective domain transfer **without requiring labeled low-light images**.

---

## 🧩 Framework Overview

The architecture of **FFINet** is shown below:

![img.png](img.png)

FFINet contains three key components:

### **1. Global–Local Augmentation (GLA)**
A hybrid illumination enhancement strategy combining:

- **Retinex decomposition**  
- **Fractional-order differential masks**

This augmentation extracts both global illumination and fine-grained texture cues, generating more representative features for low-light scenes.

### **2. Federated Fine-tuning Feature Interaction (FFI)**
A federated-learning inspired mechanism to align features between:

- **Source domain (daytime)**
- **Target domain (low-light)**

FFI enables collaborative knowledge transfer while respecting domain discrepancies, promoting robust cross-domain alignment.

### **3. Causal Attention Alignment (CAA)**
A causal reasoning module exploring interactions between:

- **MobileSAM features**
- **ResNet50 features**

CAA enhances feature consistency by modeling causal dependencies across modalities, further reducing domain gaps.

---

## 🧪 Experimental Results

FFINet achieves state-of-the-art performance across multiple UDA low-light detection benchmarks:

| Dataset | mAP@0.5 | mAP@0.5:0.95 |
|:--|:--:|:--:|
| **BDD100K (Night)** | **51.2** | – |
| **SHIFT (Night)** | **52.9** | – |
| **DARK FACE** | **37.9** | – |

FFINet consistently surpasses previous UDA-based low-light detectors, demonstrating its robustness and generalization capability.

---


## 🚀 Usage

### Training

```shell
python train_net.py \
      --num-gpus 4 \
      --config configs/faster_rcnn_R50_bdd100k.yaml\
      OUTPUT_DIR output/bdd100k
```

### Resume the training

```shell
python train_net.py \
      --resume \
      --num-gpus 4 \
      --config configs/faster_rcnn_R50_bdd100k.yaml MODEL.WEIGHTS <your weight>.pth
```

### Evaluation

```shell
python train_net.py \
      --eval-only \
      --config configs/faster_rcnn_R50_bdd100k.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

## ⚙️ Environment Setup

- Python ≥ 3.8  
- PyTorch ≥ 1.10,1 
- CUDA ≥ 11.3
- Dependencies：
  ```bash
  pip install -r requirements.txt
  ```

---



## ✏️ Citation

If you use this code or dataset in your research, please cite the following paper:

> M. Xiong, Q. Zhang, D. Li, W. Wang, Z. Zhang, K. Zhang, C. Liu, D. Chen, J. Zhang.  
> *Fine-tuning Feature Interaction Network for Unsupervised Domain Adaptive Low-Light Object Detection.*  
> **Neurocomputing**, 2025.  
> DOI: [10.1016/j.neucom.2025.131717](https://doi.org/10.1016/j.neucom.2025.131717)
