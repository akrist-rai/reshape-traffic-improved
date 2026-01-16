
* RTX 3050 Laptop
* **4 GB VRAM**
* **HA-GAT instead of GAT**
* **Multi-day training allowed**
* Ubuntu / WSL2 / Colab compatible


---

# 🚦 Spatio-Temporal Traffic Prediction

**(ST-Mamba + HA-GAT | Low-VRAM Training Setup)**

This project implements a **spatio-temporal traffic forecasting system** using:

* **ST-Mamba** for temporal modeling
* **HA-GAT (Heterophily-Aware GAT)** for spatial graph learning

The project is optimized to **run on low-VRAM GPUs (RTX 3050 – 4GB)** using:

* Small batch sizes
* Gradient accumulation
* Mixed precision (FP16)
* Multi-day training

---

## 🖥️ Target Hardware

| Component | Spec            |
| --------- | --------------- |
| GPU       | RTX 3050 Laptop |
| VRAM      | 4 GB            |
| RAM       | 8–16 GB         |
| OS        | Ubuntu / WSL2   |
| CUDA      | 12.x            |
| PyTorch   | CUDA-enabled    |

---

## 📁 Project Structure

```text
project/
│
├── models/
│   ├── st_mamba.py          # Temporal model
│   ├── ha_gat.py            # HA-GAT implementation
│   └── full_model.py        # Combined ST-Mamba + HA-GAT
│
├── datasets/
│   ├── traffic_dataset.py   # Dataset loader
│   └── preprocess.py        # Graph + time preprocessing
│
├── train.py                 # MAIN training file (run this)
├── eval.py                  # Evaluation script
│
├── configs/
│   └── low_vram.yaml        # RTX 3050 safe config
│
├── utils/
│   ├── metrics.py
│   ├── seed.py
│   └── checkpoint.py
│
├── checkpoints/
│   └── latest.pt
│
├── requirements.txt
└── README.md
```

---

## ✅ TO-DO LIST (IN ORDER)

### 🔹 Phase 1 — Environment Setup

* [ ] Install NVIDIA drivers on Windows
* [ ] Enable WSL2 + Ubuntu
* [ ] Install CUDA PyTorch inside WSL
* [ ] Verify GPU visibility (`nvidia-smi`)

---

### 🔹 Phase 2 — Model Preparation

* [ ] Replace **GAT → HA-GAT**
* [ ] Limit HA-GAT attention heads (≤ 2)
* [ ] Use sparse adjacency (NO dense N×N attention)
* [ ] Cap neighbors per node (≤ 15)

---

### 🔹 Phase 3 — Low-VRAM Training Configuration

* [ ] Reduce batch size to **1**
* [ ] Enable **FP16 (AMP)**
* [ ] Enable gradient accumulation
* [ ] Reduce temporal window if needed
* [ ] Enable gradient clipping

---

### 🔹 Phase 4 — Fault-Tolerant Training

* [ ] Enable checkpoint saving every 1k steps
* [ ] Enable resume-from-checkpoint
* [ ] Log training loss & metrics
* [ ] Monitor GPU memory usage

---

### 🔹 Phase 5 — Evaluation

* [ ] Load best checkpoint
* [ ] Run eval.py
* [ ] Export metrics (MAE, RMSE)
* [ ] Save predictions

---

## ⚙️ Low-VRAM Safe Configuration

**configs/low_vram.yaml**

```yaml
batch_size: 1
grad_accum_steps: 32        # effective batch = 32
learning_rate: 0.0001
weight_decay: 0.0001

num_nodes: 300
seq_len: 48
pred_len: 12

ha_gat:
  hidden_dim: 64
  num_heads: 2
  num_layers: 2
  max_neighbors: 15
  attention_dropout: 0.2

training:
  use_amp: true
  grad_clip: 1.0
  checkpoint_interval: 1000
```

---

## ▶️ How to RUN (RTX 3050 – 4GB)

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Start training (MAIN ENTRY POINT)

```bash
python train.py --config configs/low_vram.yaml
```

> ⏱ Training may take **days** — this is expected and correct.

---

## 🧠 Training Strategy (IMPORTANT)

* **Batch size = 1** is intentional
* Large batches are simulated using **gradient accumulation**
* Multi-day training is **stable and correct**
* HA-GAT works well with small batches

---

## 💾 Checkpointing & Resume

Checkpoints are saved in:

```text
checkpoints/latest.pt
```

Resume training:

```bash
python train.py --resume checkpoints/latest.pt
```

---

## 📊 Evaluation

```bash
python eval.py --checkpoint checkpoints/best.pt
```

Metrics:

* MAE
* RMSE
* MAPE

---

## 🚀 When to Use Google Colab

Use Colab **only if**:

* Nodes > 500
* Seq length > 96
* Batch size > 2
* Final large-scale training

Otherwise, **RTX 3050 + patience is enough**.

---

## ⚠️ Common Mistakes (DO NOT DO)

❌ Dense adjacency matrices
❌ Batch size > 2
❌ FP32 training
❌ No checkpointing
❌ High learning rate

---

## ✅ Final Notes

* This setup is **research-grade**
* Training is slow but **correct**
* HA-GAT improves heterophilic graph learning
* Time is traded for memory — intentionally


