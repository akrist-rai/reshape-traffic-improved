
## 🔹 What is Heterophily? 

### Homophily (GCN / GAT assumption)

> “Friends are similar”

* Same class neighbors
* Feature smoothing works

### Heterophily (Real-world case)

> “Opposites attract”

* Fraud ↔ normal users
* Malicious ↔ benign nodes
* Traffic bottleneck ↔ free-flow neighbors

📉 Standard GAT **fails** here due to over-smoothing.

---

## 🔹 Core Problem with Normal GAT

GAT aggregates neighbors like:

> h<sub>i</sub><sup>new</sup> = ∑<sub>j ∈ 𝒩(i)</sub> α<sub>ij</sub> h<sub>j</sub>


But under heterophily:

* Neighbors belong to **different classes**
* Aggregation **destroys node identity**

---

## 🔹 How HA-GAT Fixes This

### **1️⃣ Separates Self vs Neighbor Information**

HA-GAT **does not blindly mix** neighbors.


> h<sub>i</sub><sup>new</sup> = λ h<sub>i</sub> + ∑<sub>j</sub> α<sub>ij</sub> h<sub>j</sub>


* Self-node kept dominant
* Prevents feature dilution

---

### **2️⃣ Signed / Directional Attention**

Instead of “important vs not important”, HA-GAT learns:

* **Helpful neighbors**
* **Harmful neighbors**


> α<sub>ij</sub> ∈ [−1, +1]

Negative attention = **repulsion**, not attraction.

---

### **3️⃣ Higher-Order Neighborhood Mixing**

Heterophily often appears at **2-hop or 3-hop** distance.

HA-GAT combines:

* 1-hop (different)
* 2-hop (often similar!)

> h<sub>i</sub><sup>final</sup> = h<sub>i</sub><sup>(1)</sup> + h<sub>i</sub><sup>(2)</sup>



---

### **4️⃣ Feature-wise Attention (Key Upgrade)**

Instead of node-level only:

* Attention applied **per feature channel**
* Some features attract, others repel

---

## 🔹 Architecture Overview

```
Node Features
   ↓
Self-Embedding (Strong)
   ↓
Signed Attention (±)
   ↓
Multi-hop Aggregation
   ↓
Classifier
```

---

## 🔹 HA-GAT vs GAT vs GCN

| Property                  | GCN  | GAT    | HA-GAT |
| ------------------------- | ---- | ------ | ------ |
| Assumes homophily         | ✅    | ✅      | ❌      |
| Handles heterophily       | ❌    | ❌      | ✅      |
| Signed attention          | ❌    | ❌      | ✅      |
| Self-feature preservation | Weak | Medium | Strong |
| Real-world robustness     | Low  | Medium | High   |

---

