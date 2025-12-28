# 🛍️ StyleMatch AI: Visual-Semantic Fashion Recommender

> **A "Visual Search" engine that recommends clothing based on style similarity, not just text tags.**
> *Powered by OpenAI CLIP, PyTorch, and Streamlit.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenAI CLIP](https://img.shields.io/badge/Model-OpenAI%20CLIP-green)

---

## 📖 Overview

Traditional recommendation systems (Collaborative Filtering) rely heavily on user purchase history. They fail when a new product is launched and has no sales data yet—a challenge known as the **"Cold Start Problem."**

**StyleMatch AI** solves this by using **Computer Vision** to "understand" the product. Instead of matching text tags (e.g., "Blue Shirt"), it encodes the actual visual style of the image (texture, pattern, shape) into a high-dimensional vector space.

This allows the system to recommend items that *look* similar, even if they have never been bought together before.

### 🌟 Key Features
* **Multimodal Search:** Uses OpenAI's **CLIP** (Contrastive Language-Image Pre-training) to understand images semantically.
* **Vector Similarity:** Performs efficient Nearest Neighbor search using **Cosine Similarity** on 512-dimensional vectors.
* **Zero-Shot Capability:** Can recommend new items immediately without needing historical user data.
* **Real-Time Inference:** Optimized architecture to run purely on CPU for cost-effective edge deployment (Streamlit Cloud).

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python-based Web Framework)
* **AI Model:** `openai/clip-vit-base-patch32` (Vision Transformer)
* **Deep Learning:** PyTorch, HuggingFace Transformers
* **Data Processing:** Pandas, NumPy, Pillow
* **Dataset:** H&M Personalized Fashion Recommendations (Resized 256x256)

---

## 📐 System Architecture

The system operates in two distinct phases:

### Phase 1: Offline Ingestion (The "Brain")
1.  **Ingestion:** We load 5,000+ fashion images from the H&M dataset.
2.  **Embedding:** Each image is passed through the CLIP Vision Encoder.
3.  **Vectorization:** The model outputs a 512-dimensional vector (embedding) representing the "style" of the item.
4.  **Storage:** Vectors are normalized and stored in `.npy` files (`embeddings.npy`) for ultra-fast retrieval.

### Phase 2: Online Inference (The App)
1.  **User Input:** The user uploads a photo (e.g., a floral dress).
2.  **Real-Time Encoding:** The app converts the uploaded image into a query vector using CLIP.
3.  **Similarity Search:** It calculates the mathematical distance (dot product) between the query and all 5,000 database vectors.
4.  **Retrieval:** The system returns the Top 5 items with the highest similarity scores.

---

## 🚀 Installation & Local Setup

If you want to run this project on your own machine, follow these steps:

### 1. Clone the Repository
```bash
git clone [https://github.com/Muhammad-Shahan/stylematch-ai.git](https://github.com/Muhammad-Shahan/stylematch-ai.git)
cd stylematch-ai
