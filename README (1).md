# 🎬 StreamWave — Video Recommendation System

> **Case Study:** Designing a scalable, personalized video recommendation system similar to Netflix / YouTube.

---

## 📌 Overview

StreamWave is a global video streaming platform that delivers personalized video content to millions of users across web, mobile, and smart TV applications. This repository contains the **Python implementation** of the recommendation engine designed as part of a System Design case study.

The engine demonstrates three core recommendation techniques:

| Technique | Description |
|-----------|-------------|
| **Collaborative Filtering** | Recommends videos based on similar users' watch patterns |
| **Content-Based Filtering** | Recommends videos with similar tags/genres to what the user watched |
| **Hybrid Recommendation** | Combines both approaches with a tunable weight (α) |

---

## 📁 Repository Structure

```
streamwave-recommendation/
│
├── recommendation_engine.py   # ← Main Python implementation
└── README.md                  # ← This file
```

---

## ⚙️ Requirements

```bash
pip install numpy scikit-learn
```

> Python 3.7+ recommended

---

## 🚀 How to Run

```bash
python recommendation_engine.py
```

You will see **5 clearly labeled output sections** in your terminal — one for each step of the recommendation pipeline.

---

## 🧠 How It Works

### Step 1 — User-Video Matrix
Converts the watch history dictionary into a 2D matrix where:
- **Rows** = Users
- **Columns** = Videos
- **Values** = Ratings (1–5, or 0 if not watched)

```
           vid_A  vid_B  vid_C  vid_D  vid_E
user_1       5      3      4      0      0
user_2       4      5      0      3      0
user_3       0      4      5      0      2
user_4       3      0      4      0      5
```

---

### Step 2 — Cosine Similarity Between Users
Measures how similar users are based on their ratings vector.

```python
sim = cosine_similarity(matrix)
# Higher score = more similar taste
```

---

### Step 3 — Collaborative Filtering
For a target user, finds similar users and predicts scores for **unseen** videos using a weighted average:

```
predicted_score(video) = Σ(similarity_i × rating_i) / Σ(|similarity_i|)
```

---

### Step 4 — Content-Based Filtering
Builds a **user profile** from the average tag vectors of watched videos, then scores unseen videos by cosine similarity to that profile.

```python
user_profile = mean([tag_vector(v) for v in watched_videos])
score(video) = cosine_similarity(user_profile, tag_vector(video))
```

---

### Step 5 — Hybrid Recommendation
Combines both scores with a tunable weight `alpha`:

```
Hybrid Score = α × collab_score + (1 − α) × content_score
```

Default: `alpha = 0.6` (60% collaborative, 40% content-based)

---

## 📊 Sample Output

```
=======================================================
  SCREENSHOT 1 — User-Video Matrix
=======================================================
           vid_A   vid_B   vid_C   vid_D   vid_E
user_1         5       3       4       0       0
user_2         4       5       0       3       0
user_3         0       4       5       0       2
user_4         3       0       4       0       5

=======================================================
  SCREENSHOT 3 — Collaborative Filtering (user_1)
=======================================================
  vid_D  →  Predicted Score: 0.8230
  vid_E  →  Predicted Score: 0.5410

=======================================================
  SCREENSHOT 5 — Hybrid Recommendation (user_1)
=======================================================
  vid_D  →  Hybrid Score: 0.8200
  vid_E  →  Hybrid Score: 0.4870
```

---

## 🏗️ Full System Architecture (High-Level)

```
┌──────────────────────────────────────────┐
│              CLIENT LAYER                │
│    Web App │ Mobile App │ Smart TV       │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│         API GATEWAY / LOAD BALANCER      │
└──┬──────────┬──────────┬──────────┬──────┘
   │          │          │          │
User       Reco       Ranking   Metadata
Activity   Engine     Service   Service
Collector  (this      (LightGBM)(Elastic-
(Kafka)    repo)               search)
   │          │          │          │
┌──▼──────────▼──────────▼──────────▼──────┐
│        CACHING LAYER (Redis)             │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│  PostgreSQL │ Cassandra │ MongoDB        │
└─────────────────┬────────────────────────┘
                  │
┌─────────────────▼────────────────────────┐
│     ML PIPELINE (Spark + Airflow)        │
└──────────────────────────────────────────┘
```

---

## 🗄️ Database Design Summary

| Data | Database | Reason |
|------|----------|--------|
| User Profiles | PostgreSQL | ACID, relational |
| Watch Events | Cassandra | High write throughput |
| Video Metadata | MongoDB | Flexible schema |
| Reco Cache | Redis | Sub-ms latency |
| Search Index | Elasticsearch | Full-text search |

---

## 📈 Scalability Highlights

- **Kafka** for high-throughput event ingestion
- **Redis Cluster** for distributed caching
- **Kubernetes** for horizontal auto-scaling
- **Multi-region** deployment for global low latency
- **Circuit Breaker** pattern for fault tolerance

---

## 🔁 Recommendation Workflow

```
User Activity → Kafka → Feature Engineering
     → Candidate Generation (500 videos)
     → Ranking Model (Top 20)
     → Redis Cache → API Response
     → Feedback Loop → Model Retrain
```

---

## 📄 License

This project is created for educational/academic purposes as part of a System Design case study.

---

## 👤 Author

**StreamWave System Design Case Study**  
Subject: System Design  
Topic: Video Recommendation System  
