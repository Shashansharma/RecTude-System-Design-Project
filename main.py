"""
============================================================
  StreamWave — Video Recommendation Engine
  Case Study: Designing a Video Recommendation System
============================================================
 
Implements:
  1. Collaborative Filtering  (user similarity)
  2. Content-Based Filtering  (tag/genre similarity)
  3. Hybrid Recommendation    (weighted combination)
 
Run:
    python recommendation_engine.py
"""
 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
 
 
# ─────────────────────────────────────────────────────────────
#  SAMPLE DATA
# ─────────────────────────────────────────────────────────────
 
# watch_history: {user_id: {video_id: rating (1-5)}}
watch_history = {
    "user_1": {"vid_A": 5, "vid_B": 3, "vid_C": 4},
    "user_2": {"vid_A": 4, "vid_B": 5, "vid_D": 3},
    "user_3": {"vid_B": 4, "vid_C": 5, "vid_E": 2},
    "user_4": {"vid_A": 3, "vid_C": 4, "vid_E": 5},
}
 
# video_tags: {video_id: [tags]}
video_tags = {
    "vid_A": ["action", "thriller", "sci-fi"],
    "vid_B": ["romance", "drama"],
    "vid_C": ["action", "comedy"],
    "vid_D": ["action", "sci-fi", "adventure"],
    "vid_E": ["romance", "comedy", "drama"],
}
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 1 — Build User-Video Matrix
# ─────────────────────────────────────────────────────────────
 
def build_user_video_matrix(history: dict):
    """
    Converts watch history dict into a 2-D numpy matrix.
 
    Returns:
        matrix     : ndarray of shape (n_users, n_videos)
        all_users  : sorted list of user IDs
        all_videos : sorted list of video IDs
    """
    all_videos = sorted({v for u in history.values() for v in u})
    all_users  = sorted(history.keys())
 
    matrix = np.zeros((len(all_users), len(all_videos)))
    for i, user in enumerate(all_users):
        for j, video in enumerate(all_videos):
            matrix[i][j] = history[user].get(video, 0)
 
    return matrix, all_users, all_videos
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 2 — Compute User-User Cosine Similarity
# ─────────────────────────────────────────────────────────────
 
def compute_user_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between every pair of users.
 
    Returns:
        ndarray of shape (n_users, n_users)
    """
    return cosine_similarity(matrix)
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 3 — Collaborative Filtering Recommendation
# ─────────────────────────────────────────────────────────────
 
def recommend_collaborative(target_user: str, history: dict, top_n: int = 3):
    """
    Recommends videos using User-Based Collaborative Filtering.
 
    Algorithm:
      - Build user-video matrix
      - Compute cosine similarity between target user and all others
      - Weighted sum of unseen video ratings across similar users
      - Normalize by total similarity weight
      - Return top N videos by predicted rating
 
    Args:
        target_user : user ID to generate recommendations for
        history     : full watch history dict
        top_n       : number of recommendations to return
 
    Returns:
        List of (video_id, predicted_score) tuples
    """
    matrix, users, videos = build_user_video_matrix(history)
    sim_matrix = compute_user_similarity(matrix)
 
    target_idx = users.index(target_user)
    sim_scores = sim_matrix[target_idx]
 
    video_scores  = defaultdict(float)
    video_sim_sum = defaultdict(float)
 
    for i, user in enumerate(users):
        if user == target_user:
            continue
        similarity = sim_scores[i]
        for j, video in enumerate(videos):
            # Only consider videos the target user has NOT watched
            if video not in history[target_user]:
                video_scores[video]   += similarity * matrix[i][j]
                video_sim_sum[video]  += abs(similarity)
 
    # Normalize
    recommendations = {}
    for video in video_scores:
        if video_sim_sum[video] > 0:
            recommendations[video] = video_scores[video] / video_sim_sum[video]
 
    ranked = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 4 — Content-Based Filtering Recommendation
# ─────────────────────────────────────────────────────────────
 
def tag_vector(video_id: str, all_tags: list) -> np.ndarray:
    """One-hot encodes a video's tags against all_tags vocabulary."""
    return np.array([1 if tag in video_tags.get(video_id, []) else 0
                     for tag in all_tags])
 
 
def recommend_content_based(watched_videos: list, candidate_videos: list, top_n: int = 3):
    """
    Recommends videos based on tag/genre similarity to previously watched content.
 
    Algorithm:
      - Build a user profile vector (mean of watched video tag vectors)
      - Compute cosine similarity between user profile and each candidate
      - Return top N candidates by similarity score
 
    Args:
        watched_videos   : list of video IDs the user has already watched
        candidate_videos : list of video IDs to score
        top_n            : number of recommendations to return
 
    Returns:
        List of (video_id, similarity_score) tuples
    """
    all_tags = sorted({tag for v in video_tags.values() for tag in v})
 
    watched_vecs = [tag_vector(v, all_tags) for v in watched_videos if v in video_tags]
    if not watched_vecs:
        return []
 
    user_profile = np.mean(watched_vecs, axis=0).reshape(1, -1)
 
    scores = {}
    for video in candidate_videos:
        if video in video_tags:
            vec = tag_vector(video, all_tags).reshape(1, -1)
            scores[video] = cosine_similarity(user_profile, vec)[0][0]
 
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
 
 
# ─────────────────────────────────────────────────────────────
#  STEP 5 — Hybrid Recommendation
# ─────────────────────────────────────────────────────────────
 
def hybrid_recommend(user_id: str, history: dict, alpha: float = 0.6, top_n: int = 3):
    """
    Combines Collaborative and Content-Based scores into a single ranking.
 
    Final Score = alpha * collab_score + (1 - alpha) * content_score
 
    Args:
        user_id : target user
        history : full watch history dict
        alpha   : weight for collaborative filtering (0.0 – 1.0)
        top_n   : number of recommendations to return
 
    Returns:
        List of (video_id, hybrid_score) tuples
    """
    collab  = dict(recommend_collaborative(user_id, history, top_n=10))
 
    watched   = list(history.get(user_id, {}).keys())
    all_vids  = list({v for u in history.values() for v in u})
    unwatched = [v for v in all_vids if v not in watched]
 
    content = dict(recommend_content_based(watched, unwatched, top_n=10))
 
    all_candidates = set(collab) | set(content)
    hybrid_scores  = {
        v: alpha * collab.get(v, 0) + (1 - alpha) * content.get(v, 0)
        for v in all_candidates
    }
 
    ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
 
 
# ─────────────────────────────────────────────────────────────
#  MAIN — Print all outputs (take screenshots of each section)
# ─────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    SEP = "=" * 55
 
    # ── Screenshot 1: User-Video Matrix ─────────────────────
    print(f"\n{SEP}")
    print("  1 — User-Video Matrix")
    print(SEP)
    matrix, users, videos = build_user_video_matrix(watch_history)
    header = f"{'':>10}" + "".join(f"{v:>8}" for v in videos)
    print(header)
    for i, user in enumerate(users):
        row = f"{user:>10}" + "".join(f"{int(matrix[i][j]):>8}" for j in range(len(videos)))
        print(row)
 
    # ── Screenshot 2: Cosine Similarity Matrix ───────────────
    print(f"\n{SEP}")
    print(" 2 — User Cosine Similarity Matrix")
    print(SEP)
    sim = compute_user_similarity(matrix)
    header2 = f"{'':>10}" + "".join(f"{u:>10}" for u in users)
    print(header2)
    for i, user in enumerate(users):
        row = f"{user:>10}" + "".join(f"{sim[i][j]:>10.3f}" for j in range(len(users)))
        print(row)
 
    # ── Screenshot 3: Collaborative Filtering ───────────────
    print(f"\n{SEP}")
    print(" 3— Collaborative Filtering (user_1)")
    print(SEP)
    collab_results = recommend_collaborative("user_1", watch_history)
    for video, score in collab_results:
        print(f"  {video}  →  Predicted Score: {score:.4f}")
 
    # ── Screenshot 4: Content-Based Filtering ───────────────
    print(f"\n{SEP}")
    print("  4 — Content-Based Filtering (user_1)")
    print(SEP)
    watched_by_u1   = list(watch_history["user_1"].keys())
    all_vids        = list({v for u in watch_history.values() for v in u})
    unwatched_by_u1 = [v for v in all_vids if v not in watched_by_u1]
    content_results = recommend_content_based(watched_by_u1, unwatched_by_u1)
    for video, score in content_results:
        print(f"  {video}  →  Similarity Score: {score:.4f}")
 
    # ── Screenshot 5: Hybrid Recommendation ─────────────────
    print(f"\n{SEP}")
    print("  5 — Hybrid Recommendation (user_1)")
    print(SEP)
    hybrid_results = hybrid_recommend("user_1", watch_history)
    for video, score in hybrid_results:
        print(f"  {video}  →  Hybrid Score: {score:.4f}")
 
    print(f"\n{SEP}")
   
    print(SEP)
 