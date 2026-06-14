# Video Recommendation System
 
A Python implementation of a hybrid video recommendation engine, paired with a fully working YouTube-style web dashboard for live demonstration. This project was built as part of the System Design final evaluation.
 
---
 
## 1. Project Overview
 
This project simulates how platforms like YouTube recommend videos to users. It combines two recommendation strategies into a single hybrid ranking:
 
- **Collaborative Filtering (CF)** — uses cosine similarity between users' interaction vectors to find users with similar taste, then recommends videos those similar users enjoyed.
- **Content-Based Filtering (CBF)** — uses Jaccard similarity between tag/genre profiles to recommend videos similar to what the user already liked.
- **Hybrid Recommender** — combines CF (60% weight) and CBF (40% weight) into a single ranked list, with a cold-start fallback (randomised popular videos) for new users with no history.
The system also demonstrates several core System Design concepts:
 
- API Gateway pattern (authentication + rate limiting)
- Caching with TTL and cache invalidation on new interactions
- Microservices-style separation of concerns (User, Video, Recommendation, Interaction services)
- A working frontend (`youtube_clone.html`) — a YouTube-style dashboard where every click (like, skip, share, search) calls the same recommendation logic implemented in Python, live, in the browser
### Project Structure
 
```
shashank_studentAttend/
├── main.py              # Python recommendation engine (CF + CBF + Hybrid + API Gateway)
├── youtube_clone.html   # YouTube-style web dashboard (working demo UI)
├── README.md            # This file
└── docs/
    └── VideoRecommendationSystem_Documentation.pdf  # Full project documentation
```
 
---
 
## 2. Setup Instructions
 
1. Ensure Python 3.8 or later is installed:
```bash
   python3 --version
```
 
2. Clone this repository:
```bash
   git clone https://github.com/YOUR_USERNAME/video-recommendation-system.git
   cd video-recommendation-system
```
 
3. (Optional) Create a virtual environment:
```bash
   python3 -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
```
 
4. Install dependencies (see Dependencies section below).
---
 
## 3. Dependencies
 
| Package | Purpose |
|---|---|
| `numpy` | Vector operations for similarity calculations |
| `scikit-learn` | `cosine_similarity` for collaborative filtering |
 
Install with:
```bash
pip install numpy scikit-learn
```
 
No external dependencies are required for the web dashboard (`youtube_clone.html`) — it runs entirely with plain HTML, CSS, and JavaScript in any modern browser.
 
---
 
## 4. Execution Steps
 
### Run the recommendation engine (Python backend)
 
```bash
python3 main.py
```
 
This will:
- Load the sample dataset (`watch_history` and `video_tags`)
- Build the user-video rating matrix
- Compute collaborative filtering recommendations (user similarity via cosine similarity)
- Compute content-based recommendations (tag/genre similarity)
- Combine both into a final hybrid recommendation list
- Print the recommended videos for each sample user to the console
### Run the web dashboard (frontend demo)
 
1. Open `youtube_clone.html` directly in any modern browser (double-click the file, or right-click → Open With → Browser).
2. Use the sidebar or the "Sign in" button to switch between the four demo users (Aarav, Priya, Ravi, Sneha).
3. Browse the home feed — each video card shows a "why recommended" tag (CF / content match / trending).
4. Click any video to open the watch page. Use Like / Not Interested / Share to record live interactions — this invalidates the recommendation cache and instantly recomputes the "Up Next" list, exactly as the backend would.
5. Use the search bar and category chips to filter the feed.
---
 
## 5. Additional Project Details
 
### Algorithms Used
 
| Technique | Formula | Purpose |
|---|---|---|
| Cosine Similarity | `similarity = dot(A,B) / (‖A‖ × ‖B‖)` | Measures similarity between user rating vectors (Collaborative Filtering) |
| Jaccard Similarity | `J(A,B) = \|A ∩ B\| / \|A ∪ B\|` | Measures overlap between video tag sets (Content-Based Filtering) |
| Hybrid Score | `score(v) = 0.6 × CF_score(v) + 0.4 × CBF_score(v)` | Combines both signals into a final ranking |
 
### Key Features Demonstrated
 
- Cold-start handling for new users with no watch history
- Cache invalidation on every new interaction (watch/like/skip/share)
- Authentication and rate-limiting simulation via an API Gateway class
- A complete, working frontend that mirrors the backend's recommendation logic in real time
### Sample Dataset
 
The project ships with a small sample dataset (4 users, 5 videos with genre/tag metadata) so the algorithms can be demonstrated end-to-end without needing a real database connection. See `main.py` for the `watch_history` and `video_tags` dictionaries.
 
### Documentation
 
Full project documentation — including Problem Statement, Proposed Solution, System Architecture, Module Description, Database Design, Technology Stack, Implementation Details, Screenshots, and Future Scope — is available in `docs/VideoRecommendationSystem_Documentation.pdf` (and `.docx`).
 
### GitHub Repository
 
> https://github.com/Shashansharma/RecTude-System-Design-Project
 
*(Replace `YOUR_USERNAME` with your actual GitHub username after uploading this project as a repository.)*
 