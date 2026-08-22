# 🚀 DataPilot - ML Dashboard

![DataPilot Cover](https://via.placeholder.com/1200x400/0f766e/ffffff?text=DataPilot+-+Automated+Machine+Learning+Platform)

A full-stack machine learning web application that enables users to upload datasets, perform automated preprocessing, train models, and get intelligent model recommendations. The application handles end-to-end ML workflows with a focus on user experience and automated decision-making.

---

## 🌟 Live Demo

- **Frontend (Vercel):** [https://ml-dashboard-livid-pi.vercel.app](https://ml-dashboard-livid-pi.vercel.app)
- **Backend (Render):** [https://ml-dashboard-vqs0.onrender.com](https://ml-dashboard-vqs0.onrender.com) *(Currently on Render, migrating to DigitalOcean VPS soon)*

---

## ✨ Key Features

*   📊 **Automated Dataset Analysis:** Upload CSV/TSV files and get automatic task detection (classification vs. regression), data profiling, and target column recommendations.
*   🧠 **Intelligent Model Recommendation:** Automatically benchmarks models (Random Forest, Gradient Boosting, Logistic/Linear Regression) using dual-metric evaluation (F1-weighted + Accuracy, or R²).
*   🛠️ **Interactive Preprocessing Pipeline:** Visually handle missing values, encode categorical data, scale features, and detect/remove outliers before training.
*   ⚙️ **Model Training & Secure Artifacts:** Train models with configurable cross-validation and download secure `.skops` (AST allowlist-validated Python model) or high-performance `.onnx` models for production inference.
*   📈 **Rich Visualizations:** Interactive Plotly charts including missing value heatmaps, feature distributions, confusion matrices, ROC curves, and feature importance.
*   💾 **Session Management:** Secure, `localStorage`-based session persistence for managing multiple data science projects seamlessly.

---

## 💻 Technology Stack

### Frontend
- **Framework:** Next.js 15 (React 19, TypeScript)
- **Styling:** Custom CSS with a sleek Dark Theme
- **Visualization:** Plotly.js
- **Deployment:** Vercel

### Backend
- **Framework:** FastAPI (Python 3.13)
- **ML Core:** Scikit-learn, Pandas, NumPy
- **Deployment:** Currently on Render (Planned migration to DigitalOcean VPS using Docker)

---

## 📂 Project Layout

```text
DataPilot/
├── website/
│   ├── backend/             # FastAPI application (ML Engine & API)
│   │   ├── app/             # Routers, ML Services, and core logic
│   │   └── requirements.txt # Python dependencies
│   ├── frontend/            # Next.js application (User Interface)
│   │   ├── app/             # App router pages (/upload, /training, etc.)
│   │   ├── components/      # Reusable React components
│   │   └── package.json     # Node dependencies
│   ├── nginx/               # Nginx configuration for Docker deployment
│   └── docker-compose.yml   # Production deployment config
├── README.md                # Project documentation
└── PROJECT_RESUME.md        # Detailed portfolio summary
```

---

## 🚀 How to Run Locally

### Option 1: Manual Setup (Development)

#### 1. Start the Backend
```bash
cd website/backend
python -m venv .venv

# Activate the virtual environment:
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies and run
pip install -r requirements.txt
uvicorn app.main:app --reload
```
*Backend runs on `http://localhost:8000`*

#### 2. Start the Frontend
Open a new terminal window:
```bash
cd website/frontend
npm install

# Ensure you have the .env.local file setup:
# echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api" > .env.local

npm run dev
```
*Frontend runs on `http://localhost:3000`*

---

### Option 2: Using Docker

You can run the backend and Nginx reverse proxy using Docker Compose:

```bash
cd website
docker-compose up --build
```
This will expose the API on port `80` (via Nginx) and `8000` (directly from FastAPI). The frontend can still be run manually pointing to this API.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
