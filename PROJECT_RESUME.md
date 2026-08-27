# WasmBoost (Distributed AutoML Platform) - Project Portfolio Summary

## Project Overview
**WasmBoost** is an enterprise-grade, full-stack distributed automated machine learning (AutoML) platform that combines **zero-knowledge in-browser WebAssembly preprocessing** with a **distributed asynchronous backend training engine**, **Bayesian hyperparameter optimization (Optuna)**, **game-theoretic explainability (TreeSHAP)**, and **secure artifact serialization (ONNX & Skops)**. The application handles end-to-end data science workflows with an emphasis on data privacy, user experience, automated decision-making, and production readiness.

**Live Deployment:**
- **Frontend (Vercel):** [https://ml-dashboard-livid-pi.vercel.app](https://ml-dashboard-livid-pi.vercel.app)
- **Backend (Render):** [https://ml-dashboard-vqs0.onrender.com](https://ml-dashboard-vqs0.onrender.com) *(Currently hosted on Render; soon deploying on Microsoft Azure)*
- **GitHub Repository:** [https://github.com/Shambhujadhav4/ML_Dashboard](https://github.com/Shambhujadhav4/ML_Dashboard)

---

## Technology Stack

### Frontend & Edge Compute
- **Framework**: Next.js 15 (React 19, TypeScript, App Router)
- **Edge Runtime**: Pyodide (WebAssembly) running in dedicated Web Workers for client-side Pandas and Scikit-Learn execution
- **Styling**: Custom CSS with sleek dark glassmorphic theme and interactive 3D micro-tilt effects
- **Visualization**: Plotly.js for interactive exploratory data analysis and model evaluation charts
- **State Management**: React Hooks (`useState`, `useEffect`, `useMemo`, `useCallback`) & `localStorage` session caching
- **Real-Time Communication**: WebSocket client for live Celery training telemetry & progress streaming
- **Deployment**: Vercel

### Backend & Machine Learning Engine
- **Framework**: FastAPI (Python 3.13), Uvicorn, Gunicorn
- **ML Core**: Scikit-Learn, XGBoost, LightGBM, CatBoost, Pandas, NumPy, SciPy, Statsmodels
- **Hyperparameter Optimization**: Optuna (Bayesian optimization with median trial pruning)
- **Model Explainability**: TreeSHAP (game-theoretic feature attribution & summary plots)
- **Model Serialization**: Skops (AST allowlist-validated Python models) & ONNX / `skl2onnx` (cross-platform inference)
- **Asynchronous Task Queue**: Celery 5 with Redis 7 broker & result backend
- **WebSocket Gateway**: FastAPI native WebSockets with pub/sub progress streaming
- **Cloud & Deployment**: Currently on Render; migrating to **Microsoft Azure** (Azure Linux VM / Azure Container Apps)

### DevOps & Infrastructure
- **Containerization**: Multi-stage Dockerfiles for backend and frontend
- **Orchestration**: Unified Docker Compose (Redis, FastAPI Backend, Celery Worker, Next.js Frontend, Nginx)
- **Reverse Proxy**: Nginx with 50MB upload limits, gzip compression, and WebSocket connection upgrade routing
- **CORS Middleware**: Regex-based origin matching (`allow_origin_regex`) for multi-environment Vercel and Azure support
- **Version Control**: Git/GitHub with strict `.gitignore` protection

---

## Key Features

### 1. Zero-Knowledge Edge Preprocessing (WebAssembly / Pyodide)
- In-browser execution of Pandas, NumPy, and Scikit-Learn data cleaning pipelines.
- Data imputation, categorical encoding, feature scaling, and outlier filtering execute locally in client memory without transmitting un-sanitized raw datasets over the network.

### 2. Automated Dataset Ingestion & Profiling
- Drag-and-drop CSV/TSV file upload with automatic delimiter detection and encoding resolution.
- Automatic task detection (Binary Classification, Multiclass Classification, or Regression).
- Data profiling: missing value heatmap analysis, feature distribution summaries, and smart target column recommendations.

### 3. Intelligent Dual-Metric Model Recommendation Engine
- Automated candidate model benchmarking:
  - **Classification:** Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost), Logistic Regression.
  - **Regression:** Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso.
- Transparent dual-metric evaluation (Weighted F1 + Accuracy for classification; R² + RMSE for regression) with Stratified K-Fold cross-validation.

### 4. Distributed Asynchronous Training (Celery + Redis + WebSockets)
- Heavy ML training jobs are offloaded to background Celery workers.
- Real-time training progress, loss curves, and epoch logs streamed live over WebSockets (`/api/train/ws/{task_id}`).
- Graceful in-process execution fallback when Redis is absent.

### 5. Bayesian Hyperparameter Optimization & Explainability
- Bayesian Optimization via **Optuna** with trial pruning for automated parameter search.
- Game-theoretic feature interpretability via **TreeSHAP**, generating interactive SHAP summary plots.

### 6. Secure Model Artifact Serialization
- **ONNX (`.onnx`):** Cross-platform export for microsecond inference in C++, Go, Node.js, and Python.
- **Skops (`.skops`):** AST allowlist-validated serialization immune to Arbitrary Code Execution (ACE) security vulnerabilities inherent to legacy `.pkl` pickle files.

### 7. Interactive Visualizations & Project Sessions
- Plotly charts: Missing value heatmaps, correlation matrices, distribution histograms, box plots, scatter plots, confusion matrices, ROC/PR curves, and Optuna optimization histories.
- Persistent multi-project session management via synchronized `localStorage`.

---

## Problem Solving & Technical Decisions

### 1. In-Browser Privacy & Edge Computing (Pyodide WASM)
**Problem**: Uploading sensitive, raw tabular datasets to external servers for basic exploratory data analysis creates privacy risks and server bandwidth bottlenecks.  
**Solution**: Integrated Pyodide (Python compiled to WebAssembly) running inside dedicated Web Workers, enabling instant client-side data cleaning and feature transformations.  
**Impact**: Guaranteed zero-knowledge data privacy for preliminary transformations and 0 ms network latency for local data profiling.

### 2. Non-Blocking Distributed Asynchronous Training
**Problem**: Training complex gradient boosting models and running Optuna tuning caused HTTP request timeouts and blocked FastAPI server worker threads.  
**Solution**: Architected a Celery task queue with a Redis message broker and WebSocket pub/sub telemetry streaming.  
**Impact**: FastAPI handles thousands of concurrent requests while heavy compute tasks execute reliably in background workers with live user progress updates.

### 3. Eliminating Arbitrary Code Execution (ACE) in Model Serialization
**Problem**: Traditional Python pickle (`.pkl`) files can execute arbitrary shell commands when deserialized, posing severe security vulnerabilities.  
**Solution**: Implemented `.skops` secure serialization using AST allowlist validation alongside portable `.onnx` exports.  
**Impact**: Safe production deployment and distribution of trained model artifacts without ACE vulnerability risks.

### 4. Dual-Metric Recommendation Transparency
**Problem**: Evaluating models on a single metric (e.g. Accuracy or F1 alone) can mislead users on imbalanced datasets.  
**Solution**: Designed a dual-metric benchmarking system displaying both Weighted F1 and Accuracy (or R² and RMSE), complemented by Stratified K-Fold cross-validation.  
**Impact**: Transparent trade-off visibility for users across classification and regression tasks.

### 5. Multi-Environment CORS & Cloud Deployment (Vercel, Render, Azure)
**Problem**: Dynamic preview domains on Vercel caused cross-origin resource sharing (CORS) 403 errors when connecting to the backend.  
**Solution**: Implemented regex-based CORS origin matching (`allow_origin_regex=r"https://.*\.vercel\.app"`) with safe origin parsing and Nginx reverse proxy routing.  
**Impact**: Flawless cross-cloud communication across Vercel frontend, Render backend, and future Microsoft Azure infrastructure.

---

## Skills Demonstrated

### Full-Stack & Edge Computing
- Next.js 15, React 19, TypeScript, App Router, WebAssembly (Pyodide), Web Workers
- FastAPI, Python 3.13, Uvicorn, Gunicorn, REST API Design, WebSockets

### Machine Learning & Data Science
- Scikit-Learn, XGBoost, LightGBM, CatBoost, Optuna (Bayesian Optimization), TreeSHAP
- Model benchmarking, Stratified Cross-Validation, Feature Engineering, ONNX, Skops

### DevOps, Cloud & Distributed Systems
- Celery 5, Redis 7, Distributed Task Queues, Pub/Sub Telemetry
- Docker, Docker Compose, Nginx Reverse Proxy, Vercel, Render, Microsoft Azure (Azure VM / NSG)

---

## How to Present on Resume

### Short Version (1-2 lines)
"Built WasmBoost, a full-stack distributed AutoML platform (FastAPI + Next.js 15) featuring zero-knowledge WebAssembly edge preprocessing, asynchronous Celery/Redis training with live WebSocket telemetry, Optuna Bayesian tuning, and secure ONNX/Skops artifact serialization. Deployed on Vercel, Render, and prepared for Microsoft Azure."

### Medium Version (3-4 bullets)
- Architected WasmBoost, a full-stack distributed AutoML web application (FastAPI + Next.js 15) enabling browser-native zero-knowledge data preprocessing via Pyodide WebAssembly (WASM) and multi-algorithm model benchmarking.
- Built distributed asynchronous training pipeline using Celery and Redis, streaming real-time training telemetry and Optuna Bayesian optimization metrics to frontend over WebSockets.
- Implemented game-theoretic feature interpretability with TreeSHAP and secured model distribution against Arbitrary Code Execution (ACE) via `.skops` AST allowlist validation and cross-platform `.onnx` export.
- Configured production Docker Compose orchestration and Nginx reverse proxy with SSL, managing deployments across Vercel, Render, and Microsoft Azure.

---

## Live Demo Instructions
1. Visit: [https://ml-dashboard-livid-pi.vercel.app](https://ml-dashboard-livid-pi.vercel.app)
2. Upload a CSV dataset (or test with standard tabular datasets).
3. Experience client-side Pyodide WASM data profiling and preprocessing.
4. Review automated dual-metric recommendations (F1-score + Accuracy).
5. Train models asynchronously, track live WebSocket progress, and inspect Optuna tuning & TreeSHAP plots.
6. Download secure `.skops` or high-performance `.onnx` model artifacts.

---

## GitHub Repository
[https://github.com/Shambhujadhav4/ML_Dashboard](https://github.com/Shambhujadhav4/ML_Dashboard)
