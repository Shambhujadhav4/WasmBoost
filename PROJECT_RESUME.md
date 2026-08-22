# ML Dashboard - Project Portfolio Summary

## Project Overview
ML Dashboard is a full-stack machine learning web application that enables users to upload datasets, perform automated preprocessing, train models, and get intelligent model recommendations. The application handles end-to-end ML workflows with a focus on user experience and automated decision-making.

**Live Deployment:**
- Frontend: https://ml-dashboard-livid-pi.vercel.app
- Backend: https://ml-dashboard-vqs0.onrender.com
- GitHub: https://github.com/Shambhujadhav4/ML_Dashboard

---

## Technology Stack

### Frontend
- **Framework**: Next.js 15.5.14 (React 19, TypeScript)
- **Styling**: Custom CSS with dark theme (teal accent #0f766e)
- **Visualization**: Plotly.js for interactive charts
- **State Management**: React Hooks (useState, useEffect, useMemo)
- **API Communication**: Fetch-based REST client with error handling
- **Deployment**: Vercel

### Backend
- **Framework**: FastAPI (Python 3.13)
- **ML Libraries**: scikit-learn for model training, metrics, preprocessing
- **Data Processing**: pandas, numpy
- **CORS Middleware**: FastAPI with regex-based origin matching for multi-environment support
- **Database**: Session-based file storage (CSV datasets)
- **Deployment**: Render

### Architecture
- **Pattern**: Monorepo structure (website/backend + website/frontend)
- **API Style**: RESTful with /api prefix
- **Environment Management**: Environment-based configuration (DATAPILOT_CORS_ORIGINS, NEXT_PUBLIC_API_BASE_URL)
- **Version Control**: Git/GitHub with comprehensive .gitignore

---

## Key Features

### 1. Automated Dataset Analysis
- CSV/TSV file upload with custom encoding/delimiter support
- Automatic task detection (classification vs. regression)
- Data profiling: missing values analysis, feature types, distribution summary
- Smart target column recommendation

### 2. Intelligent Model Recommendation Engine
- Task type detection (binary/multi-class classification, regression)
- Candidate model generation (Logistic Regression, Random Forest, Gradient Boosting for classification; Linear Regression, Ridge, Lasso for regression)
- Dual-metric benchmarking: F1-weighted score + Accuracy (for classification); R² (for regression)
- Stratified cross-validation for robust metric computation
- Per-model metric scores display for transparent comparison

### 3. Preprocessing Pipeline
- **Missing Value Handling**: Drop rows/columns, imputation (mean, median, mode, custom value)
- **Categorical Encoding**: Label encoding, one-hot encoding
- **Feature Scaling**: Standard, MinMax, Robust scalers
- **Outlier Detection**: IQR-based removal/capping, Z-score removal
- **Column Selection**: Drop unnecessary features

### 4. Model Training & Evaluation
- Multi-algorithm training (sklearn models)
- Configurable train/test split with stratification
- Cross-validation support (k-fold)
- Feature importance extraction (for tree-based models)
- Model artifact download capability

### 5. Interactive Visualizations
- **Missing Values**: Heatmap showing null patterns
- **Correlation Analysis**: Heatmap for feature correlations
- **Distributions**: Histograms and density plots
- **Categorical**: Countplots for feature frequencies
- **Relationships**: Scatter plots with optional color coding
- **Model Evaluation**: Confusion matrix, ROC curve, residual plots, actual vs. predicted

### 6. Project Session Management
- localStorage-based session persistence
- Multi-project support with independent state
- Project reset functionality
- Upload history tracking

---

## Technical Architecture & Key Implementations

### Backend Highlights
- **Recommendation Service**: Core ML engine that benchmarks models, recommends preprocessing steps, and evaluates feature importance
- **Dataset Service**: Manages CSV uploads, validation, and persistent storage
- **Preprocessing Service**: Applies transformations while maintaining data versioning
- **Training Service**: Handles model training, cross-validation, and artifact management
- **Visualization Service**: Generates interactive Plotly figures for exploration
- **CORS Configuration**: Regex-based origin matching for automatic Vercel domain support

### Frontend Highlights
- **Upload Form Component**: Drag-drop interface with streaming recommendation generation and loading spinner
- **Data Preview Table**: Interactive column/row viewing with null highlighting
- **Preprocessing Dashboard**: Step-by-step workflow UI with before/after data summaries
- **Training Interface**: Model selection, hyperparameter configuration, cross-validation toggle
- **Results Overview**: Metrics cards, confusion matrix, feature importance charts
- **Project Dashboard**: Multi-project management with session persistence

### API Endpoints (Core)
- POST `/api/upload/file` - File upload & dataset creation
- GET `/api/upload/{projectId}` - Fetch project snapshot
- GET `/api/upload/{projectId}/workflow-recommendation` - Get model recommendations
- POST `/api/preprocess/*` - Apply preprocessing steps
- POST `/api/train` - Train selected model
- GET `/api/visualize/*` - Fetch visualization figures
- GET `/api/health` - Service health check

---

## Deployment Configuration

### Vercel (Frontend)
- Root Directory: `website/frontend`
- Build Command: `npm run build`
- Output Directory: `.next` (auto-managed by Next.js)
- Environment Variables:
  - `NEXT_PUBLIC_API_BASE_URL`: Backend API base URL (https://ml-dashboard-vqs0.onrender.com/api)

### Render (Backend)
- Root Directory: `website/backend`
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
- Environment Variables:
  - `DATAPILOT_CORS_ORIGINS`: Allowed frontend origins (https://ml-dashboard-livid-pi.vercel.app)
  - `DATAPILOT_CORS_ORIGIN_REGEX`: Regex for dynamic Vercel domains (https://.*\.vercel\.app)

### Git Management
- Comprehensive .gitignore protecting:
  - Personal datasets (healthcare_real_time_dataset.csv, diabetes files)
  - Environment secrets (.env, .env.local)
  - Build artifacts (__pycache__, node_modules, .next)
  - OS/editor files (.vscode, .idea, .DS_Store)

---

## Problem Solving & Technical Decisions

### 1. Dual-Metric Recommendation System
**Problem**: Single metric (F1 or accuracy) insufficient for model selection; user questioned why F1 is used.
**Solution**: Implemented dual-metric display showing both F1-weighted and accuracy for every model, making trade-offs transparent.
**Impact**: Users can now make informed decisions based on their specific use case priorities.

### 2. Vercel CORS Issue for Production
**Problem**: Frontend could not reach backend from production deployment; 403 CORS errors.
**Solution**: 
- Implemented regex-based CORS matching in FastAPI (allow_origin_regex)
- Added safe origin parsing that trims whitespace and handles multiple formats
- Backend now auto-allows all *.vercel.app domains by default
**Impact**: Frontend-to-backend communication works seamlessly in production without manual domain registration.

### 3. Next.js Output Directory Misconfiguration
**Problem**: Vercel deployment showing 404 after "successful" build; Next.js routing broken.
**Solution**: Removed explicit outputDirectory overrides from vercel.json files; let Next.js manage .next output automatically.
**Impact**: Clean deployment process with correct routing for all pages.

### 4. Monorepo Build Path Resolution
**Problem**: Vercel couldn't find package.json; attempted cd commands in buildCommand caused path parsing errors.
**Solution**: Used explicit Root Directory setting in Vercel UI (website/frontend) rather than relying on vercel.json paths.
**Impact**: Monorepo structure properly supported across both Vercel and Render.

### 5. ML Model Benchmarking Accuracy
**Problem**: Low accuracy (~36%) on healthcare dataset confused users about model quality.
**Solution**: Analyzed dataset; explained weak predictive signal, class imbalance, and missing values as root causes. User education on data quality importance.
**Impact**: User understanding of ML fundamentals and data-driven decision-making.

---

## UI/UX Enhancements

### Recommendation Panel Redesign
- **Before**: Simple text list
- **After**: 
  - Header grid: Task type, target column, benchmark metric cards
  - Best model card: Highlighted best performer with metric chips
  - Metric badges: Color-coded (blue pills) showing F1/Accuracy/R²
  - Loading spinner: 0.8s rotation animation during recommendation generation
  - Mobile responsive: 1-column layout on <900px width

### Result Visualization
- Interactive Plotly charts with zoom/pan/hover
- Feature importance tables with sorted columns
- Metric comparison cards with visual hierarchy
- Dark theme optimized for extended usage

---

## Performance & Optimization

### Frontend
- Memoized API calls to prevent redundant requests
- localStorage caching of project sessions
- CSS-in-JS with responsive breakpoints
- Client-side error boundaries

### Backend
- Stratified cross-validation for balanced metrics
- Efficient scikit-learn pipelines
- Lazy model loading
- Session-based storage to reduce I/O

---

## Skills Demonstrated

### Full-Stack Development
- Frontend: React/Next.js/TypeScript
- Backend: FastAPI/Python
- API design and REST principles

### Machine Learning
- Model selection and benchmarking
- Cross-validation strategies
- Preprocessing pipeline design
- Feature importance interpretation
- Classification vs. regression task detection

### DevOps & Deployment
- Monorepo architecture
- Environment-based configuration
- CORS troubleshooting
- Git workflow and version control
- Vercel and Render deployment

### Problem Solving
- Debugging production issues (CORS, routing, configuration)
- Implementing regex-based middleware logic
- Safe data parsing and validation
- User-centric feature refinement

### UI/UX Design
- Component-based architecture
- Dark theme design
- Responsive layouts
- Loading states and animations
- Error message clarity

---

## Project Statistics

- **Total Files**: 59 committed to GitHub
- **Code Insertions**: 10,533+ lines
- **Frontend Components**: 7 main components (upload, dashboard, preprocessing, training, results, etc.)
- **API Routes**: 6 router modules (health, upload, preprocess, train, predict, visualize)
- **Models Supported**: 6+ algorithms (Logistic Regression, Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso)
- **Preprocessing Methods**: 5 categories (missing values, encoding, scaling, outliers, column selection)
- **Visualization Types**: 9+ chart types (heatmaps, histograms, scatterplots, confusion matrix, ROC curve, residuals)

---

## What You Learned

1. **Full-stack ML application development** from data upload to model deployment
2. **Monorepo management** with independent frontend/backend deployment targets
3. **Production troubleshooting** and debugging deployment-specific issues
4. **CORS and security** in cross-origin API communication
5. **UX-driven feature development** (metric comparison, loading states, animations)
6. **Data quality analysis** and its impact on model performance
7. **Automated ML workflows** (recommendation engine, preprocessing pipelines)

---

## How to Present on Resume

### Short Version (1-2 lines)
"Built full-stack ML Dashboard web application with FastAPI backend and Next.js frontend, enabling users to upload datasets, automatically preprocess data, train ML models, and receive intelligent model recommendations with dual-metric benchmarking. Deployed on Vercel and Render."

### Medium Version (3-4 bullets)
- Designed and deployed full-stack ML web application (FastAPI + Next.js) handling dataset uploads, automated preprocessing, and intelligent model recommendations using scikit-learn
- Implemented dual-metric recommendation engine (F1-weighted + Accuracy) with stratified cross-validation for robust model benchmarking across classification/regression tasks
- Built interactive preprocessing pipeline and visualization dashboard with 9+ chart types (Plotly) for exploratory data analysis
- Solved production CORS/deployment issues using regex-based origin matching and monorepo configuration optimization, enabling seamless Vercel-to-Render communication

### Long Version (Full description)
See "Project Overview" and "Key Features" sections above.

---

## Live Demo Instructions
1. Visit: https://ml-dashboard-livid-pi.vercel.app
2. Upload a CSV dataset (provided test file: healthcare dataset)
3. Review automatically generated recommendations
4. Apply preprocessing steps interactively
5. Train a model and view results with visualizations
6. Download trained model artifacts

---

## GitHub Repository
https://github.com/Shambhujadhav4/ML_Dashboard

---

## Future Enhancements (Optional for resume)
- Advanced hyperparameter tuning UI
- Model comparison side-by-side
- Time-series support
- Deep learning model integration
- Model explainability (SHAP values)
- Batch prediction API
- Dataset versioning and lineage tracking
