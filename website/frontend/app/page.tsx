import Link from "next/link";

const highlights = [
  "Zero-Knowledge Edge Preprocessing: Run Pandas and Scikit-Learn locally in your browser via Pyodide WebAssembly without uploading raw data.",
  "Distributed Asynchronous Training: Offload heavy ML tasks to background Celery workers with live WebSocket progress streaming.",
  "Bayesian Optimization & Explainability: Automated hyperparameter tuning with Optuna and game-theoretic interpretability via TreeSHAP.",
];

const stats = [
  { label: "Latency", value: "0 ms", sub: "Client-side WASM" },
  { label: "Core Algorithms", value: "3", sub: "XGBoost · LightGBM · CatBoost" },
  { label: "Async Engine", value: "∞", sub: "Celery + Redis + WebSockets" },
];

const flow = [
  { step: "Dataset Ingestion", detail: "Local WASM Memory" },
  { step: "Exploratory Data Analysis", detail: "Feature Selection" },
  { step: "Distributed Training", detail: "Optuna Tuning" },
  { step: "Evaluation & Export", detail: "TreeSHAP + Artifacts" },
];

export default function HomePage() {
  return (
    <section className="stack home-page">
      <div className="home-hero panel">
        <div className="home-hero-content">
          <p className="eyebrow">Enterprise AutoML</p>
          <h1>Build and evaluate machine learning pipelines with absolute privacy.</h1>
          <p className="lead">
            DataPilot combines browser-native WebAssembly preprocessing for complete data privacy
            with distributed, asynchronous gradient boosting and TreeSHAP explainability, giving
            you a production-grade ML engine at the edge.
          </p>
          <div className="button-row">
            <Link href="/upload" className="button button-primary">
              Start with upload flow
            </Link>
            <Link href="/exploration" className="button button-secondary">
              Go to exploration
            </Link>
          </div>
        </div>
        <div className="home-stats">
          {stats.map((item) => (
            <div key={item.label} className="home-stat-card">
              <p>{item.label}</p>
              <strong>{item.value}</strong>
              {"sub" in item && <span className="home-stat-sub">{item.sub}</span>}
            </div>
          ))}
        </div>
      </div>

      <div className="home-grid">
        <div className="panel home-panel">
          <p className="eyebrow">Platform Highlights</p>
          <ul className="feature-list">
            {highlights.map((highlight) => (
              <li key={highlight}>{highlight}</li>
            ))}
          </ul>
        </div>

        <div className="panel home-panel">
          <p className="eyebrow">Pipeline Architecture</p>
          <div className="flow-track">
            {flow.map((item, index) => (
              <div key={item.step} className="flow-chip">
                <span>{index + 1}</span>
                <div className="flow-chip-text">
                  <strong>{item.step}</strong>
                  <em>{item.detail}</em>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="panel home-panel">
        <p className="eyebrow">Secure Production Serving</p>
        <h2>From In-Browser Exploration to Cross-Platform Deployment</h2>
        <p className="muted">
          Export trained pipelines directly to ONNX format for microsecond inference in Node.js,
          C++, and Go, or download allow-list validated <code>.skops</code> binaries for secure
          Python analysis. We ensure your models are immune to Arbitrary Code Execution (ACE)
          out of the box.
        </p>
      </div>
    </section>
  );
}
