"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import {
  fetchFeatureImportance,
  fetchModelEvaluationFigures,
  fetchProjectSnapshot,
  getModelArtifactUrl,
} from "@/lib/api";
import {
  ACTIVE_PROJECT_STORAGE_KEY,
} from "@/lib/project-session";
import type { FeatureImportanceRow, ProjectSnapshot } from "@/lib/types";

import { FeatureImportanceTable } from "./feature-importance-table";
import { MetricsCards } from "./metrics-cards";
import { PlotlyChart } from "./plotly-chart";

export function ResultsOverview({
  stepLabel = "Step 6",
  pageTitle = "Results and visualizations",
  description = "This page loads metrics and feature importance from your backend project session.",
}: {
  stepLabel?: string;
  pageTitle?: string;
  description?: string;
}) {
  const searchParams = useSearchParams();
  const [snapshot, setSnapshot] = useState<ProjectSnapshot | null>(null);
  const [featureImportance, setFeatureImportance] = useState<
    FeatureImportanceRow[] | null
  >(null);
  const [evaluationFigures, setEvaluationFigures] = useState<{
    confusion_matrix: Record<string, unknown> | null;
    roc_curve: Record<string, unknown> | null;
    actual_vs_predicted: Record<string, unknown> | null;
    residuals: Record<string, unknown> | null;
  } | null>(null);
  const [status, setStatus] = useState("Load a trained project to see live results.");
  const [error, setError] = useState<string | null>(null);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const projectIdFromUrl = searchParams.get("projectId");
    const projectIdFromStorage =
      typeof window === "undefined"
        ? null
        : window.localStorage.getItem(ACTIVE_PROJECT_STORAGE_KEY);
    const resolvedProjectId = projectIdFromUrl || projectIdFromStorage;

    setProjectId(resolvedProjectId);

    if (!resolvedProjectId) {
      setSnapshot(null);
      setFeatureImportance(null);
      setEvaluationFigures(null);
      setError(null);
      setStatus("No active trained project yet. Train a model from the dashboard first.");
      setIsLoading(false);
      return;
    }

    if (typeof window !== "undefined") {
      window.localStorage.setItem(ACTIVE_PROJECT_STORAGE_KEY, resolvedProjectId);
    }

    let isCancelled = false;
    setIsLoading(true);
    setError(null);
    setStatus("Loading trained project results...");

    fetchProjectSnapshot(resolvedProjectId)
      .then(async (response) => {
        if (isCancelled) {
          return;
        }

        setSnapshot(response);

        if (!response.model_results) {
          setFeatureImportance(null);
          setEvaluationFigures(null);
          setStatus("This project has no trained model yet.");
          return;
        }

        try {
          const [importance, figures] = await Promise.all([
            fetchFeatureImportance(resolvedProjectId),
            fetchModelEvaluationFigures(resolvedProjectId),
          ]);
          if (!isCancelled) {
            setFeatureImportance(importance);
            setEvaluationFigures({
              confusion_matrix: figures.confusion_matrix,
              roc_curve: figures.roc_curve,
              actual_vs_predicted: figures.actual_vs_predicted,
              residuals: figures.residuals,
            });
            setStatus("Live model results loaded from the FastAPI backend.");
          }
        } catch (importanceError) {
          if (!isCancelled) {
            setFeatureImportance(null);
            setEvaluationFigures(null);
            const message =
              importanceError instanceof Error
                ? importanceError.message
                : "Feature importance could not be loaded.";
            setStatus(message);
          }
        }
      })
      .catch((requestError) => {
        if (isCancelled) {
          return;
        }

        const message =
          requestError instanceof Error
            ? requestError.message
            : "Unable to load results.";
        setSnapshot(null);
        setFeatureImportance(null);
        setEvaluationFigures(null);
        setError(message);
        setStatus("The results page could not load the selected project.");
      })
      .finally(() => {
        if (!isCancelled) {
          setIsLoading(false);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [searchParams]);

  const metrics = useMemo(() => {
    if (!snapshot?.model_results || !snapshot.trained_task_type) {
      return [
        { label: "Status", value: "No model" },
        { label: "Metric 1", value: "N/A" },
        { label: "Metric 2", value: "N/A" },
        { label: "Metric 3", value: "N/A" },
      ];
    }

    const modelResults = snapshot.model_results;

    if (snapshot.trained_task_type === "classification") {
      return [
        { label: "Accuracy", value: metricValue(modelResults.accuracy) },
        { label: "F1 Score", value: metricValue(modelResults.f1_score) },
        { label: "Precision", value: metricValue(modelResults.precision) },
        { label: "Recall", value: metricValue(modelResults.recall) },
      ];
    }

    return [
      { label: "R2 Score", value: metricValue(modelResults.r2_score) },
      { label: "RMSE", value: metricValue(modelResults.rmse) },
      { label: "MAE", value: metricValue(modelResults.mae) },
      { label: "MSE", value: metricValue(modelResults.mse) },
    ];
  }, [snapshot]);

  const cvSummary = useMemo(() => {
    const rawScores = snapshot?.model_results?.cv_scores;
    if (!Array.isArray(rawScores) || !rawScores.length) {
      return null;
    }

    const scores = rawScores
      .map((score) =>
        typeof score === "number" ? score : Number.parseFloat(String(score)),
      )
      .filter((score) => Number.isFinite(score));

    if (!scores.length) {
      return null;
    }

    const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const variance =
      scores.reduce((sum, score) => sum + (score - mean) ** 2, 0) /
      scores.length;

    return {
      mean: mean.toFixed(4),
      std: Math.sqrt(variance).toFixed(4),
      scores,
    };
  }, [snapshot]);

  const classificationReport =
    typeof snapshot?.model_results?.classification_report === "string"
      ? snapshot.model_results.classification_report
      : null;
  const classificationAverage =
    typeof snapshot?.model_results?.metric_average === "string"
      ? snapshot.model_results.metric_average
      : null;
  const skopsArtifactUrl = projectId ? getModelArtifactUrl(projectId, "skops") : null;
  const onnxArtifactUrl = projectId ? getModelArtifactUrl(projectId, "onnx") : null;

  return (
    <section className="stack results-page">
      <div className="section-heading">
        <p className="eyebrow">{stepLabel}</p>
        <h1>{pageTitle}</h1>
        <p className="muted">{description}</p>
      </div>

      <MetricsCards items={metrics} />

      {classificationAverage ? (
        <p className="muted">
          Classification precision, recall, and F1 use {classificationAverage} averaging.
        </p>
      ) : null}

      <div className={`status${error ? " error" : ""}`}>
        {isLoading ? "Loading..." : status}
      </div>

      {!projectId && !isLoading ? (
        <div className="panel empty-panel">
          <h2>No trained project selected</h2>
          <p className="muted">
            Upload a dataset and train a model from the dashboard to populate this page.
          </p>
          <Link href="/upload" className="button button-primary">
            Start with upload
          </Link>
        </div>
      ) : null}

      {projectId && snapshot && !snapshot.model_results ? (
        <div className="panel empty-panel">
          <h2>No trained model yet</h2>
          <p className="muted">
            This project exists, but it has not been trained yet. Use the dashboard training form next.
          </p>
          <Link
            href={`/dashboard?projectId=${projectId}`}
            className="button button-primary"
          >
            Go to dashboard
          </Link>
        </div>
      ) : null}

      {snapshot?.model_results ? (
        <>
          <div className="panel split-panel">
            <div>
              <h2>Training snapshot</h2>
              <ul className="feature-list">
                <li>Task type: {snapshot.trained_task_type ?? "Unknown"}</li>
                <li>Target column: {snapshot.target_column ?? "Unknown"}</li>
                <li>Feature count: {snapshot.feature_columns.length}</li>
              </ul>
            </div>
            <div>
              <h2>Selected features</h2>
              <p className="muted compact-list">
                {snapshot.feature_columns.join(", ")}
              </p>
            </div>
          </div>

          {cvSummary ? (
            <div className="panel split-panel">
              <div>
                <h2>Cross-validation</h2>
                <ul className="feature-list">
                  <li>Mean score: {cvSummary.mean}</li>
                  <li>Std deviation: {cvSummary.std}</li>
                  <li>Folds: {cvSummary.scores.length}</li>
                </ul>
              </div>
              <div>
                <h2>Fold scores</h2>
                <p className="muted compact-list">
                  {cvSummary.scores.map((score) => score.toFixed(4)).join(", ")}
                </p>
              </div>
            </div>
          ) : null}

          {classificationReport ? (
            <div className="panel">
              <h2>Classification report</h2>
              <pre className="report-block">{classificationReport}</pre>
            </div>
          ) : null}

          {snapshot.artifact_available ? (
            <div className="panel">
              <div style={{ marginBottom: "1.25rem" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
                  <h2 style={{ margin: 0 }}>Model Export &amp; Serialization</h2>
                  <span
                    style={{
                      fontSize: "0.75rem",
                      fontWeight: 700,
                      padding: "2px 8px",
                      borderRadius: "999px",
                      background: "rgba(15, 118, 110, 0.15)",
                      color: "var(--accent)",
                      border: "1px solid rgba(15, 118, 110, 0.3)",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                  >
                    Enterprise Security
                  </span>
                </div>
                <p className="muted" style={{ margin: 0 }}>
                  Choose your secure model distribution format. Arbitrary Code Execution (ACE) risks from legacy pickle are fully eliminated.
                </p>
              </div>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                  gap: "16px",
                }}
              >
                {/* skops card */}
                <div
                  style={{
                    padding: "20px",
                    borderRadius: "18px",
                    border: "1px solid var(--line)",
                    background: "var(--surface-strong)",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    gap: "16px",
                  }}
                >
                  <div>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
                      <strong style={{ fontSize: "1.05rem" }}>Secure Python Model (.skops)</strong>
                      <span
                        style={{
                          fontSize: "0.72rem",
                          fontWeight: 600,
                          padding: "2px 8px",
                          borderRadius: "999px",
                          background: "#e6f4ea",
                          color: "#137333",
                        }}
                      >
                        Safe Python AST
                      </span>
                    </div>
                    <p className="muted" style={{ fontSize: "0.88rem", lineHeight: 1.5, margin: 0 }}>
                      Type allowlist-validated AST &amp; JSON archive. Replaces Python pickle with zero arbitrary code execution risk for Python environments and pipelines.
                    </p>
                    {snapshot.skops_artifact_filename && (
                      <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "8px", fontFamily: "monospace" }}>
                        📁 {snapshot.skops_artifact_filename}
                      </p>
                    )}
                  </div>
                  <div>
                    <a
                      href={skopsArtifactUrl ?? "#"}
                      download
                      className="button button-primary"
                      style={{ width: "100%", textAlign: "center" }}
                    >
                      Export as Secure Python Model (.skops)
                    </a>
                  </div>
                </div>

                {/* ONNX card */}
                <div
                  style={{
                    padding: "20px",
                    borderRadius: "18px",
                    border: "1px solid var(--line)",
                    background: "var(--surface-strong)",
                    display: "flex",
                    flexDirection: "column",
                    justifyContent: "space-between",
                    gap: "16px",
                  }}
                >
                  <div>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
                      <strong style={{ fontSize: "1.05rem" }}>Production Inference (.onnx)</strong>
                      <span
                        style={{
                          fontSize: "0.72rem",
                          fontWeight: 600,
                          padding: "2px 8px",
                          borderRadius: "999px",
                          background: "#e8f0fe",
                          color: "#1a73e8",
                        }}
                      >
                        Cross-Platform
                      </span>
                    </div>
                    <p className="muted" style={{ fontSize: "0.88rem", lineHeight: 1.5, margin: 0 }}>
                      Open Neural Network Exchange format for high-throughput production serving with ONNX Runtime across Python, C++, Go, Java, Rust, and edge runtimes.
                    </p>
                    {snapshot.onnx_artifact_filename && (
                      <p style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "8px", fontFamily: "monospace" }}>
                        📁 {snapshot.onnx_artifact_filename}
                      </p>
                    )}
                  </div>
                  <div>
                    <a
                      href={onnxArtifactUrl ?? "#"}
                      download
                      className="button button-secondary"
                      style={{ width: "100%", textAlign: "center" }}
                    >
                      Export for Production Inference (.onnx)
                    </a>
                  </div>
                </div>
              </div>
            </div>
          ) : null}

          {snapshot.trained_task_type === "classification" ? (
            <div className="panel split-panel">
              <div>
                <h2>Confusion matrix</h2>
                <PlotlyChart
                  figure={evaluationFigures?.confusion_matrix ?? null}
                  emptyMessage="Confusion matrix is not available for this model."
                  fontColor="#d9e1ef"
                />
              </div>
              <div>
                <h2>ROC curve</h2>
                <PlotlyChart
                  figure={evaluationFigures?.roc_curve ?? null}
                  emptyMessage="ROC curve is only available for supported binary classifiers."
                  fontColor="#d9e1ef"
                />
              </div>
            </div>
          ) : null}

          {snapshot.trained_task_type === "regression" ? (
            <div className="panel split-panel">
              <div>
                <h2>Actual vs predicted</h2>
                <PlotlyChart
                  figure={evaluationFigures?.actual_vs_predicted ?? null}
                  emptyMessage="Actual vs predicted plot is not available."
                  fontColor="#d9e1ef"
                />
              </div>
              <div>
                <h2>Residual analysis</h2>
                <PlotlyChart
                  figure={evaluationFigures?.residuals ?? null}
                  emptyMessage="Residual plot is not available."
                  fontColor="#d9e1ef"
                />
              </div>
            </div>
          ) : null}

          <div className="panel">
            <h2>Feature importance</h2>
            <p className="muted">
              Available for tree-based models and linear models with coefficients.
            </p>
            <FeatureImportanceTable rows={featureImportance ?? []} />
          </div>
        </>
      ) : null}
    </section>
  );
}

function metricValue(value: unknown) {
  if (typeof value === "number") {
    return value.toFixed(4);
  }

  const parsed = Number.parseFloat(String(value));
  if (Number.isFinite(parsed)) {
    return parsed.toFixed(4);
  }

  return "N/A";
}
