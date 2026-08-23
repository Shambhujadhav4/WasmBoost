"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { ACTIVE_PROJECT_STORAGE_KEY } from "@/lib/project-session";
import { pyodideClient } from "@/lib/pyodide-client";
import { usePyodide } from "@/lib/pyodide-context";
import type { ColumnInfoRow, DatasetInsights, DatasetSummary, WorkflowRecommendation } from "@/lib/types";

import { DataPreviewTable } from "./data-preview-table";

export function UploadForm() {
  const router = useRouter();
  const { isReady: isPyodideReady, statusMessage: pyodideMsg } = usePyodide();
  const [file, setFile] = useState<File | null>(null);
  const [separator, setSeparator] = useState(",");
  const [encoding, setEncoding] = useState("utf-8");
  const [headerRow, setHeaderRow] = useState(0);
  const [status, setStatus] = useState("Select a CSV file to load into client-side WebAssembly (Pyodide).");
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [insights, setInsights] = useState<DatasetInsights | null>(null);
  const [recommendation, setRecommendation] = useState<WorkflowRecommendation | null>(null);
  const [isGeneratingRecommendation, setIsGeneratingRecommendation] = useState(false);
  const [rowsToDisplay, setRowsToDisplay] = useState(10);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isErrorStatus = /failed|unable|cannot|error/i.test(status);

  function formatCount(value: number) {
    return value.toLocaleString();
  }

  function formatMetricValue(value: number | undefined) {
    return typeof value === "number" && Number.isFinite(value)
      ? value.toFixed(4)
      : "N/A";
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setStatus("Choose a file first.");
      return;
    }

    setIsSubmitting(true);
    setStatus("Reading file into browser WebAssembly memory...");

    try {
      const projectId =
        typeof crypto !== "undefined" && crypto.randomUUID
          ? crypto.randomUUID().replace(/-/g, "")
          : `proj_${Date.now()}`;

      const text = await file.text();

      setStatus("Profiling dataset in Python WebAssembly (NumPy + Pandas)...");
      const result = await pyodideClient.loadCsv(
        projectId,
        text,
        file.name,
        separator,
        headerRow
      );

      window.localStorage.setItem(ACTIVE_PROJECT_STORAGE_KEY, projectId);
      setSummary(result.summary);
      setInsights(result.insights);
      setRecommendation(result.recommendations);
      setStatus(
        "⚡ Dataset profiled 100% client-side via Pyodide WebAssembly. Review the summary below, then proceed to Exploration, Preprocessing, or Training."
      );
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Upload and WebAssembly profiling failed.";
      setStatus(message);
      setSummary(null);
      setInsights(null);
      setRecommendation(null);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="panel upload-card">
      <form className="stack" onSubmit={handleSubmit}>
        <div className="field">
          <label htmlFor="file">File</label>
          <input
            id="file"
            type="file"
            accept=".csv,.png,.jpg,.jpeg,.gif,.webp,.bmp"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
          <p className="muted">Supported: CSV and image files. The app will detect the file type automatically.</p>
        </div>

        <div className="field-grid">
          <div className="field">
            <label htmlFor="separator">Separator</label>
            <select
              id="separator"
              value={separator}
              onChange={(event) => setSeparator(event.target.value)}
            >
              <option value=",">Comma</option>
              <option value=";">Semicolon</option>
              <option value="\t">Tab</option>
              <option value="|">Pipe</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="encoding">Encoding</label>
            <select
              id="encoding"
              value={encoding}
              onChange={(event) => setEncoding(event.target.value)}
            >
              <option value="utf-8">utf-8</option>
              <option value="latin-1">latin-1</option>
              <option value="iso-8859-1">iso-8859-1</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="header-row">Header row</label>
            <input
              id="header-row"
              type="number"
              min={0}
              max={10}
              value={headerRow}
              onChange={(event) => setHeaderRow(Number(event.target.value))}
            />
          </div>
        </div>

        <div className={`status${isErrorStatus ? " error" : ""}`}>
          {status}
        </div>
        {isErrorStatus ? (
          <p className="muted">
            Quick check: backend should be running at http://localhost:8000 and frontend API base URL should match.
          </p>
        ) : null}

        <button type="submit" className="button button-primary" disabled={isSubmitting}>
          {isSubmitting ? "Uploading..." : "Create dataset session"}
        </button>
      </form>

      {summary && !insights ? (
        <div className="summary-box">
          <h2>Project created</h2>
          <p className="muted">Project ID: {summary.project_id}</p>
          <p className="muted">File type: {summary.input_kind}</p>
          <p className="muted">
            {summary.rows} rows, {summary.columns} columns, {summary.missing_values} missing values
          </p>
        </div>
      ) : null}

      {insights ? (
        <div className="uploaded-insights">
          <div className="upload-success-banner">
            <span className="success-icon" aria-hidden="true">✓</span>
            <span>
              Loaded <strong>{file?.name ?? "dataset"}</strong> - {formatCount(insights.summary.rows)} rows × {insights.summary.columns} cols
            </span>
          </div>

          <div className="overview-step-pill">Step 1: Quick Dataset Overview</div>
          <div className="overview-step-pill">Detected type: {insights.summary.input_kind}</div>

          <div className="overview-metrics-grid">
            <div className="overview-metric">
              <p>File name</p>
              <strong>{insights.summary.source_filename ?? "Unknown"}</strong>
            </div>
            <div className="overview-metric">
              <p>MIME type</p>
              <strong>{insights.summary.source_mime_type ?? "Unknown"}</strong>
            </div>
            <div className="overview-metric">
              <p>File size</p>
              <strong>{insights.summary.file_size_bytes ? `${formatCount(insights.summary.file_size_bytes)} bytes` : "Unknown"}</strong>
            </div>
            <div className="overview-metric">
              <p>Rows to inspect</p>
              <strong>{formatCount(rowsToDisplay)}</strong>
            </div>
          </div>

          <div className="overview-metrics-grid">
            <div className="overview-metric">
              <p>Rows</p>
              <strong>{formatCount(insights.summary.rows)}</strong>
            </div>
            <div className="overview-metric">
              <p>Columns</p>
              <strong>{insights.summary.columns}</strong>
            </div>
            <div className="overview-metric">
              <p>Missing values</p>
              <strong>{formatCount(insights.summary.missing_values)}</strong>
            </div>
            <div className="overview-metric">
              <p>Duplicate rows</p>
              <strong>{formatCount(insights.summary.duplicate_rows)}</strong>
            </div>
          </div>

          <details className="accordion-card" open>
            <summary>Preview Data</summary>
            <div className="preview-controls">
              <label htmlFor="preview-rows">Rows to display: {rowsToDisplay}</label>
              <input
                id="preview-rows"
                type="range"
                min={1}
                max={Math.max(1, Math.min(100, insights.summary.preview.length))}
                value={Math.min(rowsToDisplay, Math.max(1, insights.summary.preview.length))}
                onChange={(event) => setRowsToDisplay(Number(event.target.value))}
              />
            </div>
            <DataPreviewTable
              columns={insights.summary.column_names}
              rows={insights.summary.preview.slice(0, rowsToDisplay)}
              showIndex
            />
          </details>

          <details className="accordion-card" open>
            <summary>Column Info</summary>
            <DataPreviewTable
              columns={["column", "dtype", "non_null", "null", "unique"]}
              rows={insights.column_info as ColumnInfoRow[]}
              showIndex
            />
          </details>

          <details className="accordion-card">
            <summary>Descriptive Statistics</summary>
            <DataPreviewTable
              columns={Object.keys(insights.descriptive_statistics[0] ?? {})}
              rows={insights.descriptive_statistics}
              showIndex
            />
          </details>

          {isGeneratingRecommendation || recommendation ? (
            <details className="accordion-card" open>
              <summary>Recommended Workflow and Model</summary>
              {isGeneratingRecommendation ? (
                <div className="recommendation-loading">
                  <div className="recommendation-spinner" aria-hidden="true" />
                  <div>
                    <p className="recommendation-loading-title">Generating your recommendation</p>
                    <p className="muted">Analyzing target column, task type, and model benchmarks...</p>
                  </div>
                </div>
              ) : recommendation ? (
                <div className="recommendation-shell">
                  <div className="recommendation-header">
                    <div>
                      <p className="recommendation-label">Suggested task</p>
                      <strong>{recommendation.recommended_task_type ?? "Unknown"}</strong>
                    </div>
                    <div>
                      <p className="recommendation-label">Suggested target</p>
                      <strong>{recommendation.recommended_target_column ?? "Unknown"}</strong>
                    </div>
                    <div>
                      <p className="recommendation-label">Benchmark metric</p>
                      <strong>{recommendation.benchmark_metric ?? "N/A"}</strong>
                    </div>
                  </div>

                  <div className="recommendation-best-model">
                    <p className="recommendation-label">Best model</p>
                    <h3>{recommendation.best_model?.model_name ?? "Not available"}</h3>
                    <p className="muted">
                      {recommendation.best_model
                        ? `Primary score ${recommendation.best_model.mean_score.toFixed(4)} +/- ${recommendation.best_model.std_score.toFixed(4)}`
                        : "A stable top model could not be determined from this dataset."}
                    </p>
                    {recommendation.best_model ? (
                      <div className="recommendation-metric-chips">
                        <span className="recommendation-chip">Accuracy: {formatMetricValue(recommendation.best_model.metric_scores?.accuracy)}</span>
                        <span className="recommendation-chip">F1 (weighted): {formatMetricValue(recommendation.best_model.metric_scores?.f1_weighted)}</span>
                      </div>
                    ) : null}
                  </div>

                  {recommendation.suggested_preprocessing_steps.length ? (
                    <div>
                      <p className="recommendation-subtitle">Suggested workflow</p>
                      <ul className="feature-list">
                        {recommendation.suggested_preprocessing_steps.map((step) => (
                          <li key={step}>{step}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  {recommendation.candidate_models.length ? (
                    <div>
                      <p className="recommendation-subtitle">Top model candidates</p>
                      <ul className="recommendation-model-list">
                        {recommendation.candidate_models.map((candidate) => (
                          <li key={candidate.model_name} className="recommendation-model-item">
                            <div>
                              <strong>{candidate.model_name}</strong>
                              <p className="muted">
                                Score {candidate.mean_score.toFixed(4)} +/- {candidate.std_score.toFixed(4)}
                              </p>
                            </div>
                            <div className="recommendation-metric-chips">
                              <span className="recommendation-chip">Acc {formatMetricValue(candidate.metric_scores?.accuracy)}</span>
                              <span className="recommendation-chip">F1 {formatMetricValue(candidate.metric_scores?.f1_weighted)}</span>
                            </div>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}

                  {recommendation.notes.length ? (
                    <div>
                      <p className="recommendation-subtitle">Notes</p>
                      <ul className="feature-list">
                        {recommendation.notes.map((note) => (
                          <li key={note}>{note}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="recommendation-loading">
                  <p className="muted">Recommendation is not available for this dataset.</p>
                </div>
              )}
            </details>
          ) : null}

          <div className="button-row">
            <button type="button" className="button button-secondary" onClick={() => router.push(`/exploration?projectId=${summary?.project_id ?? insights.project_id}`)}>
              Go to exploration
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
