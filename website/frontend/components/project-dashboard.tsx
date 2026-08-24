"use client";

import { useEffect, useMemo, useRef, useState, type ChangeEvent, type ReactNode } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";

import {
  applyFeatureSelection,
  fetchBoxplotFigure,
  fetchCountplotFigure,
  dropColumns,
  encodeCategorical,
  fetchCorrelationFigure,
  fetchDistributionFigure,
  fetchMutualInformation,
  fetchProjectSnapshot,
  fetchScatterFigure,
  fetchVisualizationMetadata,
  handleMissingValues,
  handleOutliers,
  resetProject,
  scaleFeatures,
  trainModel,
  trainModelWithTelemetry,
} from "@/lib/api";
import {
  CLASSIFICATION_MODELS,
  REGRESSION_MODELS,
} from "@/lib/model-options";
import { ACTIVE_PROJECT_STORAGE_KEY } from "@/lib/project-session";
import { pyodideClient } from "@/lib/pyodide-client";
import type {
  DatasetSummary,
  FeatureRankingItem,
  MutualInformationItem,
  ProjectSnapshot,
  TelemetryEvent,
} from "@/lib/types";

import { DataPreviewTable } from "./data-preview-table";
import { MetricsCards } from "./metrics-cards";
import { PlotlyChart } from "./plotly-chart";

type VizMeta = {
  numeric_columns: string[];
  categorical_columns: string[];
};

type DashboardSection = "prepare" | "train" | "explore";

function syncSelection(current: string[], next: string[]) {
  if (current.length === next.length && current.every((value, index) => value === next[index])) {
    return current;
  }
  return next;
}

export function ProjectDashboard({
  initialSnapshot,
  initialSection = "prepare",
  showSectionTabs = true,
  pageTitle = "Project dashboard",
  stepLabel = "Step 2",
  description = "This page reads a real project snapshot from your FastAPI backend.",
}: {
  initialSnapshot?: ProjectSnapshot;
  initialSection?: DashboardSection;
  showSectionTabs?: boolean;
  pageTitle?: string;
  stepLabel?: string;
  description?: string;
}) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const requestedProjectId = searchParams.get("projectId");
  const [projectId, setProjectId] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<ProjectSnapshot | null>(initialSnapshot ?? null);
  const [vizMeta, setVizMeta] = useState<VizMeta | null>(null);
  const [status, setStatus] = useState("Ready to work with dataset.");
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isMutating, setIsMutating] = useState(false);
  const [activeSection, setActiveSection] = useState<DashboardSection>(initialSection);

  const [dropSelection, setDropSelection] = useState<string[]>([]);
  const [missingStrategy, setMissingStrategy] = useState<
    "drop_rows" | "mean" | "median" | "mode" | "zero" | "custom"
  >("mean");
  const [missingColumns, setMissingColumns] = useState<string[]>([]);
  const [customFillValue, setCustomFillValue] = useState("0");
  const [encodingMethod, setEncodingMethod] = useState<"label" | "onehot">("label");
  const [encodingColumns, setEncodingColumns] = useState<string[]>([]);
  const [scalingMethod, setScalingMethod] = useState<"standard" | "minmax" | "robust">("standard");
  const [scalingColumns, setScalingColumns] = useState<string[]>([]);
  const [outlierMethod, setOutlierMethod] = useState<"iqr_remove" | "iqr_cap" | "zscore_remove">("iqr_remove");
  const [outlierColumns, setOutlierColumns] = useState<string[]>([]);
  const [outlierThreshold, setOutlierThreshold] = useState(1.5);
  const [activePreprocessAction, setActivePreprocessAction] = useState<"drop" | "missing" | "encoding" | "scaling" | "outliers" | "feature_selection">("drop");

  // Phase 4: Feature Selection state
  const [fsMethod, setFsMethod] = useState<"mi" | "rfe">("mi");
  const [fsTargetColumn, setFsTargetColumn] = useState("");
  const [fsNumFeatures, setFsNumFeatures] = useState(5);
  const [fsRfeEstimator, setFsRfeEstimator] = useState("Random Forest");
  const [fsRfeStep, setFsRfeStep] = useState(1);
  const [fsMiScores, setFsMiScores] = useState<MutualInformationItem[]>([]);
  const [fsRankings, setFsRankings] = useState<FeatureRankingItem[]>([]);
  const [isComputingMi, setIsComputingMi] = useState(false);

  const [taskType, setTaskType] = useState<"classification" | "regression">("classification");
  const [selectedModel, setSelectedModel] = useState<string>(CLASSIFICATION_MODELS[0]);
  const [targetColumn, setTargetColumn] = useState("");
  const [featureColumns, setFeatureColumns] = useState<string[]>([]);
  const [testSize, setTestSize] = useState(0.2);
  const [randomState, setRandomState] = useState(42);
  const [runCv, setRunCv] = useState(true);

  // Phase 4: Hyperparameter Tuning state
  const [useAdvancedTuning, setUseAdvancedTuning] = useState(false);
  const [optunaTrials, setOptunaTrials] = useState(15);
  const [optunaPruning, setOptunaPruning] = useState(true);

  const [trainingMessage, setTrainingMessage] = useState("Configure options and click Train Model.");
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainingStage, setTrainingStage] = useState<string>("idle");
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [isTraining, setIsTraining] = useState(false);


  const [selectedChart, setSelectedChart] = useState<"histogram" | "boxplot" | "correlation" | "scatter" | "categorical" | "datatypes">("histogram");
  const [histogramColumn, setHistogramColumn] = useState("");
  const [boxplotColumn, setBoxplotColumn] = useState("");
  const [boxplotGroupBy, setBoxplotGroupBy] = useState("");
  const [scatterXColumn, setScatterXColumn] = useState("");
  const [scatterYColumn, setScatterYColumn] = useState("");
  const [scatterColorColumn, setScatterColorColumn] = useState("");
  const [categoricalColumn, setCategoricalColumn] = useState("");
  const [chartFigure, setChartFigure] = useState<Record<string, unknown> | null>(null);
  const [chartStatus, setChartStatus] = useState("Choose a chart type to load a live visualization.");
  const [chartError, setChartError] = useState<string | null>(null);

  const currentColumns = snapshot?.summary.column_names ?? [];
  const numericColumns = vizMeta?.numeric_columns ?? [];
  const categoricalColumns = vizMeta?.categorical_columns ?? [];
  const isDedicatedPreprocessing = !showSectionTabs && initialSection === "prepare";
  const isDedicatedExploration = !showSectionTabs && initialSection === "explore";
  const isDedicatedTraining = !showSectionTabs && initialSection === "train";
  const isDedicatedRoute = isDedicatedPreprocessing || isDedicatedExploration || isDedicatedTraining;
  const availableModels: readonly string[] =
    taskType === "classification" ? CLASSIFICATION_MODELS : REGRESSION_MODELS;
  const availableFeatures = currentColumns.filter((column) => column !== targetColumn);
  const dataTypeRows = useMemo(
    () =>
      currentColumns.map((column) => ({
        column,
        type: numericColumns.includes(column)
          ? "numeric"
          : categoricalColumns.includes(column)
            ? "categorical"
            : "other",
      })),
    [currentColumns, numericColumns, categoricalColumns],
  );

  useEffect(() => {
    setActiveSection(initialSection);
  }, [initialSection]);

  useEffect(() => {
    const fromUrl = searchParams.get("projectId");
    const fromStorage =
      typeof window === "undefined" ? null : window.localStorage.getItem(ACTIVE_PROJECT_STORAGE_KEY);
    const resolved = fromUrl || fromStorage;
    setProjectId(resolved);

    if (!resolved) {
      setSnapshot(null);
      setVizMeta(null);
      setError(null);
      setStatus("No active dataset yet. Upload a CSV to create one.");
      setIsLoading(false);
      return;
    }

    if (typeof window !== "undefined") {
      window.localStorage.setItem(ACTIVE_PROJECT_STORAGE_KEY, resolved);
    }

    let cancelled = false;
    setIsLoading(true);
    setError(null);
    setStatus("Loading project session...");

    (async () => {
      // 1. Check client-side WebAssembly Pyodide session first
      try {
        const wasmSummary = await pyodideClient.getSnapshot(resolved);
        const wasmMeta = await pyodideClient.getVizMetadata(resolved);
        if (!cancelled && wasmSummary && wasmSummary.columns > 0) {
          setSnapshot({
            summary: wasmSummary,
            target_column: null,
            feature_columns: wasmSummary.column_names,
            model_results: null,
            trained_task_type: null,
            artifact_available: false,
            artifact_filename: null,
          });
          setVizMeta(wasmMeta);
          setStatus("⚡ Active dataset loaded from browser WebAssembly (Pyodide) session.");
          setIsLoading(false);
          return;
        }
      } catch {
        // Fallback to server fetch if not in Pyodide memory
      }

      // 2. Fallback to FastAPI server fetch
      try {
        const [projectSnapshot, metadata] = await Promise.all([
          fetchProjectSnapshot(resolved),
          fetchVisualizationMetadata(resolved),
        ]);
        if (!cancelled) {
          setSnapshot(projectSnapshot);
          setVizMeta(metadata);
          setStatus("Project loaded from the FastAPI backend.");
        }
      } catch (requestError) {
        if (!cancelled) {
          const message = requestError instanceof Error ? requestError.message : "Unable to load the project.";
          setSnapshot(null);
          setVizMeta(null);
          setError(message);
          setStatus("The dashboard could not load the requested project.");
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [searchParams]);

  useEffect(() => {
    const keepExisting = (current: string[]) => current.filter((column) => currentColumns.includes(column));
    setDropSelection((current) => syncSelection(current, keepExisting(current)));
    setMissingColumns((current) => syncSelection(current, keepExisting(current)));
    setEncodingColumns((current) => syncSelection(current, keepExisting(current)));
    setScalingColumns((current) => syncSelection(current, keepExisting(current)));
    setOutlierColumns((current) => syncSelection(current, keepExisting(current)));
    setTargetColumn((current) => (current && currentColumns.includes(current) ? current : currentColumns[0] ?? ""));
  }, [currentColumns]);

  useEffect(() => {
    if (!availableModels.includes(selectedModel)) {
      setSelectedModel(availableModels[0]);
    }
  }, [availableModels, selectedModel]);

  useEffect(() => {
    setFeatureColumns((current) => {
      const preserved = current.filter((column) => availableFeatures.includes(column));
      return preserved.length ? preserved : availableFeatures.slice(0, Math.min(8, availableFeatures.length));
    });
  }, [targetColumn, currentColumns.join("|")]);

  useEffect(() => {
    setHistogramColumn((current) => (current && numericColumns.includes(current) ? current : numericColumns[0] ?? ""));
    setBoxplotColumn((current) => (current && numericColumns.includes(current) ? current : numericColumns[0] ?? ""));
    setScatterXColumn((current) => (current && numericColumns.includes(current) ? current : numericColumns[0] ?? ""));
    setScatterYColumn((current) =>
      current && numericColumns.includes(current)
        ? current
        : numericColumns[Math.min(1, Math.max(0, numericColumns.length - 1))] ?? ""
    );
    setCategoricalColumn((current) => (current && categoricalColumns.includes(current) ? current : categoricalColumns[0] ?? ""));
    setScatterColorColumn((current) => (current && currentColumns.includes(current) ? current : ""));
  }, [currentColumns.join("|"), numericColumns.join("|"), categoricalColumns.join("|")]);

  useEffect(() => {
    if (!projectId || !snapshot) {
      setChartFigure(null);
      setChartError(null);
      setChartStatus("Choose a chart type to load a live visualization.");
      return;
    }

    if (selectedChart === "datatypes") {
      setChartFigure(null);
      setChartError(null);
      setChartStatus("Column data types are inferred from current metadata.");
      return;
    }

    let cancelled = false;
    setChartError(null);
    setChartStatus("Loading chart...");

    const request =
      selectedChart === "histogram" && histogramColumn
        ? fetchDistributionFigure({ projectId, column: histogramColumn })
        : selectedChart === "boxplot" && boxplotColumn
          ? fetchBoxplotFigure({ projectId, column: boxplotColumn })
          : selectedChart === "categorical" && categoricalColumn
            ? fetchCountplotFigure({ projectId, column: categoricalColumn })
        : selectedChart === "correlation"
          ? fetchCorrelationFigure(projectId)
          : scatterXColumn && scatterYColumn
            ? fetchScatterFigure({
                projectId,
                xColumn: scatterXColumn,
                yColumn: scatterYColumn,
                colorColumn: scatterColorColumn || undefined,
              })
            : Promise.resolve(null);

    request
      .then((figure) => {
        if (cancelled) {
          return;
        }
        setChartFigure(figure);
        setChartStatus("Live chart loaded from the FastAPI backend.");
      })
      .catch((requestError) => {
        if (cancelled) {
          return;
        }
        const message = requestError instanceof Error ? requestError.message : "Unable to load the chart.";
        setChartFigure(null);
        setChartError(message);
        setChartStatus("The chart could not be loaded.");
      });

    return () => {
      cancelled = true;
    };
  }, [
    projectId,
    selectedChart,
    histogramColumn,
    boxplotColumn,
    boxplotGroupBy,
    categoricalColumn,
    scatterXColumn,
    scatterYColumn,
    scatterColorColumn,
    snapshot,
  ]);

  const metricItems = useMemo(
    () =>
      snapshot
        ? [
            { label: "Project ID", value: snapshot.summary.project_id.slice(0, 8) },
            { label: "Rows", value: String(snapshot.summary.rows) },
            { label: "Columns", value: String(snapshot.summary.columns) },
            { label: "Missing", value: String(snapshot.summary.missing_values) },
          ]
        : [
            { label: "Dataset", value: "Not loaded" },
            { label: "Rows", value: "N/A" },
            { label: "Columns", value: "N/A" },
            { label: "Missing", value: "N/A" },
          ],
    [snapshot],
  );

  async function runMutation(
    action: () => Promise<ProjectSnapshot | DatasetSummary | { summary: DatasetSummary }>,
    successMessage: string
  ) {
    if (!projectId) {
      return false;
    }
    setIsMutating(true);
    setError(null);
    setStatus("Processing dataset in browser WebAssembly (Pyodide)...");
    try {
      const res = await action();
      let nextSnapshot: ProjectSnapshot;
      if ("summary" in res && "rows" in (res as any).summary) {
        nextSnapshot = {
          summary: (res as any).summary,
          target_column: snapshot?.target_column ?? null,
          feature_columns: snapshot?.feature_columns ?? (res as any).summary.column_names,
          model_results: snapshot?.model_results ?? null,
          trained_task_type: snapshot?.trained_task_type ?? null,
          artifact_available: snapshot?.artifact_available ?? false,
          artifact_filename: snapshot?.artifact_filename ?? null,
        };
      } else if ("rows" in (res as any)) {
        nextSnapshot = {
          summary: res as unknown as DatasetSummary,
          target_column: snapshot?.target_column ?? null,
          feature_columns: snapshot?.feature_columns ?? (res as any).column_names,
          model_results: snapshot?.model_results ?? null,
          trained_task_type: snapshot?.trained_task_type ?? null,
          artifact_available: snapshot?.artifact_available ?? false,
          artifact_filename: snapshot?.artifact_filename ?? null,
        };
      } else {
        nextSnapshot = res as ProjectSnapshot;
      }

      // Sync metadata
      let nextMeta: VizMeta;
      try {
        nextMeta = await pyodideClient.getVizMetadata(projectId);
      } catch {
        nextMeta = await fetchVisualizationMetadata(projectId);
      }

      setSnapshot(nextSnapshot);
      setVizMeta(nextMeta);
      setStatus(`⚡ ${successMessage}`);
      return true;
    } catch (mutationError) {
      const message = mutationError instanceof Error ? mutationError.message : "Request failed unexpectedly.";
      setError(message);
      setStatus("The requested action could not be completed.");
      return false;
    } finally {
      setIsMutating(false);
    }
  }

  async function trainAndNavigateToResults() {
    if (!projectId || !targetColumn || !featureColumns.length) {
      return;
    }

    setIsTraining(true);
    setTrainingProgress(5);
    setTrainingStage("queued");
    setTrainingMessage("Exporting preprocessed dataset from WebAssembly runtime for backend handoff...");
    setTrainingLogs([`[0%] Initializing backend handoff for ${selectedModel}...`]);
    setError(null);

    let preprocessedData: Record<string, unknown>[] | undefined = undefined;
    let datasetCsv: string | undefined = undefined;
    try {
      const exported = await pyodideClient.exportPreprocessedDataset(projectId);
      if (exported && exported.records && exported.records.length > 0) {
        preprocessedData = exported.records;
        datasetCsv = exported.csv_text;
        setTrainingLogs((prev) => [
          ...prev,
          `[5%] Exported ${exported.rows} rows × ${exported.columns.length} columns from Pyodide WebAssembly for Celery queue...`,
        ]);
      }
    } catch (exportErr) {
      console.warn("Pyodide export skipped (using server session if exists):", exportErr);
    }

    try {
      const nextSnapshot = await trainModelWithTelemetry(
        {
          projectId,
          taskType,
          modelName: selectedModel,
          targetColumn,
          featureColumns,
          testSize,
          randomState,
          runCv,
          useHyperparameterTuning: useAdvancedTuning,
          nTrials: optunaTrials,
          pruningEnabled: optunaPruning,
          preprocessedData,
          datasetCsv,
        },
        (event: TelemetryEvent) => {
          setTrainingProgress(event.progress);
          setTrainingStage(event.status);
          setTrainingMessage(event.message);
          setTrainingLogs((prev) => [
            ...prev.slice(-8),
            `[${event.progress}%] ${event.message}`,
          ]);
        },
      );

      let nextMeta: VizMeta;
      try {
        nextMeta = await pyodideClient.getVizMetadata(projectId);
      } catch {
        nextMeta = await fetchVisualizationMetadata(projectId);
      }
      setSnapshot(nextSnapshot);
      setVizMeta(nextMeta);
      setTrainingProgress(100);
      setTrainingStage("completed");
      setTrainingMessage("Model training and export completed successfully! Redirecting...");
      setTimeout(() => {
        router.push(`/results?projectId=${projectId}`);
      }, 900);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Training failed unexpectedly.";
      setError(message);
      setTrainingStage("failed");
      setTrainingMessage(`Training failed: ${message}`);
    } finally {
      setIsTraining(false);
    }
  }


  function getSelectedValues(event: ChangeEvent<HTMLSelectElement>) {
    return Array.from(event.target.selectedOptions, (option) => option.value);
  }

  return (
    <section className="stack">
      {!isDedicatedRoute ? (
        <>
          <div className="section-heading">
            <p className="eyebrow">{stepLabel}</p>
            <h1>{pageTitle}</h1>
            <p className="muted">{description}</p>
          </div>

          <MetricsCards items={metricItems} />
          <div className={`status${error ? " error" : ""}`}>{isLoading ? "Loading..." : status}</div>
        </>
      ) : null}

      {!projectId && !isLoading ? (
        <div className="panel empty-panel">
          <h2>No active project yet</h2>
          <p className="muted">Upload a CSV first, then this dashboard will display the stored project session.</p>
          <Link href="/upload" className="button button-primary">Go to upload</Link>
        </div>
      ) : null}

      {snapshot ? (
        <>
          {!isDedicatedRoute ? (
          <div className="panel split-panel">
            <div>
              <h2>Dataset summary</h2>
              <ul className="feature-list">
                <li>Duplicate rows: {snapshot.summary.duplicate_rows}</li>
                <li>Preprocessing steps: {snapshot.summary.preprocessing_log.length}</li>
                <li>Trained model: {snapshot.model_results ? snapshot.trained_task_type ?? "ready" : "not trained"}</li>
              </ul>
            </div>
            <div>
              <h2>Columns</h2>
              <p className="muted compact-list">{currentColumns.join(", ")}</p>
            </div>
          </div>
          ) : null}

          {!isDedicatedRoute ? (
          <div className="workflow-links" aria-label="Dashboard workflow links">
            <Link href="/dashboard/prepare" className={`workflow-link${activeSection === "prepare" ? " active" : ""}`}>
              Preprocessing
            </Link>
            <Link href="/dashboard/train" className={`workflow-link${activeSection === "train" ? " active" : ""}`}>
              Training
            </Link>
            <Link href="/dashboard/explore" className={`workflow-link${activeSection === "explore" ? " active" : ""}`}>
              Exploration
            </Link>
          </div>
          ) : null}

          {showSectionTabs ? (
          <div className="section-tabs" role="tablist" aria-label="Dashboard workflow sections">
            <button
              type="button"
              role="tab"
              aria-selected={activeSection === "prepare"}
              className={`section-tab${activeSection === "prepare" ? " active" : ""}`}
              onClick={() => setActiveSection("prepare")}
            >
              1. Prepare data
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeSection === "train"}
              className={`section-tab${activeSection === "train" ? " active" : ""}`}
              onClick={() => setActiveSection("train")}
            >
              2. Train model
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeSection === "explore"}
              className={`section-tab${activeSection === "explore" ? " active" : ""}`}
              onClick={() => setActiveSection("explore")}
            >
              3. Explore and preview
            </button>
          </div>
          ) : null}

          {activeSection === "prepare" ? (
            isDedicatedPreprocessing ? (
              <div className="panel stack preprocess-panel">
                <div className="preprocess-tab-strip" role="tablist" aria-label="Preprocessing actions">
                  <button type="button" role="tab" aria-selected={activePreprocessAction === "drop"} className={`preprocess-tab${activePreprocessAction === "drop" ? " active" : ""}`} onClick={() => setActivePreprocessAction("drop")}>🗑️ Drop Columns</button>
                  <button type="button" role="tab" aria-selected={activePreprocessAction === "missing"} className={`preprocess-tab${activePreprocessAction === "missing" ? " active" : ""}`} onClick={() => setActivePreprocessAction("missing")}>🧼 Missing Values</button>
                  <button type="button" role="tab" aria-selected={activePreprocessAction === "encoding"} className={`preprocess-tab${activePreprocessAction === "encoding" ? " active" : ""}`} onClick={() => setActivePreprocessAction("encoding")}>🔤 Encoding</button>
                  <button type="button" role="tab" aria-selected={activePreprocessAction === "scaling"} className={`preprocess-tab${activePreprocessAction === "scaling" ? " active" : ""}`} onClick={() => setActivePreprocessAction("scaling")}>📏 Scaling</button>
                  <button type="button" role="tab" aria-selected={activePreprocessAction === "outliers"} className={`preprocess-tab${activePreprocessAction === "outliers" ? " active" : ""}`} onClick={() => setActivePreprocessAction("outliers")}>📐 Outliers</button>
                  <button type="button" role="tab" aria-selected={activePreprocessAction === "feature_selection"} className={`preprocess-tab${activePreprocessAction === "feature_selection" ? " active" : ""}`} onClick={() => setActivePreprocessAction("feature_selection")}>🎯 Feature Selection</button>
                </div>

                {activePreprocessAction === "drop" ? (
                  <div className="preprocess-action-card">
                    <div className="section-heading section-heading-compact">
                      <p className="eyebrow">Drop Columns</p>
                      <p className="preprocess-helper">Select columns you want to remove from the dataset.</p>
                    </div>
                    <PreprocessCard
                      title="Columns to drop"
                      buttonLabel="Drop Selected Columns"
                      onSubmit={() =>
                        projectId && dropSelection.length
                          ? runMutation(
                              () => pyodideClient.dropColumns(projectId, dropSelection),
                              `Dropped ${dropSelection.length} column(s).`
                            )
                          : Promise.resolve()
                      }
                      disabled={!projectId || !dropSelection.length || isMutating}
                    >
                      <label className="field">
                        <span>Columns to drop</span>
                        <MultiSelectDropdown
                          options={currentColumns}
                          placeholder="Choose columns"
                          selectedValues={dropSelection}
                          onChange={setDropSelection}
                        />
                      </label>
                    </PreprocessCard>
                  </div>
                ) : null}

                {activePreprocessAction === "missing" ? (
                  <div className="preprocess-action-card">
                    <div className="status success-banner">✅ No missing values in the current dataset!</div>
                    <PreprocessCard
                      title="Strategy"
                      buttonLabel="Apply Missing Value Treatment"
                      onSubmit={() =>
                        projectId
                          ? runMutation(
                              () =>
                                pyodideClient.handleMissing(
                                  projectId,
                                  missingStrategy,
                                  missingColumns,
                                  Number(customFillValue) || 0
                                ),
                              `Missing-value strategy "${missingStrategy}" applied successfully.`
                            )
                          : Promise.resolve()
                      }
                      disabled={!projectId || isMutating}
                    >
                      <div className="field-grid field-grid-two">
                        <label className="field">
                          <span>Strategy</span>
                          <select value={missingStrategy} onChange={(event) => setMissingStrategy(event.target.value as typeof missingStrategy)}>
                            <option value="drop_rows">Drop rows with nulls</option>
                            <option value="mean">Fill with mean</option>
                            <option value="median">Fill with median</option>
                            <option value="mode">Fill with mode</option>
                            <option value="zero">Fill with zero</option>
                            <option value="custom">Fill with custom value</option>
                          </select>
                        </label>
                        <label className="field">
                          <span>Apply to columns (empty = all)</span>
                          <MultiSelectDropdown
                            options={currentColumns}
                            placeholder="Choose options"
                            selectedValues={missingColumns}
                            onChange={setMissingColumns}
                          />
                        </label>
                      </div>
                      {missingStrategy === "custom" ? (
                        <label className="field">
                          <span>Custom fill value</span>
                          <input type="text" value={customFillValue} onChange={(event) => setCustomFillValue(event.target.value)} />
                        </label>
                      ) : null}
                    </PreprocessCard>
                  </div>
                ) : null}

                {activePreprocessAction === "encoding" ? (
                  <div className="preprocess-action-card">
                    <div className="status info-banner">Categorical columns detected: {categoricalColumns.join(", ")}</div>
                    <PreprocessCard
                      title="Encoding method"
                      buttonLabel="Apply Encoding"
                      onSubmit={() =>
                        projectId
                          ? runMutation(
                              () =>
                                pyodideClient.encodeCategoricals(
                                  projectId,
                                  encodingMethod,
                                  encodingColumns
                                ),
                              `Encoding method "${encodingMethod}" applied successfully.`
                            )
                          : Promise.resolve()
                      }
                      disabled={!projectId || isMutating}
                    >
                      <div className="field-grid field-grid-two">
                        <label className="field">
                          <span>Encoding method</span>
                          <select value={encodingMethod} onChange={(event) => setEncodingMethod(event.target.value as typeof encodingMethod)}>
                            <option value="label">Label Encoding</option>
                            <option value="onehot">One-Hot Encoding</option>
                          </select>
                        </label>
                        <label className="field">
                          <span>Columns to encode (empty = all categorical)</span>
                          <MultiSelectDropdown
                            options={categoricalColumns}
                            placeholder="Choose options"
                            selectedValues={encodingColumns}
                            onChange={setEncodingColumns}
                          />
                        </label>
                      </div>
                    </PreprocessCard>
                  </div>
                ) : null}

                {activePreprocessAction === "scaling" ? (
                  <div className="preprocess-action-card">
                    <PreprocessCard
                      title="Scaling method"
                      buttonLabel="Apply Scaling"
                      onSubmit={() =>
                        projectId
                          ? runMutation(
                              () =>
                                pyodideClient.scaleNumeric(
                                  projectId,
                                  scalingMethod,
                                  scalingColumns
                                ),
                              `Scaling method "${scalingMethod}" applied successfully.`
                            )
                          : Promise.resolve()
                      }
                      disabled={!projectId || isMutating}
                    >
                      <div className="field-grid field-grid-two">
                        <label className="field">
                          <span>Scaling method</span>
                          <select value={scalingMethod} onChange={(event) => setScalingMethod(event.target.value as typeof scalingMethod)}>
                            <option value="standard">Standard Scaler (Z-score)</option>
                            <option value="minmax">Min-Max Scaler</option>
                            <option value="robust">Robust Scaler</option>
                          </select>
                        </label>
                        <label className="field">
                          <span>Columns to scale (empty = all numeric)</span>
                          <MultiSelectDropdown
                            options={numericColumns}
                            placeholder="Choose options"
                            selectedValues={scalingColumns}
                            onChange={setScalingColumns}
                          />
                        </label>
                      </div>
                    </PreprocessCard>
                  </div>
                ) : null}

                {activePreprocessAction === "outliers" ? (
                  <div className="preprocess-action-card">
                    <PreprocessCard
                      title="Method"
                      buttonLabel="Handle Outliers"
                      onSubmit={() =>
                        projectId
                          ? runMutation(
                              () =>
                                pyodideClient.handleOutliers(
                                  projectId,
                                  outlierMethod,
                                  outlierColumns,
                                  outlierThreshold
                                ),
                              `Outlier method "${outlierMethod}" applied successfully.`
                            )
                          : Promise.resolve()
                      }
                      disabled={!projectId || isMutating}
                    >
                      <div className="field-grid field-grid-three">
                        <label className="field">
                          <span>Method</span>
                          <select value={outlierMethod} onChange={(event) => setOutlierMethod(event.target.value as typeof outlierMethod)}>
                            <option value="iqr_remove">IQR - Remove rows</option>
                            <option value="iqr_cap">IQR - Cap values</option>
                            <option value="zscore_remove">Z-score - Remove rows</option>
                          </select>
                        </label>
                        <label className="field">
                          <span>IQR threshold multiplier</span>
                          <input type="range" min={1} max={3} step={0.1} value={outlierThreshold} onChange={(event) => setOutlierThreshold(Number(event.target.value))} />
                          <span className="range-value">{outlierThreshold.toFixed(2)}</span>
                        </label>
                        <label className="field">
                          <span>Columns (empty = all numeric)</span>
                          <MultiSelectDropdown
                            options={numericColumns}
                            placeholder="Choose options"
                            selectedValues={outlierColumns}
                            onChange={setOutlierColumns}
                          />
                        </label>
                      </div>
                    </PreprocessCard>
                  </div>
                ) : null}

                {activePreprocessAction === "feature_selection" ? (
                  <div className="preprocess-action-card">
                    <div className="section-heading section-heading-compact">
                      <p className="eyebrow">Dual Feature Selection (MI & RFE)</p>
                      <p className="preprocess-helper">
                        Prune redundant or collinear features using non-linear Mutual Information ranking or Recursive Feature Elimination.
                      </p>
                    </div>

                    <div className="feature-selection-method-toggle">
                      <button
                        type="button"
                        className={`fs-toggle-btn ${fsMethod === "mi" ? "active" : ""}`}
                        onClick={() => setFsMethod("mi")}
                      >
                        Mutual Information (MI)
                      </button>
                      <button
                        type="button"
                        className={`fs-toggle-btn ${fsMethod === "rfe" ? "active" : ""}`}
                        onClick={() => setFsMethod("rfe")}
                      >
                        Recursive Feature Elimination (RFE)
                      </button>
                    </div>

                    <div className="field-grid">
                      <label className="field">
                        <span>Target Column (Required)</span>
                        <select
                          value={fsTargetColumn}
                          onChange={(e) => {
                            setFsTargetColumn(e.target.value);
                            setFsMiScores([]);
                          }}
                        >
                          <option value="">Select target column</option>
                          {currentColumns.map((col) => (
                            <option key={col} value={col}>
                              {col}
                            </option>
                          ))}
                        </select>
                      </label>

                      <label className="field">
                        <span>Number of features to keep (k)</span>
                        <input
                          type="range"
                          min={1}
                          max={Math.max(1, currentColumns.length - 1)}
                          step={1}
                          value={Math.min(fsNumFeatures, Math.max(1, currentColumns.length - 1))}
                          onChange={(e) => setFsNumFeatures(Number(e.target.value))}
                        />
                        <span className="range-value">
                          {Math.min(fsNumFeatures, Math.max(1, currentColumns.length - 1))} features
                        </span>
                      </label>

                      {fsMethod === "rfe" && (
                        <>
                          <label className="field">
                            <span>Base Estimator</span>
                            <select
                              value={fsRfeEstimator}
                              onChange={(e) => setFsRfeEstimator(e.target.value)}
                            >
                              <option value="Random Forest">Random Forest</option>
                              <option value="Gradient Boosting">Gradient Boosting</option>
                              <option value="Logistic/Ridge">Linear / Ridge / Logistic</option>
                            </select>
                          </label>
                          <label className="field">
                            <span>Step (Features pruned per iteration)</span>
                            <input
                              type="number"
                              min={1}
                              max={10}
                              value={fsRfeStep}
                              onChange={(e) => setFsRfeStep(Math.max(1, Number(e.target.value)))}
                            />
                          </label>
                        </>
                      )}
                    </div>

                    {fsMethod === "mi" && (
                      <div className="mi-action-row" style={{ marginTop: "1rem", marginBottom: "1rem" }}>
                        <button
                          type="button"
                          className="button button-secondary"
                          disabled={!projectId || !fsTargetColumn || isComputingMi || isMutating}
                          onClick={async () => {
                            if (!projectId || !fsTargetColumn) return;
                            setIsComputingMi(true);
                            setError(null);
                            try {
                              let scores: MutualInformationItem[];
                              try {
                                scores = await pyodideClient.computeMutualInformation(
                                  projectId,
                                  fsTargetColumn,
                                  availableFeatures,
                                  taskType
                                );
                              } catch {
                                const res = await fetchMutualInformation({
                                  projectId,
                                  targetColumn: fsTargetColumn,
                                });
                                scores = res.scores;
                              }
                              setFsMiScores(scores);
                              setStatus(`Computed Mutual Information for ${scores.length} features.`);
                            } catch (err) {
                              const msg = err instanceof Error ? err.message : "Failed to compute MI scores.";
                              setError(msg);
                            } finally {
                              setIsComputingMi(false);
                            }
                          }}
                        >
                          {isComputingMi ? "⏳ Computing MI dependency scores in WebAssembly..." : "🔍 Calculate Mutual Information Scores"}
                        </button>
                      </div>
                    )}

                    {fsMiScores.length > 0 && fsMethod === "mi" && (
                      <div className="mi-scores-table-container">
                        <h4>Mutual Information Dependency Ranking</h4>
                        <div className="mi-scores-list">
                          {fsMiScores.map((item, idx) => {
                            const isSelected = idx < fsNumFeatures;
                            return (
                              <div
                                key={item.feature}
                                className={`mi-score-card ${isSelected ? "selected" : "pruned"}`}
                              >
                                <div className="mi-card-header">
                                  <span className="mi-rank">#{idx + 1}</span>
                                  <strong className="mi-feature-name">{item.feature}</strong>
                                  <span className="mi-score-value">MI: {item.score.toFixed(4)}</span>
                                  <span className={`mi-badge ${isSelected ? "keep" : "drop"}`}>
                                    {isSelected ? "Keep" : "Prune"}
                                  </span>
                                </div>
                                <div className="mi-bar-track">
                                  <div
                                    className="mi-bar-fill"
                                    style={{ width: `${Math.min(100, item.normalized_score * 100)}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    <div className="button-row" style={{ marginTop: "1.25rem" }}>
                      <button
                        type="button"
                        className="button button-primary"
                        disabled={!projectId || !fsTargetColumn || isMutating || isComputingMi}
                        onClick={() => {
                          if (!projectId || !fsTargetColumn) return;
                          return runMutation(
                            async () => {
                              try {
                                const wasmRes = await pyodideClient.applyFeatureSelection(
                                  projectId,
                                  fsMethod,
                                  fsTargetColumn,
                                  fsNumFeatures,
                                  availableFeatures,
                                  fsRfeEstimator,
                                  fsRfeStep
                                );
                                if (wasmRes.rankings) {
                                  setFsRankings(wasmRes.rankings);
                                }
                                return wasmRes.summary;
                              } catch {
                                const res = await applyFeatureSelection({
                                  projectId,
                                  method: fsMethod,
                                  targetColumn: fsTargetColumn,
                                  nFeaturesToSelect: fsNumFeatures,
                                  rfeEstimator: fsRfeEstimator,
                                  step: fsRfeStep,
                                });
                                if (res.rankings) {
                                  setFsRankings(res.rankings);
                                }
                                return res.snapshot ?? (await fetchProjectSnapshot(projectId));
                              }
                            },
                            `Applied ${fsMethod.toUpperCase()} feature selection: kept ${fsNumFeatures} top features.`,
                          );
                        }}
                      >
                        {isMutating
                          ? "⏳ Pruning features in WebAssembly..."
                          : fsMethod === "mi"
                          ? `✂️ Prune to Top ${Math.min(fsNumFeatures, Math.max(1, currentColumns.length - 1))} Features (MI)`
                          : `✂️ Execute RFE Feature Selection`}
                      </button>
                    </div>
                  </div>
                ) : null}

                <div className="preprocessed-preview">
                  <h2>Preprocessed Data Preview</h2>
                  <div className="preprocessed-summary-grid">
                    <div className="preprocessed-summary-card blue">
                      Original: {snapshot.summary.rows.toLocaleString()} rows × {snapshot.summary.columns} cols
                    </div>
                    <div className="preprocessed-summary-card green">
                      After preprocessing: {snapshot.summary.rows.toLocaleString()} rows × {snapshot.summary.columns} cols
                    </div>
                  </div>
                  <DataPreviewTable columns={currentColumns} rows={snapshot.summary.preview} />
                </div>

                {projectId ? (
                  <div className="button-row preprocess-next-row">
                    <Link href={`/training?projectId=${projectId}`} className="button button-primary">
                      Go to model training
                    </Link>
                  </div>
                ) : null}
              </div>
            ) : (
              <div className="panel stack">
                <div className="section-heading">
                  <p className="eyebrow">Interactive preprocessing</p>
                  <h2>Run real data preparation from the website</h2>
                </div>

                <div className="button-row">
                  <button
                    type="button"
                    className="button button-secondary"
                    disabled={!projectId || isMutating}
                    onClick={() => {
                      if (projectId) {
                        void runMutation(() => resetProject(projectId), "Project reset to the original uploaded dataset.");
                      }
                    }}
                  >
                    {isMutating ? "Working..." : "Reset to raw dataset"}
                  </button>
                </div>

                <div className="preprocess-grid preprocess-grid-wide">
                  <PreprocessCard title="Drop columns" buttonLabel="Apply column drop" onSubmit={() => projectId && dropSelection.length ? runMutation(() => dropColumns({ projectId, columns: dropSelection }), `Dropped ${dropSelection.length} column(s).`) : Promise.resolve()} disabled={!projectId || !dropSelection.length || isMutating}>
                    <label className="field">
                      <span>Columns to remove</span>
                      <select multiple className="multi-select" value={dropSelection} onChange={(event) => setDropSelection(getSelectedValues(event))}>
                        {currentColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                      </select>
                    </label>
                  </PreprocessCard>

                  <PreprocessCard title="Handle missing values" buttonLabel="Apply missing-value treatment" onSubmit={() => projectId ? runMutation(() => handleMissingValues({ projectId, strategy: missingStrategy, columns: missingColumns, fillValue: missingStrategy === "custom" ? customFillValue : undefined }), `Missing-value strategy "${missingStrategy}" applied successfully.`) : Promise.resolve()} disabled={!projectId || isMutating}>
                    <div className="field-grid">
                      <label className="field">
                        <span>Strategy</span>
                        <select value={missingStrategy} onChange={(event) => setMissingStrategy(event.target.value as typeof missingStrategy)}>
                          <option value="drop_rows">Drop rows with nulls</option>
                          <option value="mean">Fill with mean</option>
                          <option value="median">Fill with median</option>
                          <option value="mode">Fill with mode</option>
                          <option value="zero">Fill with zero</option>
                          <option value="custom">Fill with custom value</option>
                        </select>
                      </label>
                      <label className="field field-span-2">
                        <span>Apply to columns</span>
                        <MultiSelectDropdown
                          options={currentColumns}
                          placeholder="Choose options"
                          selectedValues={missingColumns}
                          onChange={setMissingColumns}
                        />
                      </label>
                    </div>
                    {missingStrategy === "custom" ? (
                      <label className="field">
                        <span>Custom fill value</span>
                        <input type="text" value={customFillValue} onChange={(event) => setCustomFillValue(event.target.value)} />
                      </label>
                    ) : null}
                  </PreprocessCard>

                  <PreprocessCard title="Encode categorical columns" buttonLabel="Apply encoding" onSubmit={() => projectId ? runMutation(() => encodeCategorical({ projectId, method: encodingMethod, columns: encodingColumns }), `Encoding method "${encodingMethod}" applied successfully.`) : Promise.resolve()} disabled={!projectId || isMutating}>
                    <div className="field-grid">
                      <label className="field">
                        <span>Method</span>
                        <select value={encodingMethod} onChange={(event) => setEncodingMethod(event.target.value as typeof encodingMethod)}>
                          <option value="label">Label encoding</option>
                          <option value="onehot">One-hot encoding</option>
                        </select>
                      </label>
                      <label className="field field-span-2">
                        <span>Columns to encode</span>
                        <MultiSelectDropdown
                          options={categoricalColumns}
                          placeholder="Choose options"
                          selectedValues={encodingColumns}
                          onChange={setEncodingColumns}
                        />
                      </label>
                    </div>
                  </PreprocessCard>

                  <PreprocessCard title="Scale numeric columns" buttonLabel="Apply scaling" onSubmit={() => projectId ? runMutation(() => scaleFeatures({ projectId, method: scalingMethod, columns: scalingColumns }), `Scaling method "${scalingMethod}" applied successfully.`) : Promise.resolve()} disabled={!projectId || isMutating}>
                    <div className="field-grid">
                      <label className="field">
                        <span>Method</span>
                        <select value={scalingMethod} onChange={(event) => setScalingMethod(event.target.value as typeof scalingMethod)}>
                          <option value="standard">Standard scaler</option>
                          <option value="minmax">Min-max scaler</option>
                          <option value="robust">Robust scaler</option>
                        </select>
                      </label>
                      <label className="field field-span-2">
                        <span>Columns to scale</span>
                        <MultiSelectDropdown
                          options={numericColumns}
                          placeholder="Choose options"
                          selectedValues={scalingColumns}
                          onChange={setScalingColumns}
                        />
                      </label>
                    </div>
                  </PreprocessCard>

                  <PreprocessCard title="Handle outliers" buttonLabel="Apply outlier handling" onSubmit={() => projectId ? runMutation(() => handleOutliers({ projectId, method: outlierMethod, columns: outlierColumns, threshold: outlierThreshold }), `Outlier method "${outlierMethod}" applied successfully.`) : Promise.resolve()} disabled={!projectId || isMutating}>
                    <div className="field-grid">
                      <label className="field">
                        <span>Method</span>
                        <select value={outlierMethod} onChange={(event) => setOutlierMethod(event.target.value as typeof outlierMethod)}>
                          <option value="iqr_remove">IQR remove rows</option>
                          <option value="iqr_cap">IQR cap values</option>
                          <option value="zscore_remove">Z-score remove rows</option>
                        </select>
                      </label>
                      <label className="field">
                        <span>Threshold</span>
                        <input type="number" min={1} max={3} step={0.1} value={outlierThreshold} onChange={(event) => setOutlierThreshold(Number(event.target.value))} />
                      </label>
                      <label className="field field-span-2">
                        <span>Columns to process</span>
                        <MultiSelectDropdown
                          options={numericColumns}
                          placeholder="Choose options"
                          selectedValues={outlierColumns}
                          onChange={setOutlierColumns}
                        />
                      </label>
                    </div>
                  </PreprocessCard>
                </div>
              </div>
            )
          ) : null}

          {activeSection === "train" ? (
          <div className={`panel stack${isDedicatedTraining ? " training-panel" : ""}`}>
            {isDedicatedTraining ? (
              <>
                <h1 className="training-title">Model Training</h1>

                <section className="training-section">
                  <h2>1 · Task &amp; Model Selection</h2>
                  <div className="training-two-col">
                    <div className="field">
                      <span>Task type</span>
                      <div className="training-radio-row">
                        <label className="training-radio">
                          <input type="radio" name="task-type" checked={taskType === "classification"} onChange={() => setTaskType("classification")} />
                          <span>Classification</span>
                        </label>
                        <label className="training-radio">
                          <input type="radio" name="task-type" checked={taskType === "regression"} onChange={() => setTaskType("regression")} />
                          <span>Regression</span>
                        </label>
                      </div>
                    </div>
                    <label className="field">
                      <span>Algorithm</span>
                      <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                        {availableModels.map((model) => <option key={model} value={model}>{model}</option>)}
                      </select>
                    </label>
                  </div>
                </section>

                <section className="training-section">
                  <h2>2 · Target &amp; Features</h2>
                  <div className="training-two-col">
                    <label className="field">
                      <span>Target column (y)</span>
                      <select value={targetColumn} onChange={(event) => setTargetColumn(event.target.value)}>
                        {currentColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                      </select>
                    </label>
                    <label className="field">
                      <span>Feature columns (x)</span>
                      <MultiSelectDropdown
                        options={availableFeatures}
                        placeholder="Choose options"
                        selectedValues={featureColumns}
                        onChange={setFeatureColumns}
                        displayAsChips
                      />
                    </label>
                  </div>
                </section>

                <section className="training-section">
                  <h2>3 · Training Options</h2>
                  <div className="training-options-grid">
                    <label className="field">
                      <span>Test split ratio</span>
                      <input type="range" min={0.05} max={0.5} step={0.05} value={testSize} onChange={(event) => setTestSize(Number(event.target.value))} />
                      <span className="range-value">{testSize.toFixed(2)}</span>
                    </label>
                    <label className="field">
                      <span>Random seed</span>
                      <div className="seed-input-row">
                        <input type="number" min={0} max={9999} step={1} value={randomState} onChange={(event) => setRandomState(Number(event.target.value))} />
                        <button type="button" className="seed-btn" onClick={() => setRandomState((current) => Math.max(0, current - 1))}>-</button>
                        <button type="button" className="seed-btn" onClick={() => setRandomState((current) => Math.min(9999, current + 1))}>+</button>
                      </div>
                    </label>
                    <label className="field training-checkbox-field">
                      <span>Cross-validation</span>
                      <label className="training-checkbox">
                        <input type="checkbox" checked={runCv} onChange={(event) => setRunCv(event.target.checked)} />
                        <span>5-fold Cross-Validation</span>
                      </label>
                    </label>
                  </div>
                </section>

                <section className="training-section advanced-tuning-section">
                  <div className="advanced-tuning-header">
                    <div>
                      <h2>4 · Bayesian Hyperparameter Optimization (Optuna)</h2>
                      <p className="preprocess-helper" style={{ margin: 0, marginTop: "0.25rem" }}>
                        Automatically discover optimal model hyperparameters via Tree-structured Parzen Estimator (TPE) with trial pruning.
                      </p>
                    </div>
                    <label className="switch-label">
                      <input
                        type="checkbox"
                        checked={useAdvancedTuning}
                        onChange={(e) => setUseAdvancedTuning(e.target.checked)}
                      />
                      <span className="switch-slider" />
                    </label>
                  </div>

                  {useAdvancedTuning && (
                    <div className="advanced-tuning-controls field-grid" style={{ marginTop: "1rem" }}>
                      <label className="field">
                        <span>Number of Optuna Trials</span>
                        <input
                          type="range"
                          min={5}
                          max={50}
                          step={5}
                          value={optunaTrials}
                          onChange={(e) => setOptunaTrials(Number(e.target.value))}
                        />
                        <span className="range-value">{optunaTrials} trials</span>
                      </label>

                      <label className="field training-checkbox-field" style={{ justifyContent: "center" }}>
                        <span>Trial Pruning</span>
                        <label className="training-checkbox">
                          <input
                            type="checkbox"
                            checked={optunaPruning}
                            onChange={(e) => setOptunaPruning(e.target.checked)}
                          />
                          <span>Median Pruner (Fast Convergence)</span>
                        </label>
                      </label>
                    </div>
                  )}
                </section>

                <button
                  type="button"
                  className="button button-primary training-submit"
                  disabled={!projectId || !targetColumn || !featureColumns.length || isTraining || isMutating}
                  onClick={() => void trainAndNavigateToResults()}
                >
                  {isTraining
                    ? useAdvancedTuning
                      ? "Running Bayesian optimization & training..."
                      : "Training model in background..."
                    : useAdvancedTuning
                    ? `Tune & Train (${optunaTrials} Optuna Trials)`
                    : "Train Model"}
                </button>

                {(isTraining || trainingProgress > 0) && (
                  <div className="telemetry-card">
                    <div className="telemetry-header">
                      <strong>Live Training Telemetry</strong>
                      <span className={`telemetry-stage-pill ${trainingStage}`}>
                        {isTraining && <span className="pulse-dot" />}
                        {trainingStage.replace(/_/g, " ")} ({trainingProgress}%)
                      </span>
                    </div>
                    <div className="telemetry-progress-track">
                      <div
                        className="telemetry-progress-fill"
                        style={{ width: `${Math.max(5, Math.min(100, trainingProgress))}%` }}
                      />
                    </div>
                    <p className="telemetry-message">{trainingMessage}</p>
                    {trainingLogs.length > 0 && (
                      <div className="telemetry-log-box">
                        {trainingLogs.map((log, i) => (
                          <div key={i} className="telemetry-log-item">
                            {log}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="section-heading">
                  <p className="eyebrow">Model training</p>
                  <h2>Train a real model from the website</h2>
                </div>
                <div className="preprocess-grid">
                  <PreprocessCard title="Training setup" buttonLabel={isTraining ? "Training..." : "Train model"} onSubmit={() => projectId && targetColumn && featureColumns.length ? trainAndNavigateToResults() : Promise.resolve()} disabled={!projectId || !targetColumn || !featureColumns.length || isTraining || isMutating}>
                    <div className="field-grid">
                      <label className="field">
                        <span>Task type</span>
                        <select value={taskType} onChange={(event) => setTaskType(event.target.value as typeof taskType)}>
                          <option value="classification">Classification</option>
                          <option value="regression">Regression</option>
                        </select>
                      </label>
                      <label className="field field-span-2">
                        <span>Algorithm</span>
                        <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                          {availableModels.map((model) => <option key={model} value={model}>{model}</option>)}
                        </select>
                      </label>
                      <label className="field">
                        <span>Target column</span>
                        <select value={targetColumn} onChange={(event) => setTargetColumn(event.target.value)}>
                          {currentColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                        </select>
                      </label>
                      <label className="field field-span-2">
                        <span>Feature columns</span>
                        <MultiSelectDropdown
                          options={availableFeatures}
                          placeholder="Choose options"
                          selectedValues={featureColumns}
                          onChange={setFeatureColumns}
                        />
                      </label>
                    </div>
                    <div className="field-grid">
                      <label className="field">
                        <span>Test split ratio</span>
                        <input type="number" min={0.05} max={0.5} step={0.05} value={testSize} onChange={(event) => setTestSize(Number(event.target.value))} />
                      </label>
                      <label className="field">
                        <span>Random seed</span>
                        <input type="number" min={0} max={9999} step={1} value={randomState} onChange={(event) => setRandomState(Number(event.target.value))} />
                      </label>
                      <label className="field checkbox-field">
                        <span>Cross-validation</span>
                        <input type="checkbox" checked={runCv} onChange={(event) => setRunCv(event.target.checked)} />
                      </label>
                    </div>
                    {(isTraining || trainingProgress > 0) && (
                      <div className="telemetry-card">
                        <div className="telemetry-header">
                          <strong>Live Training Telemetry</strong>
                          <span className={`telemetry-stage-pill ${trainingStage}`}>
                            {isTraining && <span className="pulse-dot" />}
                            {trainingStage.replace(/_/g, " ")} ({trainingProgress}%)
                          </span>
                        </div>
                        <div className="telemetry-progress-track">
                          <div
                            className="telemetry-progress-fill"
                            style={{ width: `${Math.max(5, Math.min(100, trainingProgress))}%` }}
                          />
                        </div>
                        <p className="telemetry-message">{trainingMessage}</p>
                      </div>
                    )}
                    {projectId ? <Link href={`/results?projectId=${projectId}`} className="button button-secondary">Open results</Link> : null}
                  </PreprocessCard>
                </div>
              </>
            )}
          </div>
          ) : null}

          {activeSection === "explore" ? (
          <div className={`panel stack explore-panel${isDedicatedExploration ? " dedicated-exploration-panel" : ""}`}>
            {!isDedicatedExploration ? (
              <div className="section-heading">
                <p className="eyebrow">Live exploration</p>
                <h2>Visual exploration</h2>
              </div>
            ) : null}
            <div className="viz-tab-strip" role="tablist" aria-label="Exploration visuals">
              <button type="button" role="tab" aria-selected={selectedChart === "histogram"} className={`viz-tab${selectedChart === "histogram" ? " active" : ""}`} onClick={() => setSelectedChart("histogram")}>📊 Distributions</button>
              <button type="button" role="tab" aria-selected={selectedChart === "correlation"} className={`viz-tab${selectedChart === "correlation" ? " active" : ""}`} onClick={() => setSelectedChart("correlation")}>🔥 Correlation</button>
              <button type="button" role="tab" aria-selected={selectedChart === "scatter"} className={`viz-tab${selectedChart === "scatter" ? " active" : ""}`} onClick={() => setSelectedChart("scatter")}>📈 Scatter</button>
              <button type="button" role="tab" aria-selected={selectedChart === "boxplot"} className={`viz-tab${selectedChart === "boxplot" ? " active" : ""}`} onClick={() => setSelectedChart("boxplot")}>📦 Box Plots</button>
              <button type="button" role="tab" aria-selected={selectedChart === "categorical"} className={`viz-tab${selectedChart === "categorical" ? " active" : ""}`} onClick={() => setSelectedChart("categorical")}>🔢 Categorical</button>
              <button type="button" role="tab" aria-selected={selectedChart === "datatypes"} className={`viz-tab${selectedChart === "datatypes" ? " active" : ""}`} onClick={() => setSelectedChart("datatypes")}>🗂️ Data Types</button>
            </div>

            <div className="viz-controls-grid">
              {(selectedChart === "histogram" || selectedChart === "boxplot") ? (
                <label className="field">
                  <span>Numeric column</span>
                  <select
                    value={selectedChart === "histogram" ? histogramColumn : boxplotColumn}
                    onChange={(event) => selectedChart === "histogram" ? setHistogramColumn(event.target.value) : setBoxplotColumn(event.target.value)}
                  >
                    {numericColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                  </select>
                </label>
              ) : null}

              {selectedChart === "boxplot" ? (
                <label className="field">
                  <span>Group by (optional)</span>
                  <select value={boxplotGroupBy} onChange={(event) => setBoxplotGroupBy(event.target.value)}>
                    <option value="">None</option>
                    {categoricalColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                  </select>
                </label>
              ) : null}

              {selectedChart === "scatter" ? (
                <>
                  <label className="field">
                    <span>X-axis</span>
                    <select value={scatterXColumn} onChange={(event) => setScatterXColumn(event.target.value)}>
                      {numericColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                    </select>
                  </label>
                  <label className="field">
                    <span>Y-axis</span>
                    <select value={scatterYColumn} onChange={(event) => setScatterYColumn(event.target.value)}>
                      {numericColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                    </select>
                  </label>
                  <label className="field">
                    <span>Colour by</span>
                    <select value={scatterColorColumn} onChange={(event) => setScatterColorColumn(event.target.value)}>
                      <option value="">None</option>
                      {currentColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                    </select>
                  </label>
                </>
              ) : null}

              {selectedChart === "categorical" ? (
                <label className="field">
                  <span>Select categorical column</span>
                  <select value={categoricalColumn} onChange={(event) => setCategoricalColumn(event.target.value)}>
                    {categoricalColumns.map((column) => <option key={column} value={column}>{column}</option>)}
                  </select>
                </label>
              ) : null}
            </div>

            {selectedChart === "datatypes" ? (
              <div className="data-types-card">
                <h2>Column data types</h2>
                <DataPreviewTable columns={["column", "type"]} rows={dataTypeRows} />
              </div>
            ) : (
              <>
                <div className={`status${chartError ? " error" : ""}`}>{chartStatus}</div>
                <PlotlyChart
                  figure={chartFigure}
                  emptyMessage="No chart is available for the current dataset and selection."
                  showModeBar
                />
              </>
            )}
          </div>
          ) : null}

          {activeSection === "explore" ? (
          <div className={`panel${isDedicatedExploration ? " data-preview-card dedicated-exploration-preview" : ""}`}>
            <h2>Data preview</h2>
            <p className="muted">Showing the first {snapshot.summary.preview.length} rows from the current processed dataset.</p>
            <DataPreviewTable columns={currentColumns} rows={snapshot.summary.preview} />
          </div>
          ) : null}

          {activeSection === "explore" && isDedicatedExploration && projectId ? (
          <div className="button-row">
            <Link href={`/preprocessing?projectId=${projectId}`} className="button button-primary">
              Go to preprocessing
            </Link>
          </div>
          ) : null}

          {!isDedicatedRoute ? (
          <div className="panel split-panel">
            <div>
              <h2>Preprocessing log</h2>
              {snapshot.summary.preprocessing_log.length ? (
                <ul className="feature-list">
                  {snapshot.summary.preprocessing_log.map((entry) => <li key={entry}>{entry}</li>)}
                </ul>
              ) : (
                <p className="muted">No preprocessing steps have been applied yet.</p>
              )}
            </div>
            <div>
              <h2>Next migration step</h2>
              <ul className="feature-list">
                <li>Add model evaluation plots like confusion matrix and residual charts</li>
                <li>Persist projects with a database instead of in-memory sessions</li>
                <li>Add one standout feature like dataset chat or report export</li>
              </ul>
            </div>
          </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

function PreprocessCard({
  title,
  buttonLabel,
  onSubmit,
  disabled,
  children,
}: {
  title: string;
  buttonLabel: string;
  onSubmit: () => void | Promise<unknown>;
  disabled: boolean;
  children: ReactNode;
}) {
  return (
    <section className="preprocess-card">
      <h3>{title}</h3>
      {children}
      <button type="button" className="button button-primary" disabled={disabled} onClick={() => void onSubmit()}>
        {buttonLabel}
      </button>
    </section>
  );
}

function MultiSelectDropdown({
  options,
  selectedValues,
  onChange,
  placeholder,
  displayAsChips = false,
}: {
  options: string[];
  selectedValues: string[];
  onChange: (nextValues: string[]) => void;
  placeholder: string;
  displayAsChips?: boolean;
}) {
  const dropdownRef = useRef<HTMLDetailsElement | null>(null);
  const summary =
    selectedValues.length === 0
      ? placeholder
      : selectedValues.length <= 2
        ? selectedValues.join(", ")
        : `${selectedValues.length} selected`;

  const allSelected = options.length > 0 && selectedValues.length === options.length;

  function closeDropdown() {
    dropdownRef.current?.removeAttribute("open");
  }

  function toggleValue(value: string) {
    if (selectedValues.includes(value)) {
      onChange(selectedValues.filter((item) => item !== value));
      closeDropdown();
      return;
    }
    onChange([...selectedValues, value]);
    closeDropdown();
  }

  function toggleSelectAll() {
    onChange(allSelected ? [] : [...options]);
    closeDropdown();
  }

  return (
    <details className="multi-select-dropdown" ref={dropdownRef}>
      <summary className={`multi-select-trigger${displayAsChips ? " chips" : ""}`}>
        {displayAsChips && selectedValues.length ? (
          <span className="multi-chip-wrap">
            {selectedValues.map((value) => (
              <span key={value} className="multi-chip">{value}</span>
            ))}
          </span>
        ) : (
          summary
        )}
      </summary>
      <div className="multi-select-menu">
        {options.length ? (
          <>
            <label className="multi-select-option select-all-option">
              <input
                type="checkbox"
                checked={allSelected}
                onChange={toggleSelectAll}
              />
              <span>Select all</span>
            </label>
            {options.map((option) => (
              <label key={option} className="multi-select-option">
                <input
                  type="checkbox"
                  checked={selectedValues.includes(option)}
                  onChange={() => toggleValue(option)}
                />
                <span>{option}</span>
              </label>
            ))}
          </>
        ) : (
          <p className="muted multi-select-empty">No options available.</p>
        )}
      </div>
    </details>
  );
}
