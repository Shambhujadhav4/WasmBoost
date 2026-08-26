"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { DataPreviewTable } from "@/components/data-preview-table";
import { ChartSkeleton, PlotlyChart } from "@/components/plotly-chart";
import {
  fetchBoxplotFigure,
  fetchCorrelationFigure,
  fetchCountplotFigure,
  fetchDistributionFigure,
  fetchProjectSnapshot,
  fetchScatterFigure,
  fetchVisualizationMetadata,
} from "@/lib/api";
import { pyodideClient } from "@/lib/pyodide-client";
import { ACTIVE_PROJECT_STORAGE_KEY } from "@/lib/project-session";
import type { ProjectSnapshot, VizMetadata } from "@/lib/types";

export type ExplorationChartType =
  | "histogram"
  | "boxplot"
  | "correlation"
  | "scatter"
  | "categorical"
  | "datatypes";

interface ExplorationDashboardProps {
  initialSection?: string;
  showSectionTabs?: boolean;
  stepLabel?: string;
  pageTitle?: string;
  description?: string;
}

export function ExplorationDashboard({
  stepLabel = "Step 2",
  pageTitle = "Data exploration",
  description = "Explore histograms, box plots, correlation heatmaps, scatter plots, and categorical counts.",
}: ExplorationDashboardProps) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [projectId, setProjectId] = useState<string | null>(null);
  const [snapshot, setSnapshot] = useState<ProjectSnapshot | null>(null);
  const [vizMeta, setVizMeta] = useState<VizMetadata | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Tab & Selection State
  const [selectedChart, setSelectedChart] = useState<ExplorationChartType>("histogram");
  const [histogramColumn, setHistogramColumn] = useState<string>("");
  const [boxplotColumn, setBoxplotColumn] = useState<string>("");
  const [boxplotGroupBy, setBoxplotGroupBy] = useState<string>("");
  const [scatterXColumn, setScatterXColumn] = useState<string>("");
  const [scatterYColumn, setScatterYColumn] = useState<string>("");
  const [scatterColorColumn, setScatterColorColumn] = useState<string>("");
  const [categoricalColumn, setCategoricalColumn] = useState<string>("");

  // In-Memory Chart Cache state: Record<cacheKey, PlotlyFigure>
  const [chartCache, setChartCache] = useState<Record<string, any>>({});
  const [activeChartFigure, setActiveChartFigure] = useState<Record<string, unknown> | null>(null);
  const [isChartLoading, setIsChartLoading] = useState<boolean>(false);
  const [chartStatus, setChartStatus] = useState<string>("Initializing visualization engine...");
  const [chartError, setChartError] = useState<string | null>(null);

  // Ref to track if default initialization has already run for this dataset
  const hasInitializedDefaultRef = useRef<boolean>(false);

  const currentColumns = snapshot?.summary.column_names ?? [];
  const numericColumns = vizMeta?.numeric_columns ?? [];
  const categoricalColumns = vizMeta?.categorical_columns ?? [];

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
    [currentColumns, numericColumns, categoricalColumns]
  );

  // Resolve active project ID from URL query params or localStorage
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
      setChartStatus("No active dataset found. Please upload a dataset to begin exploration.");
    }
  }, [searchParams]);

  // Load project snapshot and metadata (Pyodide first, API fallback)
  useEffect(() => {
    if (!projectId) return;

    let cancelled = false;

    async function loadDatasetContext() {
      try {
        let loadedSnapshot: ProjectSnapshot | null = null;
        let loadedMeta: VizMetadata | null = null;

        try {
          // Attempt loading snapshot & metadata from Pyodide WebWorker
          const pyodideSummary = await pyodideClient.getSnapshot(projectId!);
          const pyodideMeta = await pyodideClient.getVizMetadata(projectId!);
          if (pyodideSummary && pyodideSummary.columns > 0) {
            loadedSnapshot = {
              summary: pyodideSummary,
              target_column: null,
              feature_columns: pyodideSummary.column_names,
              model_results: null,
              trained_task_type: null,
              artifact_available: false,
              artifact_filename: null,
            };
            loadedMeta = pyodideMeta;
          }
        } catch {
          // Pyodide session not loaded in worker; fallback to backend API
        }

        if (!loadedSnapshot) {
          loadedSnapshot = await fetchProjectSnapshot(projectId!);
        }
        if (!loadedMeta) {
          loadedMeta = await fetchVisualizationMetadata(projectId!);
        }

        if (cancelled) return;

        setSnapshot(loadedSnapshot);
        setVizMeta(loadedMeta);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : "Failed to load dataset exploration data.";
        setError(msg);
      }
    }

    loadDatasetContext();

    return () => {
      cancelled = true;
    };
  }, [projectId]);

  // Sync column selector defaults when columns metadata becomes available
  useEffect(() => {
    if (numericColumns.length > 0) {
      setHistogramColumn((cur) => (cur && numericColumns.includes(cur) ? cur : numericColumns[0]));
      setBoxplotColumn((cur) => (cur && numericColumns.includes(cur) ? cur : numericColumns[0]));
      setScatterXColumn((cur) => (cur && numericColumns.includes(cur) ? cur : numericColumns[0]));
      setScatterYColumn((cur) =>
        cur && numericColumns.includes(cur)
          ? cur
          : numericColumns[Math.min(1, Math.max(0, numericColumns.length - 1))]
      );
    }
    if (categoricalColumns.length > 0) {
      setCategoricalColumn((cur) =>
        cur && categoricalColumns.includes(cur) ? cur : categoricalColumns[0]
      );
    }
  }, [numericColumns.join("|"), categoricalColumns.join("|")]);

  /**
   * Refactored Asynchronous Chart Fetcher with In-Memory Caching (Task 3)
   * 1. Constructs key: `${chartType}-${columnName}`
   * 2. Returns immediately on cache hit.
   * 3. Dispatches to Pyodide WebWorker on cache miss, caches the result, and updates active chart.
   */
  const fetchChartData = useCallback(
    async (
      chartType: ExplorationChartType,
      columnName: string,
      options?: {
        xColumn?: string;
        yColumn?: string;
        colorColumn?: string;
        groupBy?: string;
      }
    ) => {
      if (!projectId) return;

      if (chartType === "datatypes") {
        setActiveChartFigure(null);
        setChartError(null);
        setChartStatus("Column data types inferred from dataset metadata.");
        return;
      }

      // 1. Construct deterministic cache key
      let cacheKey = `${chartType}-${columnName}`;
      if (chartType === "scatter") {
        cacheKey = `${chartType}-${options?.xColumn || ""}-${options?.yColumn || ""}-${options?.colorColumn || ""}`;
      } else if (chartType === "boxplot" && options?.groupBy) {
        cacheKey = `${chartType}-${columnName}-grouped-${options.groupBy}`;
      } else if (chartType === "correlation") {
        cacheKey = `correlation-${projectId}`;
      }

      // 2. Cache Hit: Set active chart instantly in 0ms without re-computation
      if (chartCache[cacheKey]) {
        setActiveChartFigure(chartCache[cacheKey]);
        setChartError(null);
        setChartStatus(`Loaded from React memory cache (${cacheKey}) ⚡`);
        setIsChartLoading(false);
        return;
      }

      // 3. Cache Miss: Dispatch calculation to pyodideClient Web Worker
      setIsChartLoading(true);
      setChartError(null);
      setChartStatus(`Computing ${chartType} visualization in Pyodide WebAssembly...`);

      try {
        let computedFigure: Record<string, any> | null = null;

        // Try Pyodide WebAssembly computation on background worker thread
        try {
          computedFigure = await pyodideClient.generateChart(
            projectId,
            chartType as any,
            {
              column: columnName,
              xColumn: options?.xColumn,
              yColumn: options?.yColumn,
              colorColumn: options?.colorColumn,
              groupBy: options?.groupBy,
            }
          );
        } catch {
          // Worker fallback or uninitialized session
        }

        // Fallback to FastAPI backend visualization endpoint if Pyodide returned null
        if (!computedFigure) {
          if (chartType === "histogram" && columnName) {
            computedFigure = await fetchDistributionFigure({ projectId, column: columnName });
          } else if (chartType === "boxplot" && columnName) {
            computedFigure = await fetchBoxplotFigure({ projectId, column: columnName });
          } else if (chartType === "categorical" && columnName) {
            computedFigure = await fetchCountplotFigure({ projectId, column: columnName });
          } else if (chartType === "correlation") {
            computedFigure = await fetchCorrelationFigure(projectId);
          } else if (chartType === "scatter" && options?.xColumn && options?.yColumn) {
            computedFigure = await fetchScatterFigure({
              projectId,
              xColumn: options.xColumn,
              yColumn: options.yColumn,
              colorColumn: options.colorColumn || undefined,
            });
          }
        }

        if (computedFigure) {
          // Save into in-memory React state cache
          setChartCache((prev) => ({
            ...prev,
            [cacheKey]: computedFigure,
          }));
          setActiveChartFigure(computedFigure);
          setChartStatus("Interactive visualization ready.");
        } else {
          setActiveChartFigure(null);
          setChartStatus("No chart data available for the selected column(s).");
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to generate chart visualization.";
        setActiveChartFigure(null);
        setChartError(errorMsg);
        setChartStatus("Chart computation failed.");
      } finally {
        setIsChartLoading(false);
      }
    },
    [projectId, chartCache]
  );

  /**
   * Default Initialization (Task 4)
   * Triggers once when the exploration component mounts with dataset metadata,
   * automatically selecting and computing the chart for the *first numeric column*.
   */
  useEffect(() => {
    if (
      !hasInitializedDefaultRef.current &&
      projectId &&
      numericColumns.length > 0
    ) {
      hasInitializedDefaultRef.current = true;
      const initialColumn = numericColumns[0];
      setHistogramColumn(initialColumn);
      setSelectedChart("histogram");
      fetchChartData("histogram", initialColumn);
    }
  }, [projectId, numericColumns, fetchChartData]);

  // Reactive chart updater when user changes tabs or column selection
  const handleTabChange = (type: ExplorationChartType) => {
    setSelectedChart(type);
    if (type === "histogram") {
      fetchChartData("histogram", histogramColumn || numericColumns[0] || "");
    } else if (type === "boxplot") {
      fetchChartData("boxplot", boxplotColumn || numericColumns[0] || "", {
        groupBy: boxplotGroupBy || undefined,
      });
    } else if (type === "correlation") {
      fetchChartData("correlation", "");
    } else if (type === "scatter") {
      fetchChartData("scatter", "", {
        xColumn: scatterXColumn || numericColumns[0] || "",
        yColumn: scatterYColumn || numericColumns[1] || numericColumns[0] || "",
        colorColumn: scatterColorColumn || undefined,
      });
    } else if (type === "categorical") {
      fetchChartData("categorical", categoricalColumn || categoricalColumns[0] || "");
    } else if (type === "datatypes") {
      fetchChartData("datatypes", "");
    }
  };

  const handleHistogramChange = (col: string) => {
    setHistogramColumn(col);
    fetchChartData("histogram", col);
  };

  const handleBoxplotChange = (col: string, grp?: string) => {
    setBoxplotColumn(col);
    const activeGrp = grp !== undefined ? grp : boxplotGroupBy;
    fetchChartData("boxplot", col, { groupBy: activeGrp || undefined });
  };

  const handleBoxplotGroupByChange = (grp: string) => {
    setBoxplotGroupBy(grp);
    if (boxplotColumn) {
      fetchChartData("boxplot", boxplotColumn, { groupBy: grp || undefined });
    }
  };

  const handleScatterChange = (x: string, y: string, color?: string) => {
    setScatterXColumn(x);
    setScatterYColumn(y);
    const activeColor = color !== undefined ? color : scatterColorColumn;
    setScatterColorColumn(activeColor);
    if (x && y) {
      fetchChartData("scatter", "", {
        xColumn: x,
        yColumn: y,
        colorColumn: activeColor || undefined,
      });
    }
  };

  const handleCategoricalChange = (col: string) => {
    setCategoricalColumn(col);
    fetchChartData("categorical", col);
  };

  return (
    <div className="stack dashboard-stack">
      {/* Top Header Card */}
      <div className="panel header-card">
        <div>
          <p className="eyebrow">{stepLabel}</p>
          <h1>{pageTitle}</h1>
          <p className="description">{description}</p>
        </div>
        <div className="header-actions">
          {projectId ? (
            <Link
              href={`/preprocessing?projectId=${projectId}`}
              className="button button-primary"
            >
              Go to Preprocessing →
            </Link>
          ) : (
            <button
              type="button"
              className="button button-secondary"
              onClick={() => router.push("/upload")}
            >
              Upload Dataset
            </button>
          )}
        </div>
      </div>

      {error ? <div className="status error">{error}</div> : null}

      {/* Exploration Interactive Panel */}
      <div className="panel stack explore-panel dedicated-exploration-panel">
        <div className="section-heading">
          <p className="eyebrow">Interactive Exploratory Data Analysis</p>
          <h2>Visual Analytics & Feature Profiling</h2>
        </div>

        {/* Tab Strip */}
        <div className="viz-tab-strip" role="tablist" aria-label="Exploration visuals">
          <button
            type="button"
            role="tab"
            aria-selected={selectedChart === "histogram"}
            className={`viz-tab${selectedChart === "histogram" ? " active" : ""}`}
            onClick={() => handleTabChange("histogram")}
          >
            📊 Distributions
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={selectedChart === "correlation"}
            className={`viz-tab${selectedChart === "correlation" ? " active" : ""}`}
            onClick={() => handleTabChange("correlation")}
          >
            🔥 Correlation
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={selectedChart === "scatter"}
            className={`viz-tab${selectedChart === "scatter" ? " active" : ""}`}
            onClick={() => handleTabChange("scatter")}
          >
            📈 Scatter
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={selectedChart === "boxplot"}
            className={`viz-tab${selectedChart === "boxplot" ? " active" : ""}`}
            onClick={() => handleTabChange("boxplot")}
          >
            📦 Box Plots
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={selectedChart === "categorical"}
            className={`viz-tab${selectedChart === "categorical" ? " active" : ""}`}
            onClick={() => handleTabChange("categorical")}
          >
            🔢 Categorical
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={selectedChart === "datatypes"}
            className={`viz-tab${selectedChart === "datatypes" ? " active" : ""}`}
            onClick={() => handleTabChange("datatypes")}
          >
            🗂️ Data Types
          </button>
        </div>

        {/* Dynamic Controls Grid */}
        <div className="viz-controls-grid">
          {selectedChart === "histogram" ? (
            <label className="field">
              <span>Select numeric column</span>
              <select
                value={histogramColumn}
                onChange={(e) => handleHistogramChange(e.target.value)}
              >
                {numericColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {selectedChart === "boxplot" ? (
            <>
              <label className="field">
                <span>Select numeric column</span>
                <select
                  value={boxplotColumn}
                  onChange={(e) => handleBoxplotChange(e.target.value)}
                >
                  {numericColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Group by (optional)</span>
                <select
                  value={boxplotGroupBy}
                  onChange={(e) => handleBoxplotGroupByChange(e.target.value)}
                >
                  <option value="">None (Single Box)</option>
                  {categoricalColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </label>
            </>
          ) : null}

          {selectedChart === "scatter" ? (
            <>
              <label className="field">
                <span>X-Axis Feature</span>
                <select
                  value={scatterXColumn}
                  onChange={(e) =>
                    handleScatterChange(e.target.value, scatterYColumn, scatterColorColumn)
                  }
                >
                  {numericColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Y-Axis Feature</span>
                <select
                  value={scatterYColumn}
                  onChange={(e) =>
                    handleScatterChange(scatterXColumn, e.target.value, scatterColorColumn)
                  }
                >
                  {numericColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Color By (Optional)</span>
                <select
                  value={scatterColorColumn}
                  onChange={(e) =>
                    handleScatterChange(scatterXColumn, scatterYColumn, e.target.value)
                  }
                >
                  <option value="">None</option>
                  {currentColumns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </label>
            </>
          ) : null}

          {selectedChart === "categorical" ? (
            <label className="field">
              <span>Select categorical column</span>
              <select
                value={categoricalColumn}
                onChange={(e) => handleCategoricalChange(e.target.value)}
              >
                {categoricalColumns.map((col) => (
                  <option key={col} value={col}>
                    {col}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
        </div>

        {/* Visual Render / Skeleton Area */}
        {selectedChart === "datatypes" ? (
          <div className="data-types-card">
            <h2>Column Schema & Data Types</h2>
            <DataPreviewTable columns={["column", "type"]} rows={dataTypeRows} />
          </div>
        ) : (
          <div className="chart-render-area">
            <div className={`status${chartError ? " error" : ""}`}>
              {chartStatus}
            </div>

            <PlotlyChart
              figure={activeChartFigure}
              isLoading={isChartLoading}
              loadingMessage={chartStatus}
              emptyMessage="No visualization available for the selected column(s). Choose another feature above."
              showModeBar
            />
          </div>
        )}
      </div>

      {/* Processed Dataset Preview Card */}
      {snapshot ? (
        <div className="panel data-preview-card dedicated-exploration-preview">
          <h2>Processed Data Preview</h2>
          <p className="muted">
            Showing first {snapshot.summary.preview.length} records from the active dataset.
          </p>
          <DataPreviewTable
            columns={currentColumns}
            rows={snapshot.summary.preview}
          />
        </div>
      ) : null}

      {/* Footer Navigation */}
      {projectId ? (
        <div className="button-row">
          <Link
            href={`/preprocessing?projectId=${projectId}`}
            className="button button-primary"
          >
            Continue to Preprocessing Pipeline →
          </Link>
        </div>
      ) : null}
    </div>
  );
}

export default ExplorationDashboard;
