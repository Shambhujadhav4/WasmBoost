/**
 * Pyodide WebAssembly Client Manager (Phase 5)
 * Interfaces with background Web Worker for client-side EDA & Preprocessing.
 */

import type {
  DatasetInsights,
  DatasetSummary,
  FeatureRankingItem,
  MutualInformationItem,
  WorkflowRecommendation,
} from "./types";

export type PyodideRuntimeStatus =
  | "uninitialized"
  | "loading_runtime"
  | "loading_packages"
  | "ready"
  | "busy"
  | "error";

export interface PyodideStatusEvent {
  status: PyodideRuntimeStatus;
  message: string;
  error?: string;
}

export interface PyodideLoadResult {
  summary: DatasetSummary;
  insights: DatasetInsights;
  recommendations: WorkflowRecommendation;
}

export interface PyodideExportResult {
  project_id: string;
  rows: number;
  columns: string[];
  records: Record<string, unknown>[];
  csv_text: string;
}

export interface PyodideFeatureSelectionResult {
  project_id: string;
  method: "mi" | "rfe";
  target_column: string;
  selected_features: string[];
  eliminated_features: string[];
  rankings: FeatureRankingItem[];
  summary: DatasetSummary;
}

type PromiseHandlers = {
  resolve: (value: any) => void;
  reject: (reason?: any) => void;
};

class PyodideClient {
  private worker: Worker | null = null;
  private status: PyodideRuntimeStatus = "uninitialized";
  private statusMessage: string = "Pyodide uninitialized.";
  private listeners: Set<(event: PyodideStatusEvent) => void> = new Set();
  private pendingRequests: Map<string, PromiseHandlers> = new Map();
  private initPromise: Promise<void> | null = null;

  public getStatus(): PyodideStatusEvent {
    return { status: this.status, message: this.statusMessage };
  }

  public subscribe(listener: (event: PyodideStatusEvent) => void): () => void {
    this.listeners.add(listener);
    listener(this.getStatus());
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notify(event: PyodideStatusEvent) {
    this.status = event.status;
    this.statusMessage = event.message;
    this.listeners.forEach((fn) => fn(event));
  }

  public async init(): Promise<void> {
    if (this.status === "ready") {
      return;
    }
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = new Promise<void>((resolve, reject) => {
      if (typeof window === "undefined") {
        resolve();
        return;
      }

      if (!this.worker) {
        this.worker = new Worker("/pyodide.worker.js");
        this.worker.onmessage = (event: MessageEvent) => {
          const data = event.data;
          if (data.type === "STATUS") {
            this.notify({
              status: data.status,
              message: data.message,
              error: data.error,
            });
            if (data.status === "ready") {
              this.status = "ready";
              resolve();
            } else if (data.status === "error") {
              this.status = "error";
              this.initPromise = null;
              reject(new Error(data.message || data.error));
            }
            return;
          }

          const { id, status, result, error } = data;
          if (id === "init") {
            if (status === "SUCCESS") {
              this.status = "ready";
              resolve();
            } else {
              this.status = "error";
              this.initPromise = null;
              reject(new Error(error || "Worker initialization failed."));
            }
            return;
          }

          const handler = this.pendingRequests.get(id);
          if (handler) {
            this.pendingRequests.delete(id);
            if (status === "SUCCESS") {
              handler.resolve(result);
            } else {
              handler.reject(new Error(error || "Worker operation failed."));
            }
          }
        };

        this.worker.onerror = (err) => {
          this.notify({
            status: "error",
            message: "Worker runtime encountered a fatal error.",
            error: err.message,
          });
          this.status = "error";
          this.initPromise = null;
          reject(err);
        };
      }

      // Send INIT action directly to worker without going through dispatch
      this.worker.postMessage({ id: "init", action: "INIT", payload: {} });
    });

    return this.initPromise;
  }

  private async dispatch<T>(action: string, payload: Record<string, unknown>): Promise<T> {
    if (this.status !== "ready") {
      await this.init();
    }
    if (!this.worker) {
      throw new Error("Pyodide worker is unavailable in this environment.");
    }

    const id = Math.random().toString(36).substring(2, 15);
    return new Promise<T>((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      this.worker!.postMessage({ id, action, payload });
    });
  }

  public async loadCsv(
    projectId: string,
    csvText: string,
    filename: string = "dataset.csv",
    separator: string = ",",
    headerRow: number = 0
  ): Promise<PyodideLoadResult> {
    return this.dispatch<PyodideLoadResult>("LOAD_CSV", {
      projectId,
      csvText,
      filename,
      separator,
      headerRow,
    });
  }

  public async getSnapshot(projectId: string): Promise<DatasetSummary> {
    return this.dispatch<DatasetSummary>("GET_SNAPSHOT", { projectId });
  }

  public async getInsights(projectId: string): Promise<DatasetInsights> {
    return this.dispatch<DatasetInsights>("GET_INSIGHTS", { projectId });
  }

  public async getVizMetadata(projectId: string): Promise<{
    project_id: string;
    numeric_columns: string[];
    categorical_columns: string[];
  }> {
    return this.dispatch("GET_VIZ_METADATA", { projectId });
  }

  public async dropColumns(projectId: string, columns: string[]): Promise<DatasetSummary> {
    return this.dispatch<DatasetSummary>("PREPROCESS_DROP", { projectId, columns });
  }

  public async handleMissing(
    projectId: string,
    strategy: string,
    columns?: string[],
    customValue: number = 0
  ): Promise<DatasetSummary> {
    return this.dispatch<DatasetSummary>("PREPROCESS_MISSING", {
      projectId,
      strategy,
      columns: columns || [],
      customValue,
    });
  }

  public async encodeCategoricals(
    projectId: string,
    method: "label" | "onehot",
    columns?: string[]
  ): Promise<DatasetSummary> {
    return this.dispatch<DatasetSummary>("PREPROCESS_ENCODE", {
      projectId,
      method,
      columns: columns || [],
    });
  }

  public async scaleNumeric(
    projectId: string,
    method: "standard" | "minmax" | "robust",
    columns?: string[]
  ): Promise<DatasetSummary> {
    return this.dispatch<DatasetSummary>("PREPROCESS_SCALE", {
      projectId,
      method,
      columns: columns || [],
    });
  }

  public async handleOutliers(
    projectId: string,
    method: "iqr_remove" | "iqr_cap" | "zscore_remove",
    columns?: string[],
    threshold: number = 1.5
  ): Promise<DatasetSummary> {
    return this.dispatch<DatasetSummary>("PREPROCESS_OUTLIERS", {
      projectId,
      method,
      columns: columns || [],
      threshold,
    });
  }

  public async computeMutualInformation(
    projectId: string,
    targetColumn: string,
    featureColumns?: string[],
    taskType?: "classification" | "regression"
  ): Promise<MutualInformationItem[]> {
    return this.dispatch<MutualInformationItem[]>("PREPROCESS_MUTUAL_INFO", {
      projectId,
      targetColumn,
      featureColumns: featureColumns || [],
      taskType,
    });
  }

  public async applyFeatureSelection(
    projectId: string,
    method: "mi" | "rfe",
    targetColumn: string,
    nFeaturesToSelect: number,
    featureColumns?: string[],
    rfeEstimator: string = "Random Forest",
    step: number = 1.0
  ): Promise<PyodideFeatureSelectionResult> {
    return this.dispatch<PyodideFeatureSelectionResult>("PREPROCESS_FEATURE_SELECTION", {
      projectId,
      method,
      targetColumn,
      nFeaturesToSelect,
      featureColumns: featureColumns || [],
      rfeEstimator,
      step,
    });
  }

  public async exportPreprocessedDataset(projectId: string): Promise<PyodideExportResult> {
    return this.dispatch<PyodideExportResult>("EXPORT_DATASET", { projectId });
  }
}

export const pyodideClient = new PyodideClient();
