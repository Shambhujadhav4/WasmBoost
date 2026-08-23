export type DatasetSummary = {
  project_id: string;
  input_kind: string;
  source_filename: string | null;
  source_mime_type: string | null;
  file_size_bytes: number | null;
  rows: number;
  columns: number;
  column_names: string[];
  missing_values: number;
  duplicate_rows: number;
  preview: Record<string, unknown>[];
  preprocessing_log: string[];
};

export type ColumnInfoRow = {
  column: string;
  dtype: string;
  non_null: number;
  null: number;
  unique: number;
};

export type DatasetInsights = {
  project_id: string;
  summary: DatasetSummary;
  column_info: ColumnInfoRow[];
  descriptive_statistics: Record<string, unknown>[];
};

export type ProjectSnapshot = {
  summary: DatasetSummary;
  target_column: string | null;
  feature_columns: string[];
  model_results: Record<string, unknown> | null;
  trained_task_type: string | null;
  artifact_available: boolean;
  artifact_filename: string | null;
  skops_artifact_available?: boolean;
  onnx_artifact_available?: boolean;
  skops_artifact_filename?: string | null;
  onnx_artifact_filename?: string | null;
};

export type FeatureImportanceRow = {
  feature: string;
  importance: number;
};

export type ModelRecommendation = {
  model_name: string;
  mean_score: number;
  std_score: number;
  metric_scores: Record<string, number>;
};

export type WorkflowRecommendation = {
  project_id: string;
  recommended_task_type: string | null;
  recommended_target_column: string | null;
  benchmark_metric: string | null;
  available_benchmark_metrics: string[];
  best_model: ModelRecommendation | null;
  candidate_models: ModelRecommendation[];
  suggested_preprocessing_steps: string[];
  notes: string[];
};

export type TrainTaskResponse = {
  task_id: string;
  project_id: string;
  status: string;
  message: string;
};

export type TrainStatusResponse = {
  task_id: string;
  state: string;
  status?: string | null;
  progress?: number | null;
  message?: string | null;
  result?: ProjectSnapshot | Record<string, unknown> | null;
  error?: string | null;
};

export type TelemetryEvent = {
  task_id: string;
  project_id: string;
  status: "queued" | "connected" | "preparing" | "training" | "cross_validating" | "evaluating" | "exporting" | "completed" | "failed" | string;
  progress: number;
  message: string;
  timestamp?: string;
  snapshot?: ProjectSnapshot | null;
  error?: string | null;
};

