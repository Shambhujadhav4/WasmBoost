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

export type MutualInformationItem = {
  feature: string;
  score: number;
  normalized_score: number;
};

export type MutualInformationResponse = {
  project_id: string;
  target_column: string;
  task_type: string;
  scores: MutualInformationItem[];
  figure?: Record<string, unknown> | null;
};

export type FeatureRankingItem = {
  feature: string;
  ranking: number;
  selected: boolean;
  score?: number | null;
};

export type FeatureSelectionResponse = {
  project_id: string;
  method: "mi" | "rfe";
  target_column: string;
  selected_features: string[];
  eliminated_features: string[];
  rankings: FeatureRankingItem[];
  snapshot?: ProjectSnapshot | null;
};

export type OptunaTrialRecord = {
  trial_number: number;
  value: number | null;
  best_value: number | null;
  params: Record<string, unknown>;
  state: string;
  duration_seconds?: number | null;
};

export type OptunaOptimizationSummary = {
  study_name?: string | null;
  best_params: Record<string, unknown>;
  best_value: number | null;
  direction: string;
  metric_name: string;
  n_trials: number;
  n_pruned: number;
  n_completed: number;
  trials_history: OptunaTrialRecord[];
  param_importances: Record<string, number>;
};

export type ShapFeatureContribution = {
  feature: string;
  value: number | string | null;
  shap_value: number;
};

export type ShapSampleExplanation = {
  sample_index: number;
  base_value: number;
  output_value: number;
  contributions: ShapFeatureContribution[];
};

export type ShapBeeswarmPoint = {
  feature: string;
  feature_value: number | null;
  shap_value: number;
  sample_index: number;
};

export type ShapSummaryPayload = {
  feature_importance: { feature: string; importance: number }[];
  base_value?: number | null;
  sample_explanations: ShapSampleExplanation[];
  beeswarm_points: ShapBeeswarmPoint[];
  model_framework: string;
  is_tree_model: boolean;
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
  optuna_results?: OptunaOptimizationSummary | null;
  shap_results?: ShapSummaryPayload | null;
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
  status: "queued" | "connected" | "preparing" | "tuning" | "training" | "cross_validating" | "evaluating" | "explaining" | "exporting" | "completed" | "failed" | string;
  progress: number;
  message: string;
  timestamp?: string;
  optuna_trial?: OptunaTrialRecord | null;
  snapshot?: ProjectSnapshot | null;
  error?: string | null;
};

export type VizMetadata = {
  project_id: string;
  numeric_columns: string[];
  categorical_columns: string[];
};


