import type {
  DatasetInsights,
  DatasetSummary,
  FeatureImportanceRow,
  ProjectSnapshot,
  WorkflowRecommendation,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000/api";

type UploadParams = {
  file: File;
  separator: string;
  encoding: string;
  headerRow: number;
};

export function getModelArtifactUrl(
  projectId: string,
  format: "skops" | "onnx" = "skops",
): string {
  return `${API_BASE_URL}/train/${projectId}/artifact?format=${format}`;
}

export async function uploadFile({
  file,
  separator,
  encoding,
  headerRow,
}: UploadParams): Promise<DatasetSummary> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("separator", separator === "\t" ? "\\t" : separator);
  formData.append("encoding", encoding);
  formData.append("header_row", String(headerRow));

  const response = await safeFetch(`${API_BASE_URL}/upload/file`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Upload failed."));
  }

  return response.json();
}

export async function uploadCsv(params: UploadParams): Promise<DatasetSummary> {
  return uploadFile(params);
}

export async function fetchProjectSnapshot(
  projectId: string,
): Promise<ProjectSnapshot> {
  const response = await safeFetch(`${API_BASE_URL}/upload/${projectId}`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, "Unable to load project snapshot."),
    );
  }

  return response.json();
}

export async function fetchDatasetInsights(
  projectId: string,
): Promise<DatasetInsights> {
  const response = await safeFetch(
    `${API_BASE_URL}/upload/${projectId}/insights`,
    {
      method: "GET",
      cache: "no-store",
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, "Unable to load dataset insights."),
    );
  }

  return response.json();
}

export async function fetchWorkflowRecommendation(
  projectId: string,
  benchmarkMetric?: string,
): Promise<WorkflowRecommendation> {
  const query = benchmarkMetric
    ? `?benchmark_metric=${encodeURIComponent(benchmarkMetric)}`
    : "";
  const response = await safeFetch(
    `${API_BASE_URL}/upload/${projectId}/workflow-recommendation${query}`,
    {
      method: "GET",
      cache: "no-store",
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, "Unable to load workflow recommendation."),
    );
  }

  return response.json();
}

export async function resetProject(
  projectId: string,
): Promise<ProjectSnapshot> {
  const response = await safeFetch(`${API_BASE_URL}/upload/${projectId}/reset`, {
    method: "POST",
  });

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, "Unable to reset the project."),
    );
  }

  return response.json();
}

export async function dropColumns(params: {
  projectId: string;
  columns: string[];
}): Promise<ProjectSnapshot> {
  return postJson<ProjectSnapshot>(`${API_BASE_URL}/preprocess/drop-columns`, {
    project_id: params.projectId,
    columns: params.columns,
  });
}

export async function handleMissingValues(params: {
  projectId: string;
  strategy:
    | "drop_rows"
    | "drop_columns"
    | "mean"
    | "median"
    | "mode"
    | "zero"
    | "custom";
  columns: string[];
  fillValue?: string;
}): Promise<ProjectSnapshot> {
  return postJson<ProjectSnapshot>(`${API_BASE_URL}/preprocess/missing-values`, {
    project_id: params.projectId,
    strategy: params.strategy,
    columns: params.columns,
    fill_value: params.fillValue ?? null,
  });
}

export async function encodeCategorical(params: {
  projectId: string;
  method: "label" | "onehot";
  columns: string[];
}): Promise<ProjectSnapshot> {
  return postJson<ProjectSnapshot>(`${API_BASE_URL}/preprocess/encode`, {
    project_id: params.projectId,
    method: params.method,
    columns: params.columns,
  });
}

export async function scaleFeatures(params: {
  projectId: string;
  method: "standard" | "minmax" | "robust";
  columns: string[];
}): Promise<ProjectSnapshot> {
  return postJson<ProjectSnapshot>(`${API_BASE_URL}/preprocess/scale`, {
    project_id: params.projectId,
    method: params.method,
    columns: params.columns,
  });
}

export async function handleOutliers(params: {
  projectId: string;
  method: "iqr_remove" | "iqr_cap" | "zscore_remove";
  columns: string[];
  threshold: number;
}): Promise<ProjectSnapshot> {
  return postJson<ProjectSnapshot>(`${API_BASE_URL}/preprocess/outliers`, {
    project_id: params.projectId,
    method: params.method,
    columns: params.columns,
    threshold: params.threshold,
  });
}

export async function trainModel(params: {
  projectId: string;
  taskType: "classification" | "regression";
  modelName: string;
  targetColumn: string;
  featureColumns: string[];
  testSize: number;
  randomState: number;
  runCv: boolean;
}): Promise<ProjectSnapshot> {
  return postJson<ProjectSnapshot>(`${API_BASE_URL}/train`, {
    project_id: params.projectId,
    task_type: params.taskType,
    model_name: params.modelName,
    target_column: params.targetColumn,
    feature_columns: params.featureColumns,
    test_size: params.testSize,
    random_state: params.randomState,
    run_cv: params.runCv,
  });
}

export async function fetchFeatureImportance(
  projectId: string,
): Promise<FeatureImportanceRow[] | null> {
  const response = await safeFetch(
    `${API_BASE_URL}/train/${projectId}/feature-importance`,
    {
      method: "GET",
      cache: "no-store",
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, "Unable to load feature importance."),
    );
  }

  const body = (await response.json()) as {
    feature_importance?: FeatureImportanceRow[] | null;
  };
  return body.feature_importance ?? null;
}

export async function fetchVisualizationMetadata(projectId: string): Promise<{
  project_id: string;
  numeric_columns: string[];
  categorical_columns: string[];
}> {
  const response = await safeFetch(`${API_BASE_URL}/visualize/${projectId}/metadata`, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, "Unable to load visualization metadata."),
    );
  }

  return response.json();
}

export async function fetchMissingValuesFigure(
  projectId: string,
): Promise<Record<string, unknown> | null> {
  return fetchFigure(`${API_BASE_URL}/visualize/${projectId}/missing-values`);
}

export async function fetchCorrelationFigure(
  projectId: string,
): Promise<Record<string, unknown> | null> {
  return fetchFigure(`${API_BASE_URL}/visualize/${projectId}/correlation`);
}

export async function fetchDistributionFigure(params: {
  projectId: string;
  column: string;
}): Promise<Record<string, unknown> | null> {
  return fetchFigure(
    `${API_BASE_URL}/visualize/${params.projectId}/distribution?column=${encodeURIComponent(
      params.column,
    )}`,
  );
}

export async function fetchBoxplotFigure(params: {
  projectId: string;
  column: string;
}): Promise<Record<string, unknown> | null> {
  return fetchFigure(
    `${API_BASE_URL}/visualize/${params.projectId}/boxplot?column=${encodeURIComponent(
      params.column,
    )}`,
  );
}

export async function fetchCountplotFigure(params: {
  projectId: string;
  column: string;
}): Promise<Record<string, unknown> | null> {
  return fetchFigure(
    `${API_BASE_URL}/visualize/${params.projectId}/countplot?column=${encodeURIComponent(
      params.column,
    )}`,
  );
}

export async function fetchScatterFigure(params: {
  projectId: string;
  xColumn: string;
  yColumn: string;
  colorColumn?: string;
}): Promise<Record<string, unknown> | null> {
  const query = new URLSearchParams({
    x_col: params.xColumn,
    y_col: params.yColumn,
  });
  if (params.colorColumn) {
    query.set("color_col", params.colorColumn);
  }

  return fetchFigure(
    `${API_BASE_URL}/visualize/${params.projectId}/scatter?${query.toString()}`,
  );
}

export async function fetchModelEvaluationFigures(projectId: string): Promise<{
  project_id: string;
  task_type: string | null;
  confusion_matrix: Record<string, unknown> | null;
  roc_curve: Record<string, unknown> | null;
  actual_vs_predicted: Record<string, unknown> | null;
  residuals: Record<string, unknown> | null;
}> {
  const response = await safeFetch(
    `${API_BASE_URL}/visualize/${projectId}/model-evaluation`,
    {
      method: "GET",
      cache: "no-store",
    },
  );

  if (!response.ok) {
    throw new Error(
      await readErrorMessage(response, "Unable to load model evaluation charts."),
    );
  }

  return response.json();
}

async function postJson<TResponse>(
  url: string,
  payload: Record<string, unknown>,
): Promise<TResponse> {
  const response = await safeFetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Request failed."));
  }

  return response.json();
}

async function fetchFigure(url: string): Promise<Record<string, unknown> | null> {
  const response = await safeFetch(url, {
    method: "GET",
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response, "Unable to load chart."));
  }

  const body = (await response.json()) as {
    figure?: Record<string, unknown> | null;
  };
  return body.figure ?? null;
}

async function readErrorMessage(
  response: Response,
  fallbackMessage: string,
): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail || fallbackMessage;
  } catch {
    return fallbackMessage;
  }
}

async function safeFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch {
    throw new Error(
      `Cannot reach backend API at ${API_BASE_URL}. Start backend server on http://localhost:8000 and retry.`,
    );
  }
}
