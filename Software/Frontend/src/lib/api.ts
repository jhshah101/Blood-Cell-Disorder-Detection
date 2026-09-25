/**
 * Typed client for the FastAPI inference backend (Software/Backend/backend.py).
 *
 * The base URL comes from `VITE_API_URL` (see .env.example) and falls back to
 * the local development server.
 */
export const API_URL: string =
  ((import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000").replace(/\/+$/, "");

export interface ModelInfo {
  experiment: string;
  backbone: string;
  use_eca: boolean;
  use_color_features: boolean;
  loss: string;
  classes: string[];
  img_size: number;
  normalization: string;
  augmentation: string;
  checkpoint: {
    path: string;
    epoch: number | null;
    validation: Record<string, number>;
  };
  device: string;
  disclaimer: string;
}

export interface PredictResponse {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  top_k: { class: string; probability: number }[];
  image: { filename: string; width: number; height: number };
  model: string;
  latency_ms: number;
  disclaimer: string;
}

export class ApiError extends Error {
  status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function readError(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
  } catch {
    /* not JSON */
  }
  return res.statusText || `HTTP ${res.status}`;
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 30_000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_URL}${path}`, { ...init, signal: init.signal ?? controller.signal });
    if (!res.ok) throw new ApiError(await readError(res), res.status);
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if ((err as Error).name === "AbortError") throw new ApiError("The request timed out. Is the backend running?");
    throw new ApiError(`Could not reach the backend at ${API_URL}. Start it with "uvicorn backend:app".`);
  } finally {
    clearTimeout(timer);
  }
}

export function fetchHealth(): Promise<{ status: string; model_loaded: boolean }> {
  return request("/health", {}, 5_000);
}

export function fetchModelInfo(): Promise<ModelInfo> {
  return request("/model-info", {}, 10_000);
}

export function classifyImage(file: File, signal?: AbortSignal): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  return request("/predict", { method: "POST", body: form, signal }, 60_000);
}
