export const API_BASE = "http://localhost:8000";
export const WS_BASE = "ws://localhost:8000";

export type RunStatus = "running" | "waiting_for_human" | "completed" | "error";
export type Stage =
  | "init"
  | "form_reader"
  | "data_mapper"
  | "filler"
  | "submitter"
  | "supervisor";

export interface RunResponse {
  run_id: string;
  status: RunStatus;
  form_url: string;
  stage: Stage;
  completion_message: string | null;
  error_message: string | null;
  awaiting_human: boolean;
  human_questions: string[];
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  startRun: (form_url: string) =>
    jsonFetch<RunResponse>("/runs", {
      method: "POST",
      body: JSON.stringify({ form_url }),
    }),
  listRuns: () => jsonFetch<RunResponse[]>("/runs"),
  getRun: (id: string) => jsonFetch<RunResponse>(`/runs/${id}`),
  answer: (id: string, answers: Record<string, string>) =>
    jsonFetch<RunResponse>(`/runs/${id}/answer`, {
      method: "POST",
      body: JSON.stringify({ answers }),
    }),
  health: () => jsonFetch<{ status: string }>("/health"),
};
