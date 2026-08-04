import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import {
  Activity,
  AlertCircle,
  Check,
  ChevronRight,
  CircleDot,
  Loader2,
  Plug,
  PlugZap,
  Rocket,
  Send,
  Sparkles,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, RunResponse, Stage, WS_BASE } from "@/lib/api";
import { useReconnectingSocket } from "@/hooks/useReconnectingSocket";
import { cn } from "@/lib/utils";

const STAGES: { key: Stage; label: string }[] = [
  { key: "form_reader", label: "Form Reader" },
  { key: "data_mapper", label: "Data Mapper" },
  { key: "filler", label: "Filler" },
  { key: "submitter", label: "Submitter" },
];

interface LogEntry {
  ts: string;
  text: string;
  tone?: "info" | "warn" | "error" | "success";
}

function nowStr() {
  return new Date().toLocaleTimeString([], { hour12: false });
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; className: string; dot: string }> = {
    idle: { label: "Idle", className: "bg-muted/40 text-muted-foreground border-border", dot: "bg-muted-foreground" },
    running: {
      label: "Running",
      className: "bg-info/15 text-info border-info/30",
      dot: "bg-info animate-live-pulse",
    },
    waiting_for_human: {
      label: "Awaiting Human",
      className: "bg-warning/15 text-warning border-warning/30",
      dot: "bg-warning animate-live-pulse",
    },
    completed: {
      label: "Completed",
      className: "bg-success/15 text-success border-success/30",
      dot: "bg-success",
    },
    error: {
      label: "Error",
      className: "bg-destructive/15 text-destructive border-destructive/40",
      dot: "bg-destructive",
    },
  };
  const s = map[status] ?? {
    label: status,
    className: "bg-muted/40 text-muted-foreground border-border",
    dot: "bg-muted-foreground",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-xs font-medium",
        s.className,
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", s.dot)} />
      {s.label}
    </span>
  );
}

export function AppShell() {
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<RunResponse | null>(null);
  const [runs, setRuns] = useState<RunResponse[]>([]);
  const [urlInput, setUrlInput] = useState("");
  const [starting, setStarting] = useState(false);
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [submittingAnswers, setSubmittingAnswers] = useState(false);
  const [frameCount, setFrameCount] = useState(0);

  const canvasImgRef = useRef<HTMLImageElement | null>(null);
  const lastStageRef = useRef<Stage | null>(null);
  const lastStatusRef = useRef<string | null>(null);
  const logsEndRef = useRef<HTMLDivElement | null>(null);

  const addLog = useCallback((text: string, tone: LogEntry["tone"] = "info") => {
    setLogs((l) => [...l.slice(-200), { ts: nowStr(), text, tone }]);
  }, []);

  // Initial: load active runs
  useEffect(() => {
    let ok = true;
    api
      .listRuns()
      .then((rs) => {
        if (!ok) return;
        setRuns(rs);
        setGlobalError(null);
      })
      .catch((e) => setGlobalError(`Backend unreachable: ${e.message}`));
    return () => {
      ok = false;
    };
  }, []);

  const refreshRuns = useCallback(() => {
    api.listRuns().then(setRuns).catch(() => {});
  }, []);

  // Status WebSocket
  const statusUrl = runId ? `${WS_BASE}/runs/${runId}/status/ws` : null;
  const statusSocket = useReconnectingSocket({
    url: statusUrl,
    onOpen: () => addLog("status socket connected", "success"),
    onBeforeReconnect: async () => {
      if (!runId) return;
      try {
        const r = await api.getRun(runId);
        setRun(r);
      } catch { /* ignore */ }
    },
    onMessage: (data) => {
      try {
        const r = JSON.parse(data) as RunResponse;
        setRun(r);
        if (r.stage && r.stage !== lastStageRef.current && r.stage !== "init") {
          addLog(`${r.stage} started`, "info");
          lastStageRef.current = r.stage;
        }
        if (r.status !== lastStatusRef.current) {
          if (r.status === "waiting_for_human") addLog("waiting for human input", "warn");
          if (r.status === "completed") addLog(`completed — ${r.completion_message ?? "ok"}`, "success");
          if (r.status === "error") addLog(`error — ${r.error_message ?? "unknown"}`, "error");
          lastStatusRef.current = r.status;
        }
        refreshRuns();
      } catch {
        /* ignore */
      }
    },
  });

  // Video WebSocket
  const videoUrl = runId ? `${WS_BASE}/runs/${runId}/video/ws` : null;
  const videoSocket = useReconnectingSocket({
    url: videoUrl,
    onOpen: () => addLog("video socket connected", "success"),
    onMessage: (data) => {
      const img = canvasImgRef.current;
      if (!img) return;
      img.src = `data:image/jpeg;base64,${data}`;
      setFrameCount((c) => c + 1);
    },
  });

  // Autoscroll log
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [logs]);

  // Reset answers when questions change
  useEffect(() => {
    if (run?.awaiting_human && run.human_questions.length) {
      setAnswers((prev) => {
        const next: Record<string, string> = {};
        for (const q of run.human_questions) next[q] = prev[q] ?? "";
        return next;
      });
    } else {
      setAnswers({});
    }
  }, [run?.awaiting_human, run?.human_questions]);

  const handleStart = async () => {
    if (!urlInput.trim()) return;
    setStarting(true);
    setGlobalError(null);
    setLogs([]);
    lastStageRef.current = null;
    lastStatusRef.current = null;
    setFrameCount(0);
    try {
      const r = await api.startRun(urlInput.trim());
      setRun(r);
      setRunId(r.run_id);
      addLog(`run ${r.run_id.slice(0, 8)} started`, "success");
      refreshRuns();
    } catch (e: any) {
      setGlobalError(`Failed to start run: ${e.message}`);
    } finally {
      setStarting(false);
    }
  };

  const handleReset = () => {
    setRunId(null);
    setRun(null);
    setLogs([]);
    setFrameCount(0);
    setAnswers({});
    lastStageRef.current = null;
    lastStatusRef.current = null;
    if (canvasImgRef.current) canvasImgRef.current.src = "";
    refreshRuns();
  };

  const handleReconnectRun = async (id: string) => {
    setLogs([]);
    setFrameCount(0);
    lastStageRef.current = null;
    lastStatusRef.current = null;
    if (canvasImgRef.current) canvasImgRef.current.src = "";
    try {
      const r = await api.getRun(id);
      setRun(r);
      setRunId(id);
      addLog(`reattached to run ${id.slice(0, 8)}`, "info");
    } catch (e: any) {
      setGlobalError(`Failed to attach: ${e.message}`);
    }
  };

  const handleSubmitAnswers = async () => {
    if (!runId || !run) return;
    if (Object.values(answers).some((v) => !v.trim())) return;
    setSubmittingAnswers(true);
    try {
      await api.answer(runId, answers);
      addLog("resumed with answers", "info");
    } catch (e: any) {
      setGlobalError(`Failed to submit: ${e.message}`);
    } finally {
      setSubmittingAnswers(false);
    }
  };

  const currentStageIdx = useMemo(() => {
    if (!run) return -1;
    return STAGES.findIndex((s) => s.key === run.stage);
  }, [run]);

  const status = run?.status ?? "idle";
  const isLive = videoSocket.state === "open" && frameCount > 0;

  return (
    <div className="flex h-screen w-full overflow-hidden p-3 gap-3">
      {/* LEFT */}
      <aside className="glass-panel flex w-[22%] min-w-[280px] flex-col rounded-2xl p-5">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-primary to-accent shadow-lg shadow-primary/30">
            <Sparkles className="h-4.5 w-4.5 text-primary-foreground" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight">AutoFill AI</div>
            <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Agent Console
            </div>
          </div>
        </div>

        <div className="mt-6 space-y-2">
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground">Run ID</div>
          <div className="mono truncate text-xs text-foreground/90">
            {runId ?? <span className="text-muted-foreground">No active run</span>}
          </div>
          <StatusBadge status={status} />
        </div>

        {/* Stepper */}
        <div className="mt-6 space-y-1.5">
          <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-2">
            Pipeline
          </div>
          {STAGES.map((s, i) => {
            const done = currentStageIdx > i && run?.status !== "error";
            const active = currentStageIdx === i && run?.status === "running";
            const errored = currentStageIdx === i && run?.status === "error";
            return (
              <div
                key={s.key}
                className={cn(
                  "flex items-center gap-3 rounded-lg border px-3 py-2 transition-all",
                  active && "border-primary/40 bg-primary/5 animate-stage-glow",
                  done && "border-success/30 bg-success/5",
                  errored && "border-destructive/40 bg-destructive/5",
                  !active && !done && !errored && "border-transparent bg-muted/20",
                )}
              >
                <div
                  className={cn(
                    "grid h-6 w-6 shrink-0 place-items-center rounded-full border text-[10px] mono",
                    active && "border-primary bg-primary/20 text-primary",
                    done && "border-success bg-success/20 text-success",
                    errored && "border-destructive bg-destructive/20 text-destructive",
                    !active && !done && !errored && "border-border text-muted-foreground",
                  )}
                >
                  {done ? (
                    <Check className="h-3 w-3" />
                  ) : errored ? (
                    <X className="h-3 w-3" />
                  ) : active ? (
                    <Loader2 className="h-3 w-3 animate-spin" />
                  ) : (
                    i + 1
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="mono text-xs">{s.label}</div>
                  {errored && run?.error_message && (
                    <div className="mt-0.5 text-[11px] text-destructive/90 leading-tight">
                      {run.error_message}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {run?.status === "completed" && run.completion_message && (
          <div className="mt-4 rounded-lg border border-success/30 bg-success/10 p-3">
            <div className="flex items-center gap-2 text-success text-xs font-medium">
              <Check className="h-3.5 w-3.5" /> Success
            </div>
            <div className="mt-1 text-[12px] text-success/90 leading-snug">
              {run.completion_message}
            </div>
          </div>
        )}

        <div className="mt-auto pt-4 space-y-3">
          <Button onClick={handleReset} variant="secondary" className="w-full">
            <Rocket className="h-4 w-4 mr-2" /> Start New Run
          </Button>

          <div>
            <div className="text-[10px] uppercase tracking-widest text-muted-foreground mb-1.5">
              Active Runs
            </div>
            <div className="max-h-40 overflow-auto space-y-1 pr-1">
              {runs.length === 0 && (
                <div className="text-[11px] text-muted-foreground italic">None</div>
              )}
              {runs.map((r) => (
                <button
                  key={r.run_id}
                  onClick={() => handleReconnectRun(r.run_id)}
                  className={cn(
                    "w-full text-left rounded-md border px-2 py-1.5 transition-colors",
                    r.run_id === runId
                      ? "border-primary/40 bg-primary/10"
                      : "border-border/60 bg-muted/20 hover:bg-muted/40",
                  )}
                >
                  <div className="mono text-[11px] truncate">{r.run_id.slice(0, 12)}</div>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <StatusBadge status={r.status} />
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      </aside>

      {/* CENTER */}
      <main className="glass-panel flex flex-1 flex-col rounded-2xl overflow-hidden">
        {!runId ? (
          <IdleLauncher
            url={urlInput}
            setUrl={setUrlInput}
            onStart={handleStart}
            starting={starting}
            error={globalError}
          />
        ) : (
          <>
            {/* Browser chrome */}
            <div className="flex items-center gap-2 border-b border-panel-border px-4 py-2.5">
              <div className="flex gap-1.5">
                <span className="h-2.5 w-2.5 rounded-full bg-destructive/70" />
                <span className="h-2.5 w-2.5 rounded-full bg-warning/70" />
                <span className="h-2.5 w-2.5 rounded-full bg-success/70" />
              </div>
              <div className="mono flex-1 truncate rounded-md bg-background/50 border border-border/60 px-3 py-1 text-[11px] text-muted-foreground">
                {run?.form_url ?? "—"}
              </div>
              <div className="flex items-center gap-3 text-[10px] mono text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  {videoSocket.state === "open" ? (
                    <PlugZap className="h-3 w-3 text-success" />
                  ) : (
                    <Plug className="h-3 w-3 text-muted-foreground" />
                  )}
                  video
                </span>
                <span className="flex items-center gap-1.5">
                  {statusSocket.state === "open" ? (
                    <PlugZap className="h-3 w-3 text-success" />
                  ) : (
                    <Plug className="h-3 w-3 text-muted-foreground" />
                  )}
                  status
                </span>
                {isLive && (
                  <span className="flex items-center gap-1.5 text-destructive font-medium">
                    <span className="h-1.5 w-1.5 rounded-full bg-destructive animate-live-pulse" />
                    LIVE
                  </span>
                )}
              </div>
            </div>

            {/* Video */}
            <div className="relative flex-1 min-h-0 bg-black/60">
              <img
                ref={canvasImgRef}
                alt="Live browser feed"
                className="absolute inset-0 h-full w-full object-contain"
              />
              {frameCount === 0 && (
                <div className="absolute inset-0 grid place-items-center">
                  <div className="text-center space-y-3">
                    <div className="mx-auto h-10 w-10 rounded-full border-2 border-primary/40 border-t-primary animate-spin" />
                    <div className="mono text-xs text-muted-foreground">
                      Waiting for browser stream…
                    </div>
                    <div className="text-[10px] text-muted-foreground/70">
                      Frames will appear once form_reader opens the page
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Timeline */}
            <div className="border-t border-panel-border bg-background/40">
              <div className="flex items-center justify-between px-4 py-2">
                <div className="flex items-center gap-2 text-[10px] uppercase tracking-widest text-muted-foreground">
                  <Activity className="h-3 w-3" /> Event Log
                </div>
                <div className="mono text-[10px] text-muted-foreground">
                  {frameCount} frames
                </div>
              </div>
              <div className="mono max-h-40 overflow-auto px-4 pb-3 text-[11px] leading-relaxed">
                {logs.length === 0 && (
                  <div className="text-muted-foreground/60 italic">No events yet…</div>
                )}
                {logs.map((l, i) => (
                  <div key={i} className="flex gap-3">
                    <span className="text-muted-foreground/60">{l.ts}</span>
                    <span
                      className={cn(
                        l.tone === "success" && "text-success",
                        l.tone === "warn" && "text-warning",
                        l.tone === "error" && "text-destructive",
                        l.tone === "info" && "text-foreground/85",
                      )}
                    >
                      — {l.text}
                    </span>
                  </div>
                ))}
                <div ref={logsEndRef} />
              </div>
            </div>
          </>
        )}
      </main>

      {/* RIGHT */}
      <aside className="glass-panel flex w-[32%] min-w-[300px] flex-col rounded-2xl p-5">
        <div className="flex items-center gap-2">
          <div className="grid h-8 w-8 place-items-center rounded-lg bg-warning/20 text-warning">
            <AlertCircle className="h-4 w-4" />
          </div>
          <div>
            <div className="text-sm font-semibold tracking-tight">Human Input</div>
            <div className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
              Agent Requests
            </div>
          </div>
        </div>

        {run?.status === "waiting_for_human" && run.human_questions.length > 0 ? (
          <div className="mt-6 flex flex-1 flex-col min-h-0">
            <div className="text-[11px] text-warning mb-3 flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-warning animate-live-pulse" />
              The agent needs {run.human_questions.length} answer
              {run.human_questions.length > 1 ? "s" : ""} to continue
            </div>
            <div className="flex-1 overflow-auto space-y-4 pr-1">
              {run.human_questions.map((q) => (
                <div key={q}>
                  <label className="mono text-[11px] text-foreground/80 mb-1.5 block">
                    {q}
                  </label>
                  <Input
                    value={answers[q] ?? ""}
                    onChange={(e) =>
                      setAnswers((a) => ({ ...a, [q]: e.target.value }))
                    }
                    placeholder="Your answer"
                    className="bg-background/50 border-border/60"
                  />
                </div>
              ))}
            </div>
            <Button
              onClick={handleSubmitAnswers}
              disabled={
                submittingAnswers ||
                Object.values(answers).some((v) => !v.trim())
              }
              className="mt-4 w-full"
            >
              {submittingAnswers ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Send className="h-4 w-4 mr-2" />
              )}
              Continue
              <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        ) : (
          <div className="mt-8 flex-1 grid place-items-center text-center">
            <div className="space-y-3">
              <div className="mx-auto grid h-14 w-14 place-items-center rounded-full bg-muted/30 border border-border/60">
                <CircleDot className="h-6 w-6 text-muted-foreground" />
              </div>
              <div className="text-sm text-muted-foreground">
                {run?.status === "waiting_for_human"
                  ? "Waiting for questions…"
                  : "No questions right now"}
              </div>
              <div className="mono text-[10px] text-muted-foreground/60">
                {run?.status === "waiting_for_human"
                  ? "The agent is preparing questions for you"
                  : "The agent will surface any it needs here"}
              </div>
            </div>
          </div>
        )}

        {globalError && (
          <div className="mt-4 rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-[11px] text-destructive">
            {globalError}
          </div>
        )}
      </aside>
    </div>
  );
}

function IdleLauncher({
  url,
  setUrl,
  onStart,
  starting,
  error,
}: {
  url: string;
  setUrl: (v: string) => void;
  onStart: () => void;
  starting: boolean;
  error: string | null;
}) {
  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="w-full max-w-xl space-y-6">
        <div className="text-center space-y-2">
          <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-primary to-accent shadow-xl shadow-primary/30">
            <Sparkles className="h-6 w-6 text-primary-foreground" />
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">
            Launch an autofill agent
          </h1>
          <p className="text-sm text-muted-foreground">
            Point the agent at any form URL. Watch it read, map, fill, and submit
            in real time.
          </p>
        </div>
        <div className="glass-panel rounded-xl p-4 space-y-3">
          <label className="mono text-[10px] uppercase tracking-widest text-muted-foreground">
            Form URL
          </label>
          <div className="flex gap-2">
            <Input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://example.com/apply"
              onKeyDown={(e) => e.key === "Enter" && onStart()}
              className="mono bg-background/50 border-border/60"
            />
            <Button
              onClick={onStart}
              disabled={starting || !url.trim()}
              className="min-w-32"
            >
              {starting ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Rocket className="h-4 w-4 mr-2" />
              )}
              Launch Agent
            </Button>
          </div>
          <p className="text-[11px] text-muted-foreground">
            Requires the AutoFill AI backend running locally with CORS enabled
            for this origin.
          </p>
        </div>
        {error && (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive text-center">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
