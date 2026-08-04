import { useEffect, useRef, useState } from "react";

export type SocketState = "connecting" | "open" | "closed";

interface Options {
  url: string | null;
  onMessage: (data: string) => void;
  onOpen?: () => void;
  onBeforeReconnect?: () => Promise<void> | void;
}

export function useReconnectingSocket({
  url,
  onMessage,
  onOpen,
  onBeforeReconnect,
}: Options) {
  const [state, setState] = useState<SocketState>("closed");
  const wsRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);
  const stoppedRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep latest callbacks
  const cbs = useRef({ onMessage, onOpen, onBeforeReconnect });
  cbs.current = { onMessage, onOpen, onBeforeReconnect };

  useEffect(() => {
    if (!url) {
      setState("closed");
      return;
    }
    stoppedRef.current = false;

    const connect = async () => {
      if (stoppedRef.current) return;
      if (attemptRef.current > 0) {
        try {
          await cbs.current.onBeforeReconnect?.();
        } catch {
          /* ignore */
        }
      }
      setState("connecting");
      const ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => {
        attemptRef.current = 0;
        setState("open");
        cbs.current.onOpen?.();
      };
      ws.onmessage = (e) => {
        if (typeof e.data === "string") cbs.current.onMessage(e.data);
      };
      ws.onclose = () => {
        setState("closed");
        if (stoppedRef.current) return;
        const delay = Math.min(1000 * 2 ** attemptRef.current, 10000);
        attemptRef.current += 1;
        timerRef.current = setTimeout(connect, delay);
      };
      ws.onerror = () => {
        ws.close();
      };
    };

    connect();

    return () => {
      stoppedRef.current = true;
      if (timerRef.current) clearTimeout(timerRef.current);
      wsRef.current?.close();
      wsRef.current = null;
      attemptRef.current = 0;
    };
  }, [url]);

  return { state };
}
