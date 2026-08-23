"use client";

import React, { createContext, useContext, useEffect, useState } from "react";
import { pyodideClient, type PyodideRuntimeStatus, type PyodideStatusEvent } from "./pyodide-client";

interface PyodideContextValue {
  status: PyodideRuntimeStatus;
  statusMessage: string;
  isReady: boolean;
  client: typeof pyodideClient;
}

const PyodideContext = createContext<PyodideContextValue>({
  status: "uninitialized",
  statusMessage: "Pyodide uninitialized.",
  isReady: false,
  client: pyodideClient,
});

export function PyodideProvider({ children }: { children: React.ReactNode }) {
  const [statusEvent, setStatusEvent] = useState<PyodideStatusEvent>(pyodideClient.getStatus());

  useEffect(() => {
    const unsubscribe = pyodideClient.subscribe((event) => {
      setStatusEvent(event);
    });

    // Proactively initialize in background thread
    pyodideClient.init().catch((err) => {
      console.warn("Pyodide background initialization caught:", err);
    });

    return () => {
      unsubscribe();
    };
  }, []);

  const isReady = statusEvent.status === "ready";

  return (
    <PyodideContext.Provider
      value={{
        status: statusEvent.status,
        statusMessage: statusEvent.message,
        isReady,
        client: pyodideClient,
      }}
    >
      {children}
    </PyodideContext.Provider>
  );
}

export function PyodideStatusPill() {
  const { status, statusMessage } = usePyodide();

  let label = "Pyodide Idle";
  if (status === "ready") {
    label = "Pyodide WASM Ready";
  } else if (status === "loading_runtime" || status === "loading_packages" || status === "busy") {
    label = "Pyodide WASM Loading";
  } else if (status === "error") {
    label = "Pyodide Error";
  }

  return (
    <div className={`pyodide-status-pill ${status}`} title={statusMessage}>
      <span className={`status-pill-dot ${status}`} />
      <span className="status-pill-text">{label}</span>
    </div>
  );
}

export function usePyodide() {
  return useContext(PyodideContext);
}
