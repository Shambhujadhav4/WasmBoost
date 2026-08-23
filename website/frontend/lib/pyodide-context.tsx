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
      {/* Top subtle WebAssembly engine status badge */}
      <div className="wasm-status-bar">
        <div className="wasm-status-content">
          <span className={`wasm-status-dot ${statusEvent.status}`} />
          <span className="wasm-status-text">
            <strong>WebAssembly (Pyodide) Engine:</strong> {statusEvent.message}
          </span>
        </div>
      </div>
      {children}
    </PyodideContext.Provider>
  );
}

export function usePyodide() {
  return useContext(PyodideContext);
}
