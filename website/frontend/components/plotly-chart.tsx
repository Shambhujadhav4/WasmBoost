"use client";

import dynamic from "next/dynamic";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

export function PlotlyChart({
  figure,
  emptyMessage,
  fontColor = "#0a0a0a",
  showModeBar = false,
}: {
  figure: Record<string, unknown> | null;
  emptyMessage: string;
  fontColor?: string;
  showModeBar?: boolean;
}) {
  if (!figure) {
    return <p className="muted">{emptyMessage}</p>;
  }

  const data = Array.isArray(figure.data)
    ? (figure.data as Record<string, unknown>[])
    : [];
  const rawLayout = (figure.layout as Record<string, any>) ?? {};
  const config = (figure.config as Record<string, unknown>) ?? {};

  // Deeply transform layout for crisp contrast on light backgrounds
  const processedLayout: Record<string, any> = {
    ...rawLayout,
    autosize: true,
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: {
      family: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
      color: fontColor,
      size: 12,
      ...(rawLayout.font ?? {}),
      color: fontColor,
    },
  };

  // Title styling
  if (rawLayout.title) {
    if (typeof rawLayout.title === "string") {
      processedLayout.title = {
        text: rawLayout.title,
        font: {
          family: "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
          color: fontColor,
          size: 13,
        },
      };
    } else {
      processedLayout.title = {
        ...rawLayout.title,
        font: {
          family: "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
          color: fontColor,
          size: 13,
          ...(rawLayout.title.font ?? {}),
          color: fontColor,
        },
      };
    }
  }

  // Transform all xaxis, yaxis, xaxis2, yaxis2, etc.
  for (const key of Object.keys(rawLayout)) {
    if (/^[xy]axis\d*$/.test(key)) {
      const axis = rawLayout[key] ?? {};
      const axisTitle =
        typeof axis.title === "string"
          ? { text: axis.title, font: { color: fontColor, size: 12 } }
          : typeof axis.title === "object" && axis.title !== null
          ? {
              ...axis.title,
              font: {
                color: fontColor,
                size: 12,
                ...(axis.title.font ?? {}),
                color: fontColor,
              },
            }
          : undefined;

      processedLayout[key] = {
        ...axis,
        gridcolor: "rgba(0, 0, 0, 0.07)",
        zerolinecolor: "rgba(0, 0, 0, 0.2)",
        linecolor: "rgba(0, 0, 0, 0.2)",
        tickfont: {
          family: "JetBrains Mono, ui-monospace, monospace",
          color: "#3a3a3a",
          size: 11,
          ...(axis.tickfont ?? {}),
          color: "#3a3a3a",
        },
        ...(axisTitle ? { title: axisTitle } : {}),
      };
    }
  }

  // Colorbar styling
  if (rawLayout.coloraxis?.colorbar) {
    processedLayout.coloraxis = {
      ...rawLayout.coloraxis,
      colorbar: {
        ...rawLayout.coloraxis.colorbar,
        tickfont: {
          family: "JetBrains Mono, ui-monospace, monospace",
          color: "#3a3a3a",
          size: 10,
          ...(rawLayout.coloraxis.colorbar.tickfont ?? {}),
          color: "#3a3a3a",
        },
        title: {
          font: {
            color: fontColor,
            size: 11,
            ...(rawLayout.coloraxis.colorbar.title?.font ?? {}),
            color: fontColor,
          },
          ...(typeof rawLayout.coloraxis.colorbar.title === "object"
            ? rawLayout.coloraxis.colorbar.title
            : { text: rawLayout.coloraxis.colorbar.title }),
        },
      },
    };
  }

  // Legend styling
  if (rawLayout.legend) {
    processedLayout.legend = {
      ...rawLayout.legend,
      font: {
        family: "Inter, -apple-system, sans-serif",
        color: fontColor,
        size: 11,
        ...(rawLayout.legend.font ?? {}),
        color: fontColor,
      },
    };
  }

  // Convert white/light lines in shapes to visible dark lines
  if (Array.isArray(rawLayout.shapes)) {
    processedLayout.shapes = rawLayout.shapes.map((s: any) => {
      if (s?.line?.color && (typeof s.line.color === "string")) {
        const c = s.line.color.toLowerCase();
        if (c.includes("255,255,255") || c === "#fff" || c === "#ffffff" || c === "white") {
          return {
            ...s,
            line: {
              ...s.line,
              color: "rgba(0, 0, 0, 0.25)",
            },
          };
        }
      }
      return s;
    });
  }

  // Convert white/light text in annotations
  if (Array.isArray(rawLayout.annotations)) {
    processedLayout.annotations = rawLayout.annotations.map((a: any) => {
      return {
        ...a,
        font: {
          family: "Inter, -apple-system, sans-serif",
          color: fontColor,
          size: 11,
          ...(a?.font ?? {}),
          color: fontColor,
        },
      };
    });
  }

  return (
    <div className="plot-shell">
      <Plot
        data={data}
        layout={processedLayout}
        config={{
          responsive: true,
          displayModeBar: showModeBar,
          ...config,
        }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}

