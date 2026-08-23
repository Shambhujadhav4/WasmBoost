import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";
import { TopNav } from "@/components/top-nav";
import { PyodideProvider, PyodideStatusPill } from "@/lib/pyodide-context";

export const metadata: Metadata = {
  title: "DataPilot | ML Analytics & Experimentation Engine",
  description: "Modern full-stack ML workspace featuring client-side WebAssembly EDA, Bayesian optimization, and TreeSHAP explainability.",
};

const navItems = [
  { href: "/", label: "Overview" },
  { href: "/upload", label: "Upload" },
  { href: "/exploration", label: "Exploration" },
  { href: "/preprocessing", label: "Preprocessing" },
  { href: "/training", label: "Training" },
  { href: "/results", label: "Results" },
];

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <PyodideProvider>
          <div className="shell">
            <header className="topbar">
              <Link href="/" className="brand">
                <span className="brand-mark">DP</span>
                <span className="brand-text">
                  <strong>DataPilot</strong>
                  <small>ML Workspace</small>
                </span>
              </Link>
              <TopNav items={navItems} />
              <div className="topbar-actions">
                <PyodideStatusPill />
              </div>
            </header>
            <main className="page">{children}</main>
          </div>
        </PyodideProvider>
      </body>
    </html>
  );
}
