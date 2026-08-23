import "./globals.css";
import type { Metadata } from "next";
import Link from "next/link";
import { TopNav } from "@/components/top-nav";
import { PyodideProvider } from "@/lib/pyodide-context";

export const metadata: Metadata = {
  title: "DataPilot",
  description: "Full-stack ML analytics workspace with WebAssembly EDA & Training Queue.",
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
                <span>
                  <strong>DataPilot</strong>
                  <small>ML analytics workspace</small>
                </span>
              </Link>
              <TopNav items={navItems} />
            </header>
            <main className="page">{children}</main>
          </div>
        </PyodideProvider>
      </body>
    </html>
  );
}
