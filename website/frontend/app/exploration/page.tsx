import { Suspense } from "react";
import { ExplorationDashboard } from "@/components/exploration-dashboard";

export default function ExplorationPage() {
  return (
    <Suspense fallback={null}>
      <ExplorationDashboard
        stepLabel="Step 2"
        pageTitle="Data exploration"
        description="Explore histograms, box plots, correlation heatmaps, scatter plots, and categorical counts."
      />
    </Suspense>
  );
}

