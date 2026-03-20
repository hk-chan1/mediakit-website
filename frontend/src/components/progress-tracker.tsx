"use client";

import { Progress } from "@/components/ui/progress";
import { CheckCircle, Loader2, Circle } from "lucide-react";

export type ProcessingStage = "extracting" | "analyzing" | "generating" | "done" | "error";

interface ProgressTrackerProps {
  stage: ProcessingStage;
  error?: string;
}

const stages = [
  { key: "extracting", label: "Extracting audio from video..." },
  { key: "analyzing", label: "Analyzing music & detecting notes..." },
  { key: "generating", label: "Generating piano sheet music..." },
  { key: "done", label: "Sheet music ready!" },
] as const;

function getProgress(stage: ProcessingStage): number {
  switch (stage) {
    case "extracting": return 20;
    case "analyzing": return 55;
    case "generating": return 85;
    case "done": return 100;
    case "error": return 0;
  }
}

export function ProgressTracker({ stage, error }: ProgressTrackerProps) {
  if (error) {
    return (
      <div className="space-y-4">
        <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive">
          <p className="font-medium">Processing failed</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      </div>
    );
  }

  const stageIndex = stages.findIndex((s) => s.key === stage);

  return (
    <div className="space-y-6">
      <Progress value={getProgress(stage)} className="h-3" />
      <div className="space-y-3">
        {stages.map((s, i) => {
          const isComplete = i < stageIndex || stage === "done";
          const isCurrent = s.key === stage && stage !== "done";

          return (
            <div key={s.key} className="flex items-center gap-3">
              {isComplete ? (
                <CheckCircle className="h-5 w-5 text-green-500" />
              ) : isCurrent ? (
                <Loader2 className="h-5 w-5 text-primary animate-spin" />
              ) : (
                <Circle className="h-5 w-5 text-muted-foreground/30" />
              )}
              <span
                className={`text-sm ${
                  isComplete ? "text-green-600 font-medium" : isCurrent ? "text-foreground font-medium" : "text-muted-foreground"
                }`}
              >
                {s.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
