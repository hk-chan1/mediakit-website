"use client";

import { Progress } from "@/components/ui/progress";
import { CheckCircle, Loader2, Circle } from "lucide-react";

export type ProcessingStage =
  | "extracting"
  | "separating"
  | "transcribing"
  | "quantizing"
  | "arranging"
  | "generating"
  | "done"
  | "error";

interface ProgressTrackerProps {
  stage: ProcessingStage;
  error?: string;
}

const stages = [
  { key: "extracting",   label: "Extracting audio from video…" },
  { key: "separating",   label: "Separating vocals, bass & instruments (Demucs)…" },
  { key: "transcribing", label: "Transcribing notes from each stem…" },
  { key: "quantizing",   label: "Detecting tempo & quantising to beat grid…" },
  { key: "arranging",    label: "Building two-hand piano arrangement…" },
  { key: "generating",   label: "Engraving sheet music PDF…" },
  { key: "done",         label: "Sheet music ready!" },
] as const;

function getProgress(stage: ProcessingStage): number {
  switch (stage) {
    case "extracting":   return 8;
    case "separating":   return 25;
    case "transcribing": return 45;
    case "quantizing":   return 62;
    case "arranging":    return 78;
    case "generating":   return 92;
    case "done":         return 100;
    case "error":        return 0;
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
          const isCurrent  = s.key === stage && stage !== "done";

          return (
            <div key={s.key} className="flex items-center gap-3">
              {isComplete ? (
                <CheckCircle className="h-5 w-5 text-green-500 shrink-0" />
              ) : isCurrent ? (
                <Loader2 className="h-5 w-5 text-primary animate-spin shrink-0" />
              ) : (
                <Circle className="h-5 w-5 text-muted-foreground/30 shrink-0" />
              )}
              <span
                className={`text-sm ${
                  isComplete
                    ? "text-green-600 font-medium"
                    : isCurrent
                    ? "text-foreground font-medium"
                    : "text-muted-foreground"
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
