"use client";

import { Progress } from "@/components/ui/progress";
import { CheckCircle, Loader2, Circle, Zap, Layers, Star } from "lucide-react";

export type ProcessingStage =
  | "extracting"
  | "analyzing"
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
  tier?: number | null;
  tierReason?: string | null;
  estimatedSeconds?: number | null;
  stageTimings?: Record<string, number> | null;
}

const stages = [
  { key: "extracting",   label: "Extracting audio from video" },
  { key: "analyzing",    label: "Analyzing audio complexity" },
  { key: "separating",   label: "Separating audio sources" },
  { key: "transcribing", label: "Transcribing notes" },
  { key: "quantizing",   label: "Detecting tempo & quantising rhythm" },
  { key: "arranging",    label: "Building piano arrangement" },
  { key: "generating",   label: "Engraving sheet music PDF" },
  { key: "done",         label: "Sheet music ready!" },
] as const;

const TIER_META: Record<number, { label: string; color: string; Icon: any }> = {
  1: { label: "Quick  (Tier 1)",   color: "text-emerald-600 bg-emerald-50 border-emerald-200",  Icon: Zap },
  2: { label: "Medium (Tier 2)",   color: "text-amber-600 bg-amber-50 border-amber-200",        Icon: Layers },
  3: { label: "Quality (Tier 3)",  color: "text-violet-600 bg-violet-50 border-violet-200",     Icon: Star },
};

function getProgress(stage: ProcessingStage): number {
  switch (stage) {
    case "extracting":   return 5;
    case "analyzing":    return 12;
    case "separating":   return 28;
    case "transcribing": return 52;
    case "quantizing":   return 68;
    case "arranging":    return 82;
    case "generating":   return 93;
    case "done":         return 100;
    case "error":        return 0;
  }
}

function fmt(s: number): string {
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${s % 60}s`;
}

export function ProgressTracker({
  stage, error, tier, tierReason, estimatedSeconds, stageTimings,
}: ProgressTrackerProps) {
  if (error) {
    return (
      <div className="p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive">
        <p className="font-medium">Processing failed</p>
        <p className="text-sm mt-1">{error}</p>
      </div>
    );
  }

  const stageIndex = stages.findIndex((s) => s.key === stage);
  const tierInfo = tier ? TIER_META[tier] : null;

  return (
    <div className="space-y-5">
      {/* Tier badge — shown as soon as tier is known */}
      {tierInfo && (
        <div className={`flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full border w-fit ${tierInfo.color}`}>
          <tierInfo.Icon className="h-3.5 w-3.5" />
          <span>{tierInfo.label}</span>
          {estimatedSeconds && stage !== "done" && (
            <span className="opacity-70">· est. {fmt(estimatedSeconds)}</span>
          )}
        </div>
      )}
      {tierReason && stage !== "done" && (
        <p className="text-xs text-muted-foreground -mt-2">{tierReason}</p>
      )}

      <Progress value={getProgress(stage)} className="h-2.5" />

      <div className="space-y-2.5">
        {stages.map((s, i) => {
          const isComplete = i < stageIndex || stage === "done";
          const isCurrent  = s.key === stage && stage !== "done";
          const timing     = stageTimings?.[s.key];

          return (
            <div key={s.key} className="flex items-center gap-3">
              {isComplete ? (
                <CheckCircle className="h-4 w-4 text-green-500 shrink-0" />
              ) : isCurrent ? (
                <Loader2 className="h-4 w-4 text-primary animate-spin shrink-0" />
              ) : (
                <Circle className="h-4 w-4 text-muted-foreground/25 shrink-0" />
              )}
              <span className={`text-sm flex-1 ${
                isComplete ? "text-green-600 font-medium"
                : isCurrent  ? "text-foreground font-medium"
                : "text-muted-foreground"
              }`}>
                {s.label}
              </span>
              {isComplete && timing !== undefined && (
                <span className="text-xs text-muted-foreground tabular-nums">{fmt(timing)}</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Per-stage timing summary after completion */}
      {stage === "done" && stageTimings && Object.keys(stageTimings).length > 0 && (
        <div className="text-xs text-muted-foreground border-t pt-3 space-y-1">
          <p className="font-medium text-foreground">Stage timings</p>
          {Object.entries(stageTimings).map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span className="capitalize">{k.replace("_", " ")}</span>
              <span className="tabular-nums">{fmt(v)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
