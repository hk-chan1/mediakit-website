"use client";

import { useState, useCallback } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { UploadZone } from "@/components/upload-zone";
import { UrlInput } from "@/components/url-input";
import { ProgressTracker, type ProcessingStage } from "@/components/progress-tracker";
import { SheetMusicPreview } from "@/components/sheet-music-preview";
import { HowItWorks } from "@/components/how-it-works";
import { Download, RotateCcw, Music, Zap, Layers, Star } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Mode = "quick" | "auto" | "quality";

interface MidiData {
  notes: Array<{ pitch: number; startTime: number; duration: number; velocity: number }>;
  tempo: number;
  timeSignature: [number, number];
  keySignature: number;
}

const MODE_OPTIONS: { value: Mode; label: string; desc: string; Icon: any }[] = [
  { value: "quick",   label: "Quick",   desc: "Solo instrument, fastest",   Icon: Zap },
  { value: "auto",    label: "Auto",    desc: "Smart tier selection",        Icon: Layers },
  { value: "quality", label: "Quality", desc: "Full separation, most accurate", Icon: Star },
];

export default function HomePage() {
  const [stage, setStage] = useState<ProcessingStage | null>(null);
  const [error, setError] = useState<string | undefined>();
  const [jobId, setJobId] = useState<string | null>(null);
  const [midiData, setMidiData] = useState<MidiData | null>(null);
  const [pdfReady, setPdfReady] = useState(false);
  const [mode, setMode] = useState<Mode>("auto");

  // Tier info received progressively from status polling
  const [tier, setTier] = useState<number | null>(null);
  const [tierReason, setTierReason] = useState<string | null>(null);
  const [estimatedSeconds, setEstimatedSeconds] = useState<number | null>(null);
  const [stageTimings, setStageTimings] = useState<Record<string, number> | null>(null);

  const pollStatus = useCallback(async (id: string) => {
    const maxAttempts = 200;  // 10 minutes at 3s intervals
    let attempts = 0;

    const poll = async () => {
      if (attempts >= maxAttempts) {
        setError("Processing timed out. The video may be too long or complex.");
        setStage("error");
        return;
      }
      attempts++;

      try {
        const res = await fetch(`${API_BASE}/api/status/${id}`);
        if (!res.ok) throw new Error("Failed to check status");
        const data = await res.json();

        if (data.status === "error") {
          setError(data.error || "An unexpected error occurred.");
          setStage("error");
          return;
        }

        setStage(data.stage as ProcessingStage);
        if (data.tier != null)             setTier(data.tier);
        if (data.tier_reason)              setTierReason(data.tier_reason);
        if (data.estimated_seconds != null) setEstimatedSeconds(data.estimated_seconds);
        if (data.stage_timings)            setStageTimings(data.stage_timings);

        if (data.status === "done") {
          setMidiData(data.midi_data);
          setPdfReady(true);
          return;
        }

        setTimeout(poll, 3000);
      } catch {
        setError("Lost connection to server. Please try again.");
        setStage("error");
      }
    };

    poll();
  }, []);

  const startProcessing = () => {
    setStage("extracting");
    setError(undefined);
    setMidiData(null);
    setPdfReady(false);
    setTier(null);
    setTierReason(null);
    setEstimatedSeconds(null);
    setStageTimings(null);
  };

  const handleFileUpload = async (file: File) => {
    startProcessing();
    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("mode", mode);

      const res = await fetch(`${API_BASE}/api/process/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || "Upload failed");
      }
      const data = await res.json();
      setJobId(data.job_id);
      pollStatus(data.job_id);
    } catch (err: any) {
      setError(err.message || "Failed to upload file.");
      setStage("error");
    }
  };

  const handleUrlSubmit = async (url: string) => {
    startProcessing();
    try {
      const res = await fetch(`${API_BASE}/api/process/url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url, mode }),
      });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || "Failed to process URL");
      }
      const data = await res.json();
      setJobId(data.job_id);
      pollStatus(data.job_id);
    } catch (err: any) {
      setError(err.message || "Failed to process URL.");
      setStage("error");
    }
  };

  const handleDownload = () => {
    if (jobId) window.open(`${API_BASE}/api/download/${jobId}`, "_blank");
  };

  const handleReset = () => {
    setStage(null);
    setError(undefined);
    setJobId(null);
    setMidiData(null);
    setPdfReady(false);
    setTier(null);
    setTierReason(null);
    setEstimatedSeconds(null);
    setStageTimings(null);
  };

  const isProcessing = stage !== null && stage !== "done" && stage !== "error";

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Hero */}
      <section className="text-center py-12 max-w-3xl mx-auto">
        <div className="inline-flex items-center gap-2 bg-primary/10 text-primary rounded-full px-4 py-1.5 text-sm font-medium mb-6">
          <Music className="h-4 w-4" />
          AI-Powered Music Transcription
        </div>
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight mb-4">
          Turn Any Video Into
          <br />
          <span className="text-primary">Piano Sheet Music</span>
        </h1>
        <p className="text-lg text-muted-foreground max-w-xl mx-auto">
          Upload a video or paste a link. AI separates the audio sources, detects every note,
          and generates downloadable piano sheet music in PDF format.
        </p>
      </section>

      {/* Main Card */}
      <section className="max-w-2xl mx-auto mb-16">
        <Card>
          <CardHeader>
            <CardTitle>Create Sheet Music</CardTitle>
            <CardDescription>Upload a video file or paste a video URL to get started</CardDescription>
          </CardHeader>
          <CardContent>
            {stage === null ? (
              <div className="space-y-5">
                {/* Mode selector */}
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2 uppercase tracking-wide">
                    Processing mode
                  </p>
                  <div className="grid grid-cols-3 gap-2">
                    {MODE_OPTIONS.map(({ value, label, desc, Icon }) => (
                      <button
                        key={value}
                        onClick={() => setMode(value)}
                        className={`flex flex-col items-center gap-1 rounded-lg border p-3 text-center transition-all text-sm ${
                          mode === value
                            ? "border-primary bg-primary/5 text-primary"
                            : "border-muted hover:border-primary/40 text-muted-foreground"
                        }`}
                      >
                        <Icon className="h-4 w-4" />
                        <span className="font-medium">{label}</span>
                        <span className="text-xs opacity-70 leading-tight">{desc}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <Tabs defaultValue="upload" className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="upload">Upload Video</TabsTrigger>
                    <TabsTrigger value="url">Paste URL</TabsTrigger>
                  </TabsList>
                  <TabsContent value="upload" className="mt-4">
                    <UploadZone onFileSelected={handleFileUpload} />
                  </TabsContent>
                  <TabsContent value="url" className="mt-4">
                    <UrlInput onUrlSubmit={handleUrlSubmit} />
                  </TabsContent>
                </Tabs>
              </div>
            ) : (
              <div className="space-y-6">
                <ProgressTracker
                  stage={stage}
                  error={error}
                  tier={tier}
                  tierReason={tierReason}
                  estimatedSeconds={estimatedSeconds}
                  stageTimings={stageTimings}
                />

                {midiData && <SheetMusicPreview midiData={midiData} />}

                {pdfReady && (
                  <div className="flex gap-3 pt-2">
                    <Button onClick={handleDownload} size="lg" className="flex-1">
                      <Download className="mr-2 h-5 w-5" />
                      Download PDF
                    </Button>
                    <Button onClick={handleReset} variant="outline" size="lg">
                      <RotateCcw className="mr-2 h-4 w-4" />
                      New
                    </Button>
                  </div>
                )}

                {stage === "error" && (
                  <Button onClick={handleReset} variant="outline" className="w-full">
                    <RotateCcw className="mr-2 h-4 w-4" />
                    Try Again
                  </Button>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </section>

      <div id="how-it-works">
        <HowItWorks />
      </div>
    </div>
  );
}
