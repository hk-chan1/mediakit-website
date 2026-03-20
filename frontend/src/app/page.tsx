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
import { Download, RotateCcw, Music } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface MidiData {
  notes: Array<{ pitch: number; startTime: number; duration: number; velocity: number }>;
  tempo: number;
  timeSignature: [number, number];
  keySignature: number;
}

export default function HomePage() {
  const [stage, setStage] = useState<ProcessingStage | null>(null);
  const [error, setError] = useState<string | undefined>();
  const [jobId, setJobId] = useState<string | null>(null);
  const [midiData, setMidiData] = useState<MidiData | null>(null);
  const [pdfReady, setPdfReady] = useState(false);

  const pollStatus = useCallback(
    async (id: string) => {
      const maxAttempts = 120; // 10 minutes max
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
    },
    []
  );

  const handleFileUpload = async (file: File) => {
    setStage("extracting");
    setError(undefined);
    setMidiData(null);
    setPdfReady(false);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(`${API_BASE}/api/process/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Upload failed");
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
    setStage("extracting");
    setError(undefined);
    setMidiData(null);
    setPdfReady(false);

    try {
      const res = await fetch(`${API_BASE}/api/process/url`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || "Failed to process URL");
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
    if (jobId) {
      window.open(`${API_BASE}/api/download/${jobId}`, "_blank");
    }
  };

  const handleReset = () => {
    setStage(null);
    setError(undefined);
    setJobId(null);
    setMidiData(null);
    setPdfReady(false);
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
          Upload a video or paste a link. Our AI extracts the music, detects notes and chords, and generates
          downloadable piano sheet music in PDF format.
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
            ) : (
              <div className="space-y-6">
                <ProgressTracker stage={stage} error={error} />

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

      {/* How It Works */}
      <div id="how-it-works">
        <HowItWorks />
      </div>
    </div>
  );
}
