import { Upload, Music, FileDown } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";

const steps = [
  {
    icon: Upload,
    title: "Upload or Paste",
    description: "Upload a video file (MP4, MOV, WebM) or paste a URL from YouTube, Vimeo, or any video platform.",
  },
  {
    icon: Music,
    title: "AI Analyzes Music",
    description:
      "Our AI extracts the audio, detects the melody, chords, and rhythm, then arranges it for piano.",
  },
  {
    icon: FileDown,
    title: "Download Sheet Music",
    description: "Get a professional piano sheet music PDF with treble and bass clef, key signature, and tempo.",
  },
];

export function HowItWorks() {
  return (
    <section className="py-16">
      <h2 className="text-3xl font-bold text-center mb-12">How It Works</h2>
      <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
        {steps.map((step, i) => (
          <Card key={i} className="text-center border-none shadow-none bg-muted/50">
            <CardContent className="pt-8 pb-6 px-6">
              <div className="mx-auto w-14 h-14 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                <step.icon className="h-7 w-7 text-primary" />
              </div>
              <div className="text-xs font-bold text-primary mb-2">STEP {i + 1}</div>
              <h3 className="font-semibold text-lg mb-2">{step.title}</h3>
              <p className="text-sm text-muted-foreground">{step.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </section>
  );
}
