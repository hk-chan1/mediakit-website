"use client";

import { useEffect, useRef } from "react";

interface SheetMusicPreviewProps {
  midiData: {
    notes: Array<{
      pitch: number;
      startTime: number;
      duration: number;
      velocity: number;
    }>;
    tempo: number;
    timeSignature: [number, number];
    keySignature: number;
  } | null;
}

const NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

function midiToNoteName(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  const note = NOTE_NAMES[midi % 12];
  return `${note}${octave}`;
}

function midiToVexKey(midi: number): string {
  const octave = Math.floor(midi / 12) - 1;
  const noteIndex = midi % 12;
  const vexNotes = ["c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"];
  return `${vexNotes[noteIndex]}/${octave}`;
}

export function SheetMusicPreview({ midiData }: SheetMusicPreviewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const renderedRef = useRef(false);

  useEffect(() => {
    if (!midiData || !containerRef.current || renderedRef.current) return;

    const renderSheet = async () => {
      try {
        const VF = await import("vexflow");
        const { Factory } = VF.default || VF;

        const container = containerRef.current!;
        container.innerHTML = "";

        const width = Math.min(container.clientWidth, 900);
        const factory = new Factory({
          renderer: { elementId: container.id, width, height: 400 },
        });

        const score = factory.EasyScore();
        const system = factory.System({ width: width - 40 });

        // Split notes into treble (>= C4/60) and bass (< C4/60)
        const trebleNotes = midiData.notes
          .filter((n) => n.pitch >= 60)
          .sort((a, b) => a.startTime - b.startTime)
          .slice(0, 16);

        const bassNotes = midiData.notes
          .filter((n) => n.pitch < 60)
          .sort((a, b) => a.startTime - b.startTime)
          .slice(0, 16);

        // Build treble voice
        const trebleKeys =
          trebleNotes.length > 0
            ? trebleNotes.map((n) => midiToVexKey(n.pitch)).join(", ")
            : "c/4, e/4, g/4, c/5";

        const trebleDurations = trebleNotes.length > 0
          ? trebleNotes.map(() => "q").join(", ")
          : "q, q, q, q";

        // Build bass voice
        const bassKeys =
          bassNotes.length > 0
            ? bassNotes.map((n) => midiToVexKey(n.pitch)).join(", ")
            : "c/3, e/3, g/3, c/3";

        const bassDurations = bassNotes.length > 0
          ? bassNotes.map(() => "q").join(", ")
          : "q, q, q, q";

        // Take only groups of 4 for proper measures
        const trebleStr = trebleKeys.split(", ").slice(0, 4).join(", ");
        const trebleDurStr = trebleDurations.split(", ").slice(0, 4).join(", ");
        const bassStr = bassKeys.split(", ").slice(0, 4).join(", ");
        const bassDurStr = bassDurations.split(", ").slice(0, 4).join(", ");

        const notesWithDurations = trebleStr
          .split(", ")
          .map((note, i) => {
            const dur = trebleDurStr.split(", ")[i] || "q";
            return `${note}/${dur}`;
          })
          .join(", ");

        const bassNotesWithDurations = bassStr
          .split(", ")
          .map((note, i) => {
            const dur = bassDurStr.split(", ")[i] || "q";
            return `${note}/${dur}`;
          })
          .join(", ");

        system
          .addStave({ voices: [score.voice(score.notes(notesWithDurations))] })
          .addClef("treble")
          .addTimeSignature(`${midiData.timeSignature[0]}/${midiData.timeSignature[1]}`);

        system
          .addStave({ voices: [score.voice(score.notes(bassNotesWithDurations, { clef: "bass" }))] })
          .addClef("bass")
          .addTimeSignature(`${midiData.timeSignature[0]}/${midiData.timeSignature[1]}`);

        system.addConnector("brace");
        system.addConnector("singleRight");
        system.addConnector("singleLeft");

        factory.draw();
        renderedRef.current = true;
      } catch (err) {
        console.error("VexFlow render error:", err);
        if (containerRef.current) {
          containerRef.current.innerHTML = `<div class="p-8 text-center text-muted-foreground">
            <p class="font-medium">Sheet music preview</p>
            <p class="text-sm mt-2">Notes detected: ${midiData.notes.length}</p>
            <p class="text-sm">Tempo: ${midiData.tempo} BPM</p>
            <p class="text-sm">The full PDF download will contain the complete sheet music.</p>
          </div>`;
        }
      }
    };

    renderSheet();
  }, [midiData]);

  if (!midiData) return null;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-lg">Sheet Music Preview</h3>
        <div className="flex gap-4 text-sm text-muted-foreground">
          <span>Tempo: {midiData.tempo} BPM</span>
          <span>
            Time: {midiData.timeSignature[0]}/{midiData.timeSignature[1]}
          </span>
          <span>Notes: {midiData.notes.length}</span>
        </div>
      </div>
      <div
        id="sheet-music-container"
        ref={containerRef}
        className="bg-white rounded-lg border p-4 min-h-[200px] overflow-x-auto"
      />
    </div>
  );
}
