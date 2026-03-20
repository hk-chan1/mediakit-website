import numpy as np
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio to MIDI note data using Spotify's Basic Pitch.
    Returns structured data with notes, tempo, time/key signatures.
    """
    model_output, midi_data, note_events = predict(
        audio_path,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=58,  # ms
        midi_tempo=120,
    )

    if not note_events or len(note_events) == 0:
        raise RuntimeError(
            "No musical notes detected in the audio. "
            "The video may not contain music, or the audio quality may be too low."
        )

    # Convert note_events to our format
    # Each note_event is (start_time_s, end_time_s, pitch_midi, velocity, [pitch_bend])
    notes = []
    for event in note_events:
        start_time = float(event[0])
        end_time = float(event[1])
        pitch = int(event[2])
        velocity = int(event[3] * 127) if event[3] <= 1.0 else int(event[3])

        notes.append({
            "pitch": pitch,
            "startTime": round(start_time, 3),
            "duration": round(end_time - start_time, 3),
            "velocity": min(127, max(1, velocity)),
        })

    # Sort by start time
    notes.sort(key=lambda n: (n["startTime"], n["pitch"]))

    # Estimate tempo from note density
    if len(notes) >= 2:
        total_duration = notes[-1]["startTime"] - notes[0]["startTime"]
        if total_duration > 0:
            estimated_tempo = int(min(200, max(60, (len(notes) / total_duration) * 15)))
        else:
            estimated_tempo = 120
    else:
        estimated_tempo = 120

    # Detect key signature (simplified: find most common pitch class)
    pitch_classes = [n["pitch"] % 12 for n in notes]
    if pitch_classes:
        from collections import Counter
        most_common_pc = Counter(pitch_classes).most_common(1)[0][0]
        key_sig = most_common_pc  # 0=C, 1=C#, etc.
    else:
        key_sig = 0

    return {
        "notes": notes,
        "tempo": estimated_tempo,
        "timeSignature": [4, 4],
        "keySignature": key_sig,
    }
