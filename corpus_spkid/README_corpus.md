# Corpus

This folder contains a compact snapshot of the multi-speaker real-time agent runs.

- **audio/**: utterances re-saved as FLAC for size efficiency
- **manifest.csv**: one row per utterance with transcript, predicted label, similarity, and basics
- **speakers_db.npz**: snapshot of the speaker embedding centroids at build time (optional)
- Source of audio files: `data\live`

## Columns in manifest.csv
- `utt_id`: basename of the utterance (timestamp-based)
- `orig_path`: original path under `data\live`
- `export_path`: relative path under `audio/`
- `duration_sec`: float seconds
- `rms`: root-mean-square amplitude of the original audio
- `transcript`: ASR output (Whisper)
- `claimed_name`: name extracted from transcript if the speaker introduced themselves in that utterance
- `pred_label`: speaker label predicted by the SpeakerID model
- `pred_sim`: cosine similarity to the predicted speaker centroid
