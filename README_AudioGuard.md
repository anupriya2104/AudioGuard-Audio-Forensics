# AudioGuard — Audio Splicing Detection & Forensic Localization

**Author and developer: Anu Priya**   
**Implementation:** Python · Librosa · NumPy · SciPy · Gradio · Plotly

AudioGuard is an audio-forensics prototype that analyzes an audio recording for abrupt changes in acoustic characteristics that may indicate an inserted or moved audio segment. It estimates the timestamps of suspicious edit boundaries and presents supporting visual evidence so that an examiner can review the recording.

> **Important:** AudioGuard is an investigative aid, not an automated authenticity verdict. A flagged boundary is a candidate for human examination; it is not proof that a recording has been manipulated. Likewise, the absence of a flagged boundary does not establish that a recording is authentic.

---

## Table of contents

- [Motivation](#motivation)
- [What AudioGuard does](#what-audioguard-does)
- [Methodology](#methodology)
- [Acoustic features](#acoustic-features)
- [Detection and localization](#detection-and-localization)
- [Dataset and evaluation](#dataset-and-evaluation)
- [Web application](#web-application)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Run locally](#run-locally)
- [Deploy on Hugging Face Spaces](#deploy-on-hugging-face-spaces)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Future improvements](#future-improvements)
- [Author](#author)

## Motivation

Audio recordings can be important sources of digital evidence in phone calls, interviews, CCTV systems and voice messages. Audio splicing involves inserting a segment from another recording into an existing recording. Even when the edit sounds natural, the inserted segment may carry different microphone characteristics, room acoustics or background noise.

AudioGuard explores whether these inconsistencies can be identified through acoustic feature analysis, and whether the likely edit locations can be shown to an examiner.

## What AudioGuard does

- Accepts audio recordings for analysis (the project targets common formats such as WAV and MP3, subject to the installed audio-decoding support).
- Converts the signal to mono and resamples it to **16 kHz** for analysis.
- Extracts frame-level acoustic descriptors.
- Compares feature statistics in windows before and after each time instant.
- Uses an adaptive threshold and peak detection to identify candidate splice boundaries.
- Displays the waveform, spectrogram and discontinuity-score curve.
- Reports the estimated timestamp and the feature group that contributed the strongest change.
- Provides audio playback around detected boundaries and a downloadable CSV report in the web interface.

## Methodology

The processing pipeline is:

```text
Input audio (WAV / MP3)
        │
        ▼
Pre-processing
(mono, 16 kHz, framing)
        │
        ▼
Acoustic feature extraction
(22 features per frame)
        │
        ▼
Robust feature normalization
(median and median absolute deviation)
        │
        ▼
Left/right window comparison
(0.5-second windows)
        │
        ▼
Discontinuity score
        │
        ▼
Adaptive threshold + peak detection
        │
        ▼
Candidate edit timestamps + visual evidence + report
```

The presentation describes a frame length of approximately **32 ms** and a hop of **10 ms** (about 100 frames per second). The implementation uses Librosa for signal analysis and NumPy/SciPy for numerical processing and peak detection. The approach is signal-processing based; it does not require model training or a GPU.

## Acoustic features

The project extracts 22 features per frame, organized into four groups:

| Feature group | Features | What they represent |
|---|---|---|
| MFCC | 13 Mel-frequency cepstral coefficients | Spectral envelope and characteristics associated with voice, microphone and recording channel |
| Level | Log RMS energy | Signal loudness and level changes |
| Spectral shape | Spectral centroid, spectral flatness, zero-crossing rate and spectral flux | Brightness, noisiness and abrupt spectral changes |
| Background noise | Noise-floor estimates in 0–1, 1–2, 2–4 and 4–8 kHz bands | Background noise and recording-environment characteristics |

Each feature is robustly normalized using the median and median absolute deviation (MAD). In the project experiments described in the presentation, background-noise features were observed to be particularly useful for identifying recording changes.

## Detection and localization

For each candidate time instant, the system compares a **0.5-second window before** the instant with a **0.5-second window after** it. It calculates the difference between the feature-group means and combines the group scores into a discontinuity score:

\[
D_g(t) = \operatorname{RMS}(\mu_{right,g}(t)-\mu_{left,g}(t))
\]

\[
D(t) = \max_g D_g(t)
\]

An adaptive threshold is derived from the score distribution:

\[
T = \operatorname{median}(D) + k \cdot \operatorname{MAD}(D)
\]

Peaks above the threshold are reported as candidate boundaries, subject to a minimum separation between detections. The feature group with the largest change is recorded as the main cue for each candidate boundary.

The threshold factor is configurable in the application. It controls the sensitivity of detection: a lower threshold can identify more subtle changes but may also increase false alarms.

## Dataset and evaluation

The project presentation describes an evaluation dataset of **72 speech clips** derived from LibriSpeech recordings, organized as 3 speakers × 6 conditions × 4 clips. The conditions are:

1. Genuine, unedited speech
2. Cross-recording splice
3. Same-recording splice
4. Noise mismatch
5. Crossfade, using 50 ms fades
6. Loudness-matched inserted segment

Known edit timestamps are retained as ground truth. The described evaluation procedure tunes the threshold on one speaker and tests on the other two speakers. A detection is considered correct when it falls within **±0.25 seconds** of a true edit point. The presentation identifies precision, recall, F1-score and localization error as evaluation metrics.

**Results note:** Numeric aggregate results and processing-time values are not included here because the project presentation's results table contains unfilled values. Please run the notebook's evaluation cells and report the measured values before adding performance claims to this README, a portfolio or a resume.

## Web application

The Gradio interface is named **AudioGuard**. It is designed to let a user:

1. Upload or record an audio sample.
2. Adjust the detection threshold factor.
3. Run the analysis.
4. Inspect the waveform, spectrogram and discontinuity-score plot.
5. Review candidate boundaries and their main acoustic cues.
6. Listen to audio around a candidate boundary.
7. Export the analysis results as a CSV report.

The interface is intended to make the algorithm's output reviewable rather than presenting a black-box authenticity decision.

## Repository structure

```text
AudioGuard/
├── app.py                         # Gradio web application and analysis pipeline
├── Audio_Splicing_Forensics.ipynb # Research, experiments and evaluation notebook
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── examples/                      # Optional sample audio (add only distributable files)
```

If your GitHub repository uses a different notebook filename, update the tree above to match the actual filename.

## Installation

**Requirements:** Python 3.9+ is recommended. A working audio backend may be required by Librosa/SoundFile for the formats you use.

Clone the repository:

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/AudioGuard.git
cd AudioGuard
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run locally

Start the Gradio application:

```bash
python app.py
```

Open the local URL printed in the terminal (typically `http://127.0.0.1:7860`).

To explore the experiments and evaluation workflow, open the notebook:

```bash
jupyter notebook Audio_Splicing_Forensics.ipynb
```

## Deploy on Hugging Face Spaces

AudioGuard can be deployed as a Gradio Space:

1. Create a new Space at [Hugging Face Spaces](https://huggingface.co/spaces).
2. Select **Gradio** as the SDK.
3. Upload `app.py` and `requirements.txt` (and this README).
4. Ensure the Space README metadata identifies `sdk: gradio` and `app_file: app.py`.
5. Wait for the build to complete, then open the Space URL and test it with audio samples you are permitted to use.

Do not publish private, confidential or personally identifying recordings in a public Space. Check the hosting platform's current storage, file-size and runtime limits before deployment.

## Limitations and responsible use

- Acoustic discontinuities are not unique to editing. A change in speaker, microphone distance, room, background noise, compression or recording device can also create a peak.
- Some edits may be difficult to detect, particularly when the inserted segment comes from the same recording or has been carefully matched in noise and loudness.
- Crossfades and other editing operations can soften transitions and affect localization accuracy.
- The estimated timestamp is approximate and depends on framing, window size, threshold and peak separation settings.
- The current method is a feature-discontinuity detector, not a complete audio-authentication system. It does not establish provenance, identify an editor or prove intent.
- Results should be corroborated with listening, metadata examination and other forensic methods. Preserve the original file and document any processing performed.

## Future improvements

- Expand testing across more speakers, recording devices, rooms and audio conditions.
- Add a reproducible evaluation report with per-condition precision, recall, F1-score and localization error.
- Study threshold calibration and sensitivity across different recording environments.
- Improve handling of same-recording splices, crossfades and loudness-matched edits.
- Add exportable evidence summaries with configuration details and reproducibility metadata.

## Author

**Anu Priya**  
B.Tech, Computer Science and Engineering  

This repository presents the project implementation and documentation under my authorship. The project is shared for educational and research demonstration purposes.
