---
title: AudioGuard
emoji: 🎧
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# AudioGuard — Audio Splicing Forensic Analyzer

Course mini project for the Multimedia Forensics Lab, IIIT Kottayam.

Upload a recording; AudioGuard flags suspicious edit boundaries from acoustic-feature
discontinuities (MFCC, level, spectral shape, background noise), shows the evidence,
and exports a CSV report. A flagged boundary is evidence to examine, not proof of tampering.
