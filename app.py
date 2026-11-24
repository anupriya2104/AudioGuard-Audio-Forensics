"""AudioGuard - Audio Splicing Forensic Analyzer (Hugging Face Space).
import os, glob, time, warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import find_peaks
warnings.filterwarnings("ignore")

SR = 16000
FRAME_LENGTH = 512
HOP_LENGTH = 160
N_MFCC = 13
BEST_K = 2.5          # threshold factor chosen on the validation set (update from notebook section 8)


FEATURE_NAMES = ([f"mfcc_{i}" for i in range(N_MFCC)] +
                 ["log_rms", "centroid_khz", "flatness", "zcr", "flux"] +
                 ["noise_0_1k", "noise_1_2k", "noise_2_4k", "noise_4_8k"])
NOISE_BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
FEATURE_GROUPS = {                       # column indices of each group
    "MFCC (timbre / channel)": list(range(0, 13)),
    "Level": [13],
    "Spectral shape": [14, 15, 16, 17],
    "Background noise": [18, 19, 20, 21],
}

def extract_features(y, sr=SR):
    # Return (features [n_frames x 19], frame_times [n_frames]).
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("Audio must be mono.")
    if len(y) < FRAME_LENGTH * 4:
        raise ValueError("Audio is too short for analysis.")

    S = np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH))  # magnitude spectrogram
    P = S ** 2

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(
        librosa.feature.melspectrogram(S=P, sr=sr, n_mels=40)), n_mfcc=N_MFCC)
    rms = librosa.feature.rms(S=S, frame_length=FRAME_LENGTH)[0]
    log_rms = np.log10(rms + 1e-6)
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0] / 1000.0
    flatness = np.log10(librosa.feature.spectral_flatness(S=S)[0] + 1e-6)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

    logS = np.log10(S + 1e-6)
    flux = np.concatenate([[0.0], np.mean(np.maximum(np.diff(logS, axis=1), 0), axis=0)])

    # Noise floor per band: the 10th-percentile log-magnitude of the bins in that band.
    # Between harmonics and in pauses this is dominated by background noise / mic hiss.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH)
    noise_floor = np.column_stack([np.percentile(logS[(freqs >= lo) & (freqs < hi)], 10, axis=0)
                                   for lo, hi in NOISE_BANDS])

    n = min(mfcc.shape[1], len(log_rms), len(centroid), len(flatness), len(zcr), len(flux))
    feats = np.column_stack([mfcc[:, :n].T, log_rms[:n], centroid[:n], flatness[:n],
                             zcr[:n], flux[:n], noise_floor[:n]])
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP_LENGTH)
    return feats.astype(np.float32), times


def robust_standardize(F, clip=4.0):
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0) * 1.4826
    return np.clip((F - med) / (mad + 1e-6), -clip, clip)

def discontinuity_scores(Z, win):
    # Returns D(t) [n] and the per-group scores [n x groups]. Edge frames get 0.
    n = len(Z)
    csum = np.vstack([np.zeros((1, Z.shape[1])), np.cumsum(Z, axis=0)])   # fast window means
    t = np.arange(win, n - win)
    left = (csum[t] - csum[t - win]) / win
    right = (csum[t + win] - csum[t]) / win
    diff2 = (right - left) ** 2
    group_scores = np.zeros((n, len(FEATURE_GROUPS)), dtype=np.float32)
    for gi, cols in enumerate(FEATURE_GROUPS.values()):
        group_scores[t, gi] = np.sqrt(diff2[:, cols].mean(axis=1))
    return group_scores.max(axis=1), group_scores

def detect_splices(y, sr=SR, window_seconds=0.5, threshold_factor=3.0,
                   min_separation_seconds=0.75):
    # Returns: times, scores, threshold, detections (DataFrame).
    # A detection is a candidate boundary, NOT proof of manipulation.
    feats, times = extract_features(y, sr)
    frames_per_sec = sr / HOP_LENGTH
    win = max(3, int(window_seconds * frames_per_sec))
    if len(feats) < 2 * win + 5:
        raise ValueError("Audio too short for the selected window size.")

    Z = robust_standardize(feats)
    scores, group_scores = discontinuity_scores(Z, win)

    valid = scores[win:len(scores) - win]
    med = np.median(valid)
    mad = np.median(np.abs(valid - med)) * 1.4826
    threshold = float(med + threshold_factor * mad)

    peaks, props = find_peaks(scores, height=threshold,
                              distance=max(1, int(min_separation_seconds * frames_per_sec)))
    det = pd.DataFrame({
        "timestamp_seconds": np.round(times[peaks], 3),
        "anomaly_score": np.round(props["peak_heights"], 3),
        "score_over_threshold": np.round(props["peak_heights"] / threshold, 2),
        "main_cue": [list(FEATURE_GROUPS)[i] for i in group_scores[peaks].argmax(axis=1)],
    }).sort_values("timestamp_seconds").reset_index(drop=True)
    return times, scores, threshold, det


import inspect
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------- Look & feel (matches the presentation deck) ----------------
AG = dict(bg="#0E1A27", panel="#14202E", raised="#1A2A3B", line="#26384C", text="#E8EEF3",
          muted="#8FA3B5", teal="#13A89E", orange="#E4572E", amber="#F2B134", blue="#5B7DB1")
CUE_COLORS = {"MFCC (timbre / channel)": AG["blue"], "Level": AG["amber"],
              "Spectral shape": "#A99BF0", "Background noise": AG["orange"]}

_theme_vars = dict(
    body_background_fill=AG["bg"], body_text_color=AG["text"], body_text_color_subdued=AG["muted"],
    background_fill_primary=AG["panel"], background_fill_secondary=AG["raised"],
    block_background_fill=AG["panel"], block_border_color=AG["line"], border_color_primary=AG["line"],
    block_label_background_fill=AG["raised"], block_label_text_color=AG["muted"],
    block_title_text_color=AG["text"], input_background_fill=AG["raised"], input_border_color=AG["line"],
    button_primary_background_fill=AG["teal"], button_primary_background_fill_hover="#0F8F86",
    button_primary_text_color="#FFFFFF", button_secondary_background_fill=AG["raised"],
    button_secondary_text_color=AG["text"], slider_color=AG["teal"], color_accent=AG["teal"],
    table_even_background_fill=AG["panel"], table_odd_background_fill=AG["raised"],
    table_border_color=AG["line"],
)
# Same colours in light and dark mode; keep only the variables this Gradio version knows.
_known = set(inspect.signature(gr.themes.Base.set).parameters)
AG_THEME = gr.themes.Base(
    primary_hue="teal", neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(**{k + sfx: v for k, v in _theme_vars.items() for sfx in ("", "_dark") if k + sfx in _known})

AG_CSS = """
.gradio-container {max-width: 1280px !important; margin: 0 auto !important;}
footer {display: none !important;}
.ag-header svg {flex:none;}
.ag-header {display:flex; align-items:center; gap:16px; padding:8px 4px 4px;}
.ag-header h1 {margin:0; font-size:28px; font-weight:700; letter-spacing:-0.01em; color:#E8EEF3;}
.ag-header p {margin:2px 0 0; color:#8FA3B5; font-size:15px;}
.ag-badge {margin-left:auto; border:1px solid #26384C; color:#F2B134; border-radius:999px;
           padding:4px 12px; font-size:12px; text-transform:uppercase; letter-spacing:.08em;}
.ag-note {color:#8FA3B5; font-size:13px; line-height:1.5; padding:4px 2px;}
.ag-empty {border:1px dashed #26384C; border-radius:14px; padding:40px 24px; text-align:center; color:#8FA3B5;}
.ag-empty b {color:#E8EEF3; display:block; font-size:17px; margin-bottom:6px;}
.ag-verdict {display:flex; align-items:center; gap:12px; border-radius:14px; padding:14px 18px;
             font-size:16px; font-weight:600; margin-bottom:12px;}
.ag-verdict.alert {background:rgba(228,87,46,.12); border:1px solid rgba(228,87,46,.5); color:#FF9B7A;}
.ag-verdict.clear {background:rgba(19,168,158,.12); border:1px solid rgba(19,168,158,.5); color:#5FD6CC;}
.ag-verdict .dot {width:10px; height:10px; border-radius:50%; background:currentColor; flex:none;}
.ag-verdict .file {margin-left:auto; font-weight:400; font-size:13px; color:#8FA3B5;
                   font-family:'IBM Plex Mono',monospace; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:45%;}
.ag-tiles {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px;}
.ag-tile {background:#1A2A3B; border:1px solid #26384C; border-radius:14px; padding:14px 16px;}
.ag-tile .v {font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; color:#E8EEF3; line-height:1.2;}
.ag-tile .v.txt {font-family:'IBM Plex Sans',sans-serif; font-size:20px; padding-top:4px;}
.ag-tile .l {font-size:12px; color:#8FA3B5; text-transform:uppercase; letter-spacing:.06em; margin-top:4px;}
.ag-cards {display:flex; flex-direction:column; gap:10px;}
.ag-card {display:grid; grid-template-columns:110px 1fr auto; align-items:center; gap:16px;
          background:#1A2A3B; border:1px solid #26384C; border-radius:12px; padding:12px 16px;}
.ag-card .ts {font-family:'IBM Plex Mono',monospace; font-size:20px; font-weight:600; color:#E8EEF3;}
.ag-card .ts small {display:block; font-size:11px; color:#8FA3B5; font-weight:400; font-family:'IBM Plex Sans',sans-serif;}
.ag-bar {height:8px; background:#26384C; border-radius:999px; overflow:hidden;}
.ag-bar span {display:block; height:100%; border-radius:999px;}
.ag-card .meta {font-size:12px; color:#8FA3B5; margin-top:6px;}
.ag-chip {font-size:12px; padding:4px 10px; border-radius:999px; border:1px solid; white-space:nowrap;}
@media (max-width: 640px) {
  .ag-card {grid-template-columns:1fr; gap:8px;}
  .ag-verdict {flex-wrap:wrap;} .ag-verdict .file {margin-left:0; max-width:100%;}
  .ag-badge {display:none;}
}
"""

AG_LOGO = ('<svg width="44" height="44" viewBox="0 0 44 44" aria-hidden="true">'
           + "".join(f'<rect x="{3 + i * 5}" y="{22 - h / 2}" width="3" height="{h}" rx="1.5" fill="{AG["orange"] if 3 <= i <= 5 else AG["teal"]}"/>'
                     for i, h in enumerate([10, 20, 30, 16, 36, 24, 14, 22]))
           + "</svg>")
AG_HEADER = f"""<div class="ag-header">{AG_LOGO}<div><h1>AudioGuard</h1>
<p>Audio splicing forensic analyzer — finds where a recording may have been edited, and why</p></div>
<span class="ag-badge">Prototype</span></div>"""
AG_EMPTY = """<div class="ag-empty"><b>No recording analyzed yet</b>
Upload a WAV/MP3, or pick one of the examples, and press <i>Analyze recording</i>.</div>"""


def evidence_figure(y, sr, t, s, thr, det):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.26, 0.42, 0.32],
                        subplot_titles=("Waveform", "Spectrogram", "Discontinuity score D(t)"))
    # waveform as a min/max envelope (fast even for long files)
    hop = max(1, len(y) // 2000)
    yy = y[:hop * (len(y) // hop)].reshape(-1, hop)
    tx = (np.arange(len(yy)) * hop + hop / 2) / sr
    fig.add_trace(go.Scatter(x=np.r_[tx, tx[::-1]], y=np.r_[yy.max(1), yy.min(1)[::-1]], fill="toself",
                             fillcolor=AG["teal"], line=dict(width=0), hoverinfo="skip"), 1, 1)
    # spectrogram (down-sampled in time for the browser)
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)), ref=np.max)
    step = max(1, S_db.shape[1] // 900)
    fig.add_trace(go.Heatmap(
        x=librosa.frames_to_time(np.arange(0, S_db.shape[1], step), sr=sr, hop_length=HOP_LENGTH),
        y=librosa.fft_frequencies(sr=sr, n_fft=FRAME_LENGTH), z=S_db[:, ::step],
        colorscale="Magma", zmin=-80, zmax=0, showscale=False,
        hovertemplate="%{x:.2f} s · %{y:.0f} Hz · %{z:.0f} dB<extra></extra>"), 2, 1)
    # score + threshold
    fig.add_trace(go.Scatter(x=t, y=s, line=dict(color=AG["text"], width=1.3),
                             hovertemplate="%{x:.2f} s · score %{y:.2f}<extra></extra>"), 3, 1)
    fig.add_hline(y=thr, line=dict(color=AG["amber"], dash="dash", width=1), row=3, col=1,
                  annotation_text=f"threshold {thr:.2f}", annotation_font_color=AG["amber"],
                  annotation_position="top left")
    for i, ts in enumerate(det.timestamp_seconds, 1):
        for r in (1, 2, 3):
            fig.add_vline(x=ts, line=dict(color=AG["orange"], width=1.6), row=r, col=1)
        fig.add_annotation(x=ts, y=1.0, yref="paper", text=f"#{i}", showarrow=False, yshift=12,
                           font=dict(color=AG["orange"], size=12))
    fig.update_layout(height=600, margin=dict(l=55, r=20, t=40, b=45), showlegend=False,
                      paper_bgcolor=AG["panel"], plot_bgcolor=AG["panel"], hovermode="closest",
                      font=dict(family="IBM Plex Sans, sans-serif", color=AG["muted"], size=12))
    fig.update_annotations(font_color=AG["text"], selector=dict(text="Waveform"))
    fig.update_xaxes(gridcolor=AG["line"], zeroline=False, range=[0, len(y) / sr])
    fig.update_yaxes(gridcolor=AG["line"], zeroline=False)
    fig.update_yaxes(title_text="Hz", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


def summary_html(name, dur, det, elapsed):
    n = len(det)
    if n:
        verdict = f'<div class="ag-verdict alert"><span class="dot"></span>{n} suspicious boundar{"y" if n == 1 else "ies"} — review the evidence<span class="file">{name}</span></div>'
        cue = det.main_cue.value_counts().idxmax()
    else:
        verdict = f'<div class="ag-verdict clear"><span class="dot"></span>No suspicious boundaries above threshold<span class="file">{name}</span></div>'
        cue = "—"
    cue_style = f' style="color:{CUE_COLORS.get(cue, AG["text"])}"'
    tiles = [(f"{dur:.1f} s", "Duration", "", ""), (str(n), "Flagged boundaries", "", ""),
             (cue.split(" (")[0], "Main cue", " txt", cue_style), (f"{elapsed * 1000:.0f} ms", "Analysis time", "", "")]
    return verdict + '<div class="ag-tiles">' + "".join(
        f'<div class="ag-tile"><div class="v{c}"{st}>{v}</div><div class="l">{l}</div></div>'
        for v, l, c, st in tiles) + "</div>"


def cards_html(det):
    if not len(det):
        return '<div class="ag-empty"><b>Nothing flagged</b>No point in this recording crossed the adaptive threshold.</div>'
    top = det.anomaly_score.max()
    out = []
    for i, r in enumerate(det.itertuples(), 1):
        col = CUE_COLORS.get(r.main_cue, AG["muted"])
        out.append(f"""<div class="ag-card"><div class="ts">{r.timestamp_seconds:.2f} s<small>boundary #{i}</small></div>
<div><div class="ag-bar"><span style="width:{100 * r.anomaly_score / top:.0f}%;background:{col}"></span></div>
<div class="meta">score {r.anomaly_score:.2f} · {r.score_over_threshold:.2f}× threshold</div></div>
<span class="ag-chip" style="color:{col};border-color:{col}">{r.main_cue}</span></div>""")
    return '<div class="ag-cards">' + "".join(out) + "</div>"


def ag_analyze(path, k):
    if not path:
        raise gr.Error("Upload a recording first.")
    t0 = time.perf_counter()
    try:
        y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        raise gr.Error("Could not read that file as audio. Please upload a WAV, MP3, FLAC or OGG recording.")
    if len(y) < 3 * SR:
        raise gr.Error("Recording is too short — please use at least 3 seconds of audio.")
    t, s, thr, det = detect_splices(y, threshold_factor=k)
    elapsed = time.perf_counter() - t0
    name = os.path.basename(path)

    report = det.copy()
    report.insert(0, "file", name)
    report["threshold_factor"] = k
    report["note"] = "Candidate boundary from acoustic discontinuity; not proof of tampering."
    report_path = f"audioguard_report_{os.path.splitext(name)[0]}.csv"
    report.to_csv(report_path, index=False)

    choices = [f"#{i} · {ts:.2f} s · {cue}" for i, (ts, cue) in
               enumerate(zip(det.timestamp_seconds, det.main_cue), 1)]
    first = choices[0] if choices else None
    return (summary_html(name, len(y) / SR, det, elapsed), evidence_figure(y, SR, t, s, thr, det),
            cards_html(det), det, gr.Dropdown(choices=choices, value=first),
            ag_snippet(first, (y, det)), report_path, (y, det))


def ag_snippet(choice, state):
    if not choice or state is None:
        return None
    y, det = state
    ts = float(det.timestamp_seconds.iloc[int(choice.split(" ")[0][1:]) - 1])
    a, b = max(0, int((ts - 1) * SR)), min(len(y), int((ts + 1) * SR))
    return (SR, y[a:b])


_style = dict(theme=AG_THEME, css=AG_CSS)
_blocks_takes_style = "css" in inspect.signature(gr.Blocks.__init__).parameters   # Gradio 4/5 vs 6
with gr.Blocks(title="AudioGuard", **(_style if _blocks_takes_style else {})) as demo:
    state = gr.State()
    gr.HTML(AG_HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=300):
            audio_in = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Recording")
            k_in = gr.Slider(1.5, 6.0, value=BEST_K, step=0.5, label="Detection threshold k",
                             info="Lower = more sensitive, but more false alarms")
            run_btn = gr.Button("Analyze recording", variant="primary", size="lg")
            _ex = sorted(glob.glob("examples/*.wav") + glob.glob("examples/*.mp3"))
            if _ex:
                gr.Examples(label="Example recordings", examples=[[p] for p in _ex],
                            example_labels=[os.path.splitext(os.path.basename(p))[0].replace("_", " ") for p in _ex],
                            inputs=[audio_in])
            gr.HTML('<div class="ag-note">A flagged boundary is a place where the acoustic properties '
                    'change abruptly. It is evidence to examine, not proof of tampering.</div>')
        with gr.Column(scale=8, min_width=320):
            summary_out = gr.HTML(AG_EMPTY)
            with gr.Tabs():
                with gr.Tab("Evidence"):
                    plot_out = gr.Plot(show_label=False)
                with gr.Tab("Boundaries"):
                    cards_out = gr.HTML()
                    table_out = gr.Dataframe(label="Raw values", interactive=False)
                with gr.Tab("Listen"):
                    pick = gr.Dropdown(label="Flagged boundary", choices=[], interactive=True)
                    snippet_out = gr.Audio(label="±1 second around the boundary", interactive=False)
                with gr.Tab("Report"):
                    report_out = gr.File(label="CSV report")

    outputs = [summary_out, plot_out, cards_out, table_out, pick, snippet_out, report_out, state]
    run_btn.click(ag_analyze, [audio_in, k_in], outputs)
    pick.change(ag_snippet, [pick, state], snippet_out)

if __name__ == "__main__":
    demo.launch(**({} if _blocks_takes_style else _style))
