from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.signal import welch

from core.biomarker_interpreter import BiomarkerInterpreter
from core.eeg_simulator import EEGSimulator
from core.feature_extraction import EEGFeatureExtractor
from core.model_inference import ASDClassifier
from core.shap_explainer import SHAPExplainer


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FEATURE_FILE = DATA_DIR / "features_sheffield_v3 - features_sheffield_v3.csv.csv"


st.set_page_config(
    page_title="EEG-ASD Dashboard",
    page_icon="EEG",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
        :root {
            --ink: #172033;
            --muted: #71809a;
            --line: #dbe7f2;
            --card: #ffffff;
            --bg: #f6fbff;
            --blue: #2f86d7;
            --green: #45ad78;
            --green-soft: #e8faef;
            --red: #d94b62;
            --red-soft: #fff0f3;
            --yellow-soft: #fffbe4;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 0%, rgba(47, 134, 215, 0.09), transparent 28rem),
                linear-gradient(180deg, #fbfdff 0%, var(--bg) 100%);
            color: var(--ink);
        }

        .main .block-container {
            max-width: 1380px;
            padding: 1.1rem 1.5rem 2.5rem;
        }

        h1, h2, h3 {
            color: var(--ink);
            letter-spacing: 0;
        }

        h1 {
            font-size: 1.8rem;
            margin-bottom: 0.2rem;
        }

        h2 {
            font-size: 1.45rem;
            margin-top: 1.2rem;
        }

        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 18px 14px;
            box-shadow: 0 14px 34px rgba(23, 32, 51, 0.06);
        }

        div[data-testid="stMetric"] label {
            color: var(--ink);
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            color: #111827;
            font-size: 2.1rem;
        }

        .card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.05rem 1.2rem;
            box-shadow: 0 14px 34px rgba(23, 32, 51, 0.06);
            min-height: 100%;
        }

        .card-title {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .big-decision {
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1.15;
            margin: 0.35rem 0 0.55rem;
        }

        .decision-control {
            color: var(--blue);
        }

        .decision-asd {
            color: var(--red);
        }

        .subject-name {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }

        .muted {
            color: var(--muted);
            font-weight: 650;
        }

        .pill {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.32rem 0.72rem;
            font-size: 0.78rem;
            font-weight: 800;
            margin: 0.2rem 0;
        }

        .pill-control {
            color: var(--green);
            background: var(--green-soft);
            border: 1px solid #c8f0db;
        }

        .pill-asd {
            color: var(--red);
            background: var(--red-soft);
            border: 1px solid #ffd3dc;
        }

        .status-ok {
            color: var(--green);
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.7rem;
        }

        .status-bad {
            color: var(--red);
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.7rem;
        }

        .notice {
            background: var(--yellow-soft);
            border: 1px solid #fbf4be;
            border-radius: 8px;
            color: #2f3b4f;
            font-weight: 750;
            padding: 1rem 1.15rem;
            margin: 0.7rem 0 1.1rem;
        }

        div[data-testid="stSelectbox"] label {
            color: var(--ink);
            font-weight: 700;
        }

        div[data-baseweb="select"] > div {
            background: #ffffff;
            border-color: #63a8e9;
            border-radius: 8px;
        }

        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid #cfdbe7;
            border-radius: 8px;
            box-shadow: 0 10px 26px rgba(23, 32, 51, 0.04);
        }

        [data-testid="stExpander"] summary {
            color: var(--ink);
            font-weight: 800;
        }

        #MainMenu, footer, header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_classifier():
    return ASDClassifier(model_dir=str(MODEL_DIR))


@st.cache_resource
def load_explainer(_model, feature_names):
    return SHAPExplainer(_model, list(feature_names))


@st.cache_data
def load_dashboard_data():
    if not FEATURE_FILE.exists():
        matches = sorted(DATA_DIR.glob("features_sheffield_v3*.csv*"))
        if not matches:
            raise FileNotFoundError("No Sheffield feature CSV found in the data folder.")
        feature_file = matches[0]
    else:
        feature_file = FEATURE_FILE

    df = pd.read_csv(feature_file)
    with open(DATA_DIR / "test_subjects.json", "r") as f:
        test_info = json.load(f)
    with open(MODEL_DIR / "model_metadata.json", "r") as f:
        metadata = json.load(f)
    with open(DATA_DIR / "shap_expected_value.json", "r") as f:
        shap_expected = json.load(f)["expected_value"]
    shap_test = np.load(DATA_DIR / "shap_values_test.npy")

    return df, test_info, metadata, feature_file.name, shap_test, shap_expected


def label_name(label):
    if isinstance(label, str):
        return "ASD" if label.lower() == "asd" else "Control"
    return "ASD" if int(label) == 1 else "Control"


def label_int(label):
    if isinstance(label, str):
        return 1 if label.lower() == "asd" else 0
    return int(label)


def badge(label):
    cls = "pill-asd" if label == "ASD" else "pill-control"
    return f"<span class='pill {cls}'>{label}</span>"


def card_html(title, body, extra=""):
    title_html = f"<div class='card-title'>{title}</div>" if title else ""
    html = f"<div class='card'>{title_html}{body.strip()}{extra.strip()}</div>"
    st.markdown(html, unsafe_allow_html=True)


def feature_card(idx, feat):
    value_html = ""
    if feat.get("value") is not None:
        value_html = f"<div class='muted'>Feature value: {feat['value']:.4g}</div>"
    st.markdown(
        (
            "<div class='card'>"
            f"{badge(feat['direction'])}"
            f"<span style='font-weight:850; font-size:1.02rem; margin-left:0.35rem;'>{idx}. {feat['feature']}</span>"
            f"<div class='muted' style='margin-top:0.55rem;'>SHAP: {feat['shap_value']:+.4f}</div>"
            f"{value_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def probability_gauge(probability_asd, predicted_label):
    value = probability_asd * 100
    bar_color = "#d94b62" if predicted_label == "ASD" else "#46aa78"
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": "ASD Probability (%)", "font": {"size": 22, "color": "#172033"}},
            number={"font": {"size": 64, "color": "#111827"}, "suffix": ""},
            gauge={
                "shape": "angular",
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#6f7d91"},
                "bar": {"color": bar_color, "thickness": 0.22},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#8d96a6",
                "steps": [
                    {"range": [0, 40], "color": "#eaf7ef"},
                    {"range": [40, 60], "color": "#fff3cc"},
                    {"range": [60, 100], "color": "#fde4ea"},
                ],
                "threshold": {
                    "line": {"color": "#172033", "width": 4},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=295,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=16, r=16, t=42, b=16),
        font=dict(color="#172033", family="Arial"),
    )
    return fig


def feature_comparison_chart(subject_values, asd_values, control_values, feature_names):
    all_vals = np.concatenate([subject_values, asd_values, control_values]).astype(float)
    min_val, max_val = np.nanmin(all_vals), np.nanmax(all_vals)
    scale = max(max_val - min_val, 1e-10)

    def normalize(values):
        return (np.asarray(values, dtype=float) - min_val) / scale

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=normalize(subject_values),
            theta=feature_names,
            fill="toself",
            name="Selected subject",
            line_color="#2f86d7",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=normalize(asd_values),
            theta=feature_names,
            fill="toself",
            name="ASD average",
            line_color="#d94b62",
            opacity=0.55,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=normalize(control_values),
            theta=feature_names,
            fill="toself",
            name="Control average",
            line_color="#45ad78",
            opacity=0.55,
        )
    )
    fig.update_layout(
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="#ffffff", radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        margin=dict(l=30, r=30, t=50, b=20),
    )
    return fig


@st.cache_data
def simulate_eeg(condition, noise_level, duration, seed):
    simulator = EEGSimulator(duration=duration, n_channels=32, seed=seed)
    eeg = simulator.generate(condition=condition.lower(), noise_level=noise_level)
    extractor = EEGFeatureExtractor(srate=eeg["srate"])
    features = extractor.extract(eeg["data"][np.newaxis, :, :], eeg["channel_names"])
    return eeg, features, extractor.feature_names


def waveform_chart(eeg, max_channels=8):
    channels = eeg["channel_names"][:max_channels]
    time = eeg["time"]
    fig = go.Figure()
    for idx, channel in enumerate(channels):
        offset = idx * 55
        microvolts = eeg["data"][idx] * 1e6 + offset
        fig.add_trace(
            go.Scatter(
                x=time,
                y=microvolts,
                mode="lines",
                name=channel,
                line=dict(width=1.2),
            )
        )
    fig.update_layout(
        height=330,
        title="Synthetic EEG Waveform",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude plus channel offset (uV)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=50, r=20, t=55, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(gridcolor="#e7eef6")
    fig.update_yaxes(gridcolor="#edf3f8")
    return fig


def spectrum_chart(eeg):
    psds = []
    for signal in eeg["data"]:
        freqs, pxx = welch(signal, fs=eeg["srate"], nperseg=min(512, len(signal)))
        psds.append(pxx)
    mean_psd = np.mean(psds, axis=0)
    mask = (freqs >= 1) & (freqs <= 40)
    fig = go.Figure(
        go.Scatter(
            x=freqs[mask],
            y=10 * np.log10(mean_psd[mask] + 1e-20),
            mode="lines",
            line=dict(color="#2f86d7", width=2.4),
        )
    )
    fig.update_layout(
        height=330,
        title="Average Power Spectrum",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power (dB)",
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=50, r=20, t=55, b=45),
    )
    fig.update_xaxes(gridcolor="#e7eef6")
    fig.update_yaxes(gridcolor="#edf3f8")
    return fig


def sanitize_features(features, names, training_stats):
    clean = np.asarray(features, dtype=float).copy()
    for idx, value in enumerate(clean):
        if np.isfinite(value):
            continue
        stats = training_stats.get(names[idx], {}) if training_stats else {}
        clean[idx] = stats.get("mean", 0.0)
    return clean


def shap_result_from_saved(shap_values, feature_names, expected_value, feature_values=None):
    sv_flat = np.asarray(shap_values, dtype=float).flatten()
    if feature_values is not None:
        feature_values = np.asarray(feature_values, dtype=float).flatten()
    sorted_idx = np.argsort(np.abs(sv_flat))[::-1]
    top_features = []
    for idx in sorted_idx[:10]:
        top_features.append(
            {
                "feature": feature_names[idx],
                "shap_value": float(sv_flat[idx]),
                "direction": "ASD" if sv_flat[idx] > 0 else "Control",
                "magnitude": float(abs(sv_flat[idx])),
                "value": (
                    float(feature_values[idx])
                    if feature_values is not None and idx < len(feature_values)
                    else None
                ),
            }
        )
    return {
        "shap_values": sv_flat,
        "base_value": float(expected_value),
        "top_features": top_features,
        "feature_names": feature_names,
    }


def waterfall_from_saved(explanation):
    top_feats = explanation["top_features"][:10]
    features = [f["feature"] for f in reversed(top_feats)]
    values = [f["shap_value"] for f in reversed(top_feats)]
    colors = ["#d94b62" if v > 0 else "#45ad78" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Feature Contributions",
        xaxis_title="SHAP value: Control direction to ASD direction",
        yaxis_title="",
        template="plotly_white",
        height=460,
        margin=dict(l=170, r=70, t=55, b=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=12, color="#172033"),
    )
    fig.update_xaxes(
        zeroline=True,
        zerolinecolor="#768296",
        gridcolor="#e7eef6",
        linecolor="#dbe7f2",
    )
    fig.update_yaxes(gridcolor="#edf3f8", linecolor="#dbe7f2")
    fig.add_vline(x=0, line_dash="dash", line_color="#768296", opacity=0.65)
    return fig


try:
    classifier = load_classifier()
    df, test_info, metadata, feature_source, shap_test, shap_expected = load_dashboard_data()
except Exception as exc:
    st.error(f"Dashboard could not load: {exc}")
    st.stop()

selected_features = classifier.selected_features
feature_cols = [c for c in df.columns if c not in ["Subject", "Label"]]
test_subjects = test_info["subjects"]
test_labels = test_info["labels"]

def render_engine_status():
    if classifier.model is not None:
        card_html(
            "Inference Engine",
            "<div class='muted'>Native XGBoost model loaded from <b>models/gwo_xgboost_model.pkl</b>.</div>",
        )
    else:
        card_html(
            "Inference Engine",
            "<div class='muted'>Native XGBoost unavailable. Subject tab can replay saved held-out predictions.</div>",
        )


def render_subject_analysis():
    top_left, top_right = st.columns([1.05, 1.45], gap="large")

    with top_left:
        st.subheader("Subject Selection")
        options = [
            f"{subject} ({label_name(label)})"
            for subject, label in zip(test_subjects, test_labels)
        ]
        selected_idx = st.selectbox(
            "Choose a held-out subject",
            options=range(len(options)),
            format_func=lambda idx: options[idx],
            label_visibility="visible",
        )
        selected_subject = test_subjects[selected_idx]
        true_label = label_name(test_labels[selected_idx])

        card_html(
            "Selected Subject",
            f"""
            <div class="subject-name">{selected_subject}</div>
            {badge(true_label)}
            <div class="muted" style="margin-top:0.45rem;">Feature source: {feature_source}</div>
            """,
        )

    with top_right:
        st.subheader("Testing Protocol")
        card_html(
            "Subject-Level Evaluation",
            f"""
            <div class="muted" style="font-size:0.96rem; line-height:1.75;">
                {metadata.get("n_subjects_train", test_info.get("n_train", 44))} training subjects and
                {metadata.get("n_subjects_test", test_info.get("n_test", 12))} held-out test subjects.
                The test split was separated before model evaluation, so this page replays predictions
                on subjects not used for training.
            </div>
            """,
        )

    subject_row = df[df["Subject"] == selected_subject]
    if subject_row.empty:
        st.error(f"Subject {selected_subject} was not found in the feature matrix.")
        st.stop()

    features_79 = subject_row[feature_cols].to_numpy(dtype=float).flatten()
    try:
        result = classifier.predict(features_79, feature_names=feature_cols)
        using_saved_predictions = False
    except RuntimeError:
        selected_idx_by_name = [feature_cols.index(f) for f in selected_features if f in feature_cols]
        features_sel = features_79[selected_idx_by_name]
        features_scaled = classifier.scaler.transform(features_sel.reshape(1, -1)).flatten()
        probability_asd = float(test_info["probabilities_asd"][selected_idx])
        probability_control = float(test_info["probabilities_control"][selected_idx])
        prediction = int(test_info["predictions"][selected_idx])
        confidence = max(probability_asd, probability_control)
        result = {
            "prediction": prediction,
            "label": "ASD" if prediction == 1 else "Control",
            "probability_asd": probability_asd,
            "probability_control": probability_control,
            "confidence": confidence,
            "features_scaled": features_scaled,
            "features_raw": features_sel,
            "z_scores": classifier._compute_z_scores(features_sel),
            "low_confidence": confidence < 0.65,
        }
        using_saved_predictions = True

    predicted_label = result["label"]
    is_correct = result["prediction"] == label_int(true_label)

    st.subheader("Prediction Results")
    pred_col, gauge_col, metrics_col = st.columns([1.0, 1.38, 1.0], gap="large")

    with pred_col:
        status_cls = "status-ok" if is_correct else "status-bad"
        status_text = "Correct prediction" if is_correct else "Misclassification"
        decision_cls = "decision-asd" if predicted_label == "ASD" else "decision-control"
        predicted_pill = badge(predicted_label).replace(predicted_label, "Predicted class")
        card_html(
            "Model Decision",
            f"""
            <div class="big-decision {decision_cls}">{predicted_label}</div>
            {predicted_pill}
            <div class="{status_cls}">{status_text}</div>
            """,
        )

    with gauge_col:
        st.plotly_chart(
            probability_gauge(result["probability_asd"], predicted_label),
            use_container_width=True,
        )

    with metrics_col:
        st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        st.metric("P(ASD)", f"{result['probability_asd']:.3f}")
        st.metric("P(Control)", f"{result['probability_control']:.3f}")
        if result["low_confidence"]:
            st.markdown(
                "<div class='notice'>Low confidence: this subject is near the decision boundary.</div>",
                unsafe_allow_html=True,
            )

    st.subheader("SHAP Explainability")
    if classifier.model is not None:
        explainer = load_explainer(classifier.model, tuple(selected_features))
        shap_result = explainer.explain_instance(
            result["features_scaled"],
            result.get("features_raw"),
        )
        shap_fig = explainer.create_waterfall_plotly(shap_result)
    else:
        shap_result = shap_result_from_saved(
            shap_test[selected_idx],
            selected_features,
            shap_expected,
            result.get("features_raw"),
        )
        shap_fig = waterfall_from_saved(shap_result)
        if using_saved_predictions:
            st.caption("Using saved held-out predictions and SHAP values because the native XGBoost runtime is unavailable.")

    shap_col, top_col = st.columns([1.4, 1.0], gap="large")
    with shap_col:
        st.plotly_chart(shap_fig, use_container_width=True)

    with top_col:
        st.markdown("### Top Contributing Features")
        for idx, feat in enumerate(shap_result["top_features"][:5], 1):
            feature_card(idx, feat)

    st.subheader("Biomarker Interpretation")
    interpreter = BiomarkerInterpreter()
    interpretation = interpreter.interpret(result, shap_result)

    card_html(
        "Clinical Summary",
        f"<div style='font-weight:750; line-height:1.6;'>{interpretation['summary']}</div>",
    )

    if interpretation.get("biomarker_profile"):
        card_html(
            "Biomarker Profile",
            f"<div style='font-weight:700; line-height:1.6;'>{interpretation['biomarker_profile']}</div>",
        )

    domain_summary = interpretation.get("domain_summary", {})
    if domain_summary:
        st.markdown("### Domain-Level Evidence")
        domain_cols = st.columns(len(domain_summary))
        for col, (domain, detail) in zip(domain_cols, domain_summary.items()):
            with col:
                proportion = detail.get("proportion", 0) * 100
                card_html(
                    domain,
                    f"""
                    <div class="big-decision decision-control" style="font-size:1.6rem;">{proportion:.0f}%</div>
                    <div class="muted" style="line-height:1.55;">{detail.get('text', '')}</div>
                    """,
                )

    if interpretation["confidence_note"]:
        clean_note = interpretation["confidence_note"].replace("⚠️ ", "").replace("ℹ️ ", "")
        st.markdown(f"<div class='notice'>{clean_note}</div>", unsafe_allow_html=True)

    for interp in interpretation["feature_interpretations"]:
        title = f"{interp['display_name']} ({interp['domain']})"
        with st.expander(title, expanded=True):
            st.markdown(badge(interp["direction"]), unsafe_allow_html=True)
            if not interp.get("is_primary_biomarker", True):
                st.caption("Supporting / low-weight feature in the current biomarker model.")
            st.write(interp["explanation"])
            st.markdown(f"**SHAP value:** `{interp['shap_value']:+.4f}`")
            if interp["z_score"] is not None:
                z_score = interp["z_score"]
                range_text = "outside training range" if abs(z_score) > 2 else "within training range"
                st.markdown(f"**Z-score vs training set:** `{z_score:+.2f}` ({range_text})")

    st.subheader("Feature Comparison")
    gwo_indices = [feature_cols.index(f) for f in selected_features if f in feature_cols]
    subject_gwo_features = features_79[gwo_indices]
    train_df = df[~df["Subject"].isin(test_subjects)].copy()
    train_df["_label_name"] = train_df["Label"].map(label_name)
    train_asd = train_df[train_df["_label_name"] == "ASD"][selected_features].mean().values
    train_control = train_df[train_df["_label_name"] == "Control"][selected_features].mean().values
    st.plotly_chart(
        feature_comparison_chart(subject_gwo_features, train_asd, train_control, selected_features),
        use_container_width=True,
    )


def render_simulator():
    st.subheader("EEG Simulator")
    sim_col, sim_result_col = st.columns([1.15, 1.0], gap="large")

    with sim_col:
        condition = st.radio(
            "Synthetic profile",
            ["Control", "ASD"],
            horizontal=True,
        )
        noise_level = st.slider("Noise level", 0.05, 0.8, 0.30, 0.05)
        duration = st.slider("Signal duration (seconds)", 4.0, 12.0, 8.0, 1.0)
        seed = st.number_input("Simulation seed", min_value=1, max_value=9999, value=42, step=1)

    eeg, sim_features, sim_feature_names = simulate_eeg(
        condition,
        float(noise_level),
        float(duration),
        int(seed),
    )
    sim_features = sanitize_features(sim_features, sim_feature_names, classifier.training_stats)

    try:
        sim_result = classifier.predict(sim_features, feature_names=sim_feature_names)
        sim_error = None
    except Exception as exc:
        sim_result = None
        sim_error = str(exc)

    with sim_result_col:
        if sim_result is None:
            st.error(f"Simulator prediction failed: {sim_error}")
        else:
            sim_decision_cls = "decision-asd" if sim_result["label"] == "ASD" else "decision-control"
            card_html(
                "Simulated Model Decision",
                f"""
                <div class="big-decision {sim_decision_cls}">{sim_result['label']}</div>
                {badge(sim_result['label']).replace(sim_result['label'], "Predicted class")}
                <div class="muted" style="margin-top:0.75rem;">
                    P(ASD): {sim_result['probability_asd']:.3f}<br>
                    P(Control): {sim_result['probability_control']:.3f}<br>
                    Confidence: {sim_result['confidence'] * 100:.1f}%
                </div>
                """,
            )

    st.subheader("Extracted Feature Vector")
    selected_feature_set = set(selected_features)
    feature_df = pd.DataFrame(
        {
            "feature": sim_feature_names,
            "value": sim_features,
            "used_by_model": [name in selected_feature_set for name in sim_feature_names],
        }
    )
    feature_df["value"] = feature_df["value"].map(lambda value: f"{value:.6g}")
    st.dataframe(
        feature_df[feature_df["used_by_model"]].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )
    with st.expander("Show all 79 extracted features"):
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

    wave_col, psd_col = st.columns(2, gap="large")
    with wave_col:
        st.plotly_chart(waveform_chart(eeg), use_container_width=True)
    with psd_col:
        st.plotly_chart(spectrum_chart(eeg), use_container_width=True)


st.title("EEG-ASD Subject Dashboard")
render_engine_status()
subject_tab, simulator_tab = st.tabs(["Subject Analysis", "EEG Simulator"])

with subject_tab:
    render_subject_analysis()

with simulator_tab:
    render_simulator()
