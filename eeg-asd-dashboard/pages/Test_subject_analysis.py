"""
Test Subject Analysis Page
============================
Interactive analysis of held-out test subjects.
Allows selecting a test subject and viewing:
- Prediction with confidence
- SHAP explanation
- Feature analysis
- Biomarker interpretation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
import sys
sys.path.append('..')

from core.model_inference import ASDClassifier
from core.shap_explainer import SHAPExplainer
from core.biomarker_interpreter import BiomarkerInterpreter
from core.feature_extraction import GWO_SELECTED_FEATURES

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Test Subject Analysis", page_icon="🔬", layout="wide")

st.markdown("""
<h1 style='text-align: center;'>
    🔬 Test Subject Analysis
</h1>
<p style='text-align: center; color: #B0BEC5;'>
    Interactive prediction and explainability for held-out test subjects
</p>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================
@st.cache_resource
def load_model():
    return ASDClassifier(model_dir="models/")

@st.cache_data
def load_data():
    data_dir = Path("data")
    feature_file = data_dir / "features_sheffield_v3 - features_sheffield_v3.csv.csv"
    if not feature_file.exists():
        matches = sorted(data_dir.glob("features_sheffield_v3*.csv*"))
        if not matches:
            raise FileNotFoundError("No Sheffield feature CSV found in the data folder.")
        feature_file = matches[0]
    df = pd.read_csv(feature_file)
    with open("data/test_subjects.json", 'r') as f:
        test_info = json.load(f)
    return df, test_info

try:
    classifier = load_model()
    df, test_info = load_data()
    test_subjects = test_info['subjects']
    test_labels = test_info['labels']
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please ensure model artifacts and data files are in the correct directories.")
    st.stop()

# ============================================================
# SUBJECT SELECTION
# ============================================================
st.markdown("---")
col_select, col_info = st.columns([1, 2])

with col_select:
    st.markdown("### 📋 Select Test Subject")
    
    # Display test subjects with their true labels
    subject_options = []
    for i, (subj, label) in enumerate(zip(test_subjects, test_labels)):
        label_str = "🔴 ASD" if label == 1 else "🟢 Control"
        subject_options.append(f"{subj} ({label_str})")
    
    selected_idx = st.selectbox(
        "Choose a subject from the 20% held-out test set:",
        range(len(subject_options)),
        format_func=lambda x: subject_options[x]
    )
    
    selected_subject = test_subjects[selected_idx]
    true_label = test_labels[selected_idx]
    
    st.markdown(f"""
    **Selected:** `{selected_subject}`  
    **True Class:** {'ASD' if true_label == 1 else 'Control'}  
    **Set:** Held-out Test (never seen during training)
    """)

with col_info:
    st.markdown("### ℹ️ Testing Protocol")
    st.markdown("""
    - **Total subjects:** 56 (28 ASD, 28 Control)
    - **Training set:** 44 subjects (80%)
    - **Test set:** 12 subjects (20%) — **selected before any model development**
    - **Evaluation:** Subject-level (no data leakage)
    - **This page:** Replays prediction on individual test subjects
    """)

# ============================================================
# RUN PREDICTION
# ============================================================
st.markdown("---")
st.markdown("## 🎯 Prediction Results")

# Get subject features
subject_row = df[df['Subject'] == selected_subject]
if subject_row.empty:
    st.error(f"Subject {selected_subject} not found in feature matrix.")
    st.stop()

# Extract features (all 79)
feature_cols = [c for c in df.columns if c not in ['Subject', 'Label']]
features_79 = subject_row[feature_cols].values.flatten()

# Run prediction
result = classifier.predict(features_79, feature_names=feature_cols)

# ============================================================
# PREDICTION PANEL
# ============================================================
pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 1])

with pred_col1:
    # Prediction badge
    if result['label'] == 'ASD':
        st.markdown("""
        <div style='text-align:center; padding:20px; background:#1e2130; 
             border-radius:12px; border:2px solid #EF5350;'>
            <h3 style='color:#EF5350;'>🧠 Predicted: ASD</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center; padding:20px; background:#1e2130; 
             border-radius:12px; border:2px solid #66BB6A;'>
            <h3 style='color:#66BB6A;'>🧠 Predicted: Control</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Correctness
    correct = (result['prediction'] == true_label)
    if correct:
        st.success("✅ Correct prediction")
    else:
        st.error("❌ Misclassification")

with pred_col2:
    # Probability gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=result['probability_asd'] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ASD Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#EF5350" if result['probability_asd'] > 0.5 else "#66BB6A"},
            'steps': [
                {'range': [0, 30], 'color': "#1B5E20"},
                {'range': [30, 50], 'color': "#F9A825"},
                {'range': [50, 70], 'color': "#E65100"},
                {'range': [70, 100], 'color': "#B71C1C"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig_gauge.update_layout(
        height=250, 
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

with pred_col3:
    # Confidence and metrics
    st.markdown("### 📊 Metrics")
    st.metric("Confidence", f"{result['confidence']*100:.1f}%")
    st.metric("P(ASD)", f"{result['probability_asd']:.3f}")
    st.metric("P(Control)", f"{result['probability_control']:.3f}")
    
    if result['low_confidence']:
        st.warning("⚠️ Low confidence — near decision boundary")

# ============================================================
# SHAP EXPLANATION
# ============================================================
st.markdown("---")
st.markdown("## 🔍 SHAP Explainability")

explainer = SHAPExplainer(classifier.model, GWO_SELECTED_FEATURES)
shap_result = explainer.explain_instance(
    result['features_scaled'],
    result.get('features_raw')
)

# SHAP waterfall plot
shap_col1, shap_col2 = st.columns([2, 1])

with shap_col1:
    fig_waterfall = explainer.create_waterfall_plotly(shap_result)
    st.plotly_chart(fig_waterfall, use_container_width=True)

with shap_col2:
    st.markdown("### Top Contributing Features")
    for i, feat in enumerate(shap_result['top_features'][:5], 1):
        direction_icon = "🔴" if feat['direction'] == 'ASD' else "🟢"
        st.markdown(
            f"**{i}.** {direction_icon} `{feat['feature']}`  \n"
            f"   SHAP: {feat['shap_value']:+.4f} → {feat['direction']}"
        )

# ============================================================
# BIOMARKER INTERPRETATION
# ============================================================
st.markdown("---")
st.markdown("## 🧬 Biomarker Interpretation")

interpreter = BiomarkerInterpreter()
interpretation = interpreter.interpret(result, shap_result)

# Summary card
st.markdown(f"""
<div style='background:#1e2130; padding:20px; border-radius:12px; 
     border-left:4px solid #4FC3F7; margin-bottom:20px;'>
    <h4 style='color:#4FC3F7;'>Clinical Summary</h4>
    <p style='color:#E0E0E0; font-size:1.05rem;'>{interpretation['summary']}</p>
</div>
""", unsafe_allow_html=True)

if interpretation['confidence_note']:
    st.warning(interpretation['confidence_note'])

if interpretation.get('biomarker_profile'):
    st.info(interpretation['biomarker_profile'])

domain_summary = interpretation.get('domain_summary', {})
if domain_summary:
    st.markdown("### Domain-Level Evidence")
    cols = st.columns(len(domain_summary))
    for col, (domain, detail) in zip(cols, domain_summary.items()):
        with col:
            st.metric(domain, f"{detail.get('proportion', 0) * 100:.0f}%")
            st.caption(detail.get('text', ''))

# Per-feature interpretations
st.markdown("### 📋 Feature-Level Findings")
for interp in interpretation['feature_interpretations']:
    with st.expander(f"{'🔴' if interp['direction'] == 'ASD' else '🟢'} "
                     f"{interp['display_name']} ({interp['domain']})"):
        st.markdown(f"**Explanation:** {interp['explanation']}")
        if not interp.get('is_primary_biomarker', True):
            st.caption("Supporting / low-weight feature in the current biomarker model.")
        st.markdown(f"**SHAP Value:** {interp['shap_value']:+.4f}")
        if interp['z_score'] is not None:
            z = interp['z_score']
            st.markdown(f"**Z-score vs training set:** {z:+.2f} "
                       f"({'⚠️ Abnormal' if abs(z) > 2 else '✓ Normal range'})")

# ============================================================
# FEATURE COMPARISON
# ============================================================
st.markdown("---")
st.markdown("## 📊 Feature Comparison")

# Get GWO feature values for this subject
gwo_feature_idx = [feature_cols.index(f) for f in GWO_SELECTED_FEATURES 
                   if f in feature_cols]
subject_gwo_features = features_79[gwo_feature_idx]

# Get training set statistics
train_df = df[~df['Subject'].isin(test_subjects)]
train_asd = train_df[train_df['Label'] == 'ASD'][GWO_SELECTED_FEATURES].mean()
train_control = train_df[train_df['Label'] == 'Control'][GWO_SELECTED_FEATURES].mean()

# Radar chart comparison
fig_radar = go.Figure()

# Normalize for radar chart
all_vals = np.concatenate([subject_gwo_features, train_asd.values, train_control.values])
min_val, max_val = np.nanmin(all_vals), np.nanmax(all_vals)
normalize = lambda x: (x - min_val) / (max_val - min_val + 1e-10)

fig_radar.add_trace(go.Scatterpolar(
    r=normalize(subject_gwo_features),
    theta=GWO_SELECTED_FEATURES,
    fill='toself',
    name=f'Subject ({result["label"]})',
    line_color='#4FC3F7'
))
fig_radar.add_trace(go.Scatterpolar(
    r=normalize(train_asd.values),
    theta=GWO_SELECTED_FEATURES,
    fill='toself',
    name='ASD Average',
    line_color='#EF5350',
    opacity=0.5
))
fig_radar.add_trace(go.Scatterpolar(
    r=normalize(train_control.values),
    theta=GWO_SELECTED_FEATURES,
    fill='toself',
    name='Control Average',
    line_color='#66BB6A',
    opacity=0.5
))

fig_radar.update_layout(
    polar=dict(bgcolor='#1e2130'),
    template="plotly_dark",
    title="Feature Profile: Subject vs Group Averages",
    height=500
)
st.plotly_chart(fig_radar, use_container_width=True)
