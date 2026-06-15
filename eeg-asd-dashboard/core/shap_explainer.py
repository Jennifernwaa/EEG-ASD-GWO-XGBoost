"""
SHAP Explainability Module
============================
Computes and visualizes SHAP values for individual predictions.
Uses TreeSHAP for exact Shapley value computation on XGBoost.
"""

import numpy as np
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib"),
)
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Optional
import io


class SHAPExplainer:
    """
    SHAP explainability for the GWO-XGBoost ASD classifier.
    
    Uses TreeSHAP (Lundberg et al., 2019) for exact computation.
    Provides both global and local (per-subject) explanations.
    """
    
    def __init__(self, model, feature_names: list):
        """Create a TreeSHAP explainer for the trained model."""
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)
    
    def explain_instance(self, features_scaled: np.ndarray,
                         feature_values: Optional[np.ndarray] = None) -> Dict:
        """
        Compute SHAP values for a single prediction.
        
        Parameters
        ----------
        features_scaled : np.ndarray
            Scaled feature vector [1, n_features]
            
        Returns
        -------
        dict : SHAP values and interpretation
        """
        if features_scaled.ndim == 1:
            features_scaled = features_scaled.reshape(1, -1)
        if feature_values is not None:
            feature_values = np.asarray(feature_values).flatten()
        
        shap_values = self.explainer.shap_values(features_scaled)
        
        # Handle binary output format
        if isinstance(shap_values, list):
            sv = shap_values[1]  # ASD class
        else:
            sv = shap_values
        
        sv_flat = sv.flatten()
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]
        
        # Sort by absolute SHAP value
        abs_shap = np.abs(sv_flat)
        sorted_idx = np.argsort(abs_shap)[::-1]
        
        # Top contributors
        top_features = []
        for idx in sorted_idx[:10]:
            top_features.append({
                'feature': self.feature_names[idx],
                'shap_value': float(sv_flat[idx]),
                'direction': 'ASD' if sv_flat[idx] > 0 else 'Control',
                'magnitude': float(abs_shap[idx]),
                'value': (
                    float(feature_values[idx])
                    if feature_values is not None and idx < len(feature_values)
                    else None
                )
            })
        
        return {
            'shap_values': sv_flat,
            'base_value': float(base_value),
            'top_features': top_features,
            'feature_names': self.feature_names,
            'explanation_object': shap.Explanation(
                values=sv_flat,
                base_values=base_value,
                data=features_scaled.flatten(),
                feature_names=self.feature_names
            )
        }
    
    def create_waterfall_plotly(self, explanation: Dict) -> go.Figure:
        """
        Create an interactive horizontal SHAP contribution plot.

        Positive bars indicate features that push the prediction toward ASD;
        negative bars indicate features that push it toward Control.
        """
        top_feats = explanation['top_features'][:10]
        
        features = [f['feature'] for f in reversed(top_feats)]
        values = [f['shap_value'] for f in reversed(top_feats)]
        colors = ['#EF5350' if v > 0 else '#66BB6A' for v in values]
        
        fig = go.Figure(go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            text=[f'{v:+.3f}' for v in values],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Feature Contributions",
            xaxis_title="SHAP value: Control direction to ASD direction",
            yaxis_title="",
            template="plotly_white",
            height=460,
            margin=dict(l=170, r=70, t=55, b=60),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=12, color="#172033")
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
    
    def create_force_plot_html(self, explanation: Dict) -> str:
        """
        Generate SHAP's native force plot as embeddable HTML.

        The returned string includes SHAP's JavaScript plus the rendered plot,
        so Streamlit can display the local explanation directly.
        """
        exp = explanation['explanation_object']
        force_plot = shap.plots.force(exp, show=False, matplotlib=False)
        return shap.getjs() + force_plot.html()
