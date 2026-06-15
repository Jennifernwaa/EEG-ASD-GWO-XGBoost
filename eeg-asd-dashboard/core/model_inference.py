"""
Model Inference Module
=======================
Loads the GWO-XGBoost trained model and performs predictions.
Handles feature selection (13 GWO features) and scaling.
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import RobustScaler
from typing import Dict, Tuple, Optional
from pathlib import Path


class ASDClassifier:
    """
    Production inference wrapper for the GWO-XGBoost ASD classifier.
    
    Loads:
    - Trained XGBoost model (with GWO-tuned hyperparameters)
    - RobustScaler (fitted on training set)
    - GWO-selected feature indices
    - Training set statistics (for z-score comparison)
    """
    
    def __init__(self, model_dir: str = "models/"):
        """
        Initialize the classifier wrapper and load saved model artifacts.

        The dashboard creates this once, then reuses the loaded model, scaler,
        feature list, and training statistics for subject-level predictions.
        """
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.selected_features = None
        self.feature_names_all = None
        self.training_stats = None
        self.load_error = None
        self._load_artifacts()
    
    def _load_artifacts(self):
        """
        Load the serialized model, scaler, selected features, and stats files.

        If XGBoost cannot be loaded on the host machine, the wrapper records
        the error so the app can fall back to saved predictions.
        """
        # Load trained XGBoost model
        model_path = self.model_dir / "gwo_xgboost_model.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
            except Exception as exc:
                # macOS often needs libomp for XGBoost. The dashboard can still
                # replay saved held-out predictions when the native library is absent.
                self.load_error = str(exc)
                self.model = None
        
        # Load scaler
        scaler_path = self.model_dir / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Load selected feature names
        features_path = self.model_dir / "gwo_selected_features.json"
        if features_path.exists():
            with open(features_path, 'r') as f:
                self.selected_features = json.load(f)
        
        # Load training statistics
        stats_path = self.model_dir / "training_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.training_stats = json.load(f)
    
    def predict(self, features: np.ndarray, 
                feature_names: list = None) -> Dict:
        """
        Predict ASD/Control with full output.
        
        Parameters
        ----------
        features : np.ndarray
            Feature vector (79 features or 13 GWO-selected)
        feature_names : list, optional
            Feature names corresponding to input
            
        Returns
        -------
        dict : Prediction results including probabilities
        """
        # Select GWO features if full 79-feature input
        if len(features) == 79 and feature_names is not None:
            selected_idx = [feature_names.index(f) 
                          for f in self.selected_features 
                          if f in feature_names]
            features_sel = features[selected_idx]
        elif len(features) == 13:
            features_sel = features
        else:
            raise ValueError(f"Expected 79 or 13 features, got {len(features)}")
        
        # Scale features
        features_scaled = self.scaler.transform(
            features_sel.reshape(1, -1)
        )
        
        if self.model is None:
            raise RuntimeError(
                "XGBoost model is unavailable. "
                f"Original load error: {self.load_error or 'unknown error'}"
            )

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Confidence
        confidence = max(probabilities)
        
        # Z-score comparison against training set
        z_scores = self._compute_z_scores(features_sel)
        
        return {
            'prediction': int(prediction),
            'label': 'ASD' if prediction == 1 else 'Control',
            'probability_asd': float(probabilities[1]),
            'probability_control': float(probabilities[0]),
            'confidence': float(confidence),
            'features_scaled': features_scaled.flatten(),
            'features_raw': features_sel,
            'z_scores': z_scores,
            'low_confidence': confidence < 0.65
        }
    
    def _compute_z_scores(self, features: np.ndarray) -> Dict:
        """
        Compare selected feature values against the training distribution.

        These z-scores are used only for interpretation/display, not for the
        actual model prediction.
        """
        if self.training_stats is None:
            return {}
        
        z_scores = {}
        for i, feat_name in enumerate(self.selected_features):
            if feat_name in self.training_stats:
                mean = self.training_stats[feat_name]['mean']
                std = self.training_stats[feat_name]['std']
                if std > 0:
                    z_scores[feat_name] = (features[i] - mean) / std
                else:
                    z_scores[feat_name] = 0.0
        
        return z_scores
    
    def get_model_info(self) -> Dict:
        """
        Return static model metadata and performance metrics for the dashboard.

        This keeps display labels and summary numbers separate from inference
        logic so the UI can show model context without touching artifacts.
        """
        return {
            'model_type': 'XGBoost',
            'optimization': 'Grey Wolf Optimizer (GWO)',
            'n_features_total': 79,
            'n_features_selected': 13,
            'feature_reduction': '83.5%',
            'selected_features': self.selected_features,
            'cv_accuracy': 81.94,
            'test_accuracy': 75.00,
            'cv_auc': 0.8275,
            'test_auc': 0.8611,
            'cv_sensitivity': 91.0,
            'cv_specificity': 73.0,
        }
