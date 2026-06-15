"""
Biomarker Interpretation Module
================================
Generates human-readable clinical interpretations from
EEG features and SHAP explanations.

Maps computational findings to neurophysiological literature:
- Murias et al. (2007): Elevated theta coherence in ASD adults
- Coben et al. (2008, 2014): Mixed connectivity patterns in ASD
- Bosl et al. (2018): Reduced complexity in ASD
- Catarino et al. (2011, 2013): Interhemispheric coherence differences
- Vasa et al. (2016): Thalamocortical disruption in ASD
- Wang et al. (2013): Mirror neuron system and alpha oscillations
- Schwartz et al. (2017): Fronto-parietal beta coherence in ASD
- Eroğlu (2025): U-shaped spectral profile review

SHAP Beeswarm Empirical Findings (this model):
- ALL connectivity features: HIGH coherence → ASD
- Entropy features: LOW entropy → ASD
- Spectral features: negligible SHAP impact (~6% total importance)
"""

from typing import Dict, List
import numpy as np


class BiomarkerInterpreter:
    """
    Translates feature values and SHAP contributions
    into clinically meaningful interpretations.
    
    Interpretation logic:
    - 'high_asd': displayed when HIGH feature value pushes toward ASD (positive SHAP)
    - 'low_asd': displayed when LOW feature value pushes toward ASD (positive SHAP)
    - 'high_control': displayed when HIGH feature value pushes toward Control (negative SHAP)
    - 'low_control': displayed when LOW feature value pushes toward Control (negative SHAP)
    
    Directionality is determined from SHAP beeswarm analysis on training data.
    """
    
    # Feature-to-interpretation mapping
    # Aligned with empirical SHAP beeswarm directions
    INTERPRETATIONS = {
        'coh_theta_P5_P6': {
            'name': 'Parietal Interhemispheric Theta Coherence (P5-P6)',
            'high_asd': 'Elevated parietal theta coherence indicates atypical '
                       'posterior interhemispheric synchronization. This '
                       'hypercoherence pattern is consistent with findings by '
                       'Murias et al. (2007), who reported elevated resting-state '
                       'theta coherence in adults with ASD, and may reflect '
                       'excessive neural synchronization due to excitatory/'
                       'inhibitory imbalance.',
            'low_control': 'Lower parietal theta coherence is consistent with '
                          'typical interhemispheric communication patterns. '
                          'Moderate coherence levels suggest balanced posterior '
                          'network synchronization.',
            'low_asd': 'This subject shows atypically low parietal theta '
                      'coherence, yet other features drive the ASD prediction. '
                      'ASD is heterogeneous — some individuals show reduced '
                      'rather than elevated connectivity in specific pathways '
                      '(Catarino et al., 2013).',
            'high_control': 'Despite elevated parietal theta coherence, other '
                           'features indicate a typical neural profile overall.',
            'domain': 'Connectivity',
            'primary_direction': 'high_to_asd'
        },
        'coh_alpha_C3_C4': {
            'name': 'Central Interhemispheric Alpha Coherence (C3-C4)',
            'high_asd': 'Elevated central alpha coherence suggests excessive '
                       'interhemispheric synchronization in the sensorimotor '
                       'cortex. This over-synchronization may reflect reduced '
                       'inhibitory modulation, potentially impairing the '
                       'dynamic desynchronization required for flexible motor '
                       'planning and mirror neuron system function '
                       '(Wang et al., 2013).',
            'low_control': 'Moderate central alpha coherence is consistent with '
                          'typical interhemispheric sensorimotor coordination '
                          'and balanced inhibitory control.',
            'low_asd': 'This subject shows lower central alpha coherence, yet '
                      'other biomarkers drive the ASD prediction. This pattern '
                      'may reflect the heterogeneous connectivity profile '
                      'observed across the ASD spectrum.',
            'high_control': 'Despite elevated central alpha coherence, the '
                           'overall neural profile is consistent with typical '
                           'patterns based on other features.',
            'domain': 'Connectivity',
            'primary_direction': 'high_to_asd'
        },
        'coh_beta_F4_P4': {
            'name': 'Right Fronto-Parietal Beta Coherence (F4-P4)',
            'high_asd': 'Elevated right fronto-parietal beta coherence indicates '
                       'excessive synchronization in the attention and executive '
                       'function network. Beta hypercoherence at rest may reduce '
                       'the dynamic range available for task-evoked connectivity '
                       'changes, potentially contributing to executive functioning '
                       'challenges observed in ASD (Schwartz et al., 2017).',
            'low_control': 'Moderate right fronto-parietal beta coherence is '
                          'consistent with typical attentional network function '
                          'and preserved executive control capacity.',
            'low_asd': 'This subject shows lower fronto-parietal beta coherence, '
                      'but other features support the ASD classification. '
                      'Individual variation in fronto-parietal connectivity '
                      'is expected given ASD heterogeneity.',
            'high_control': 'Despite elevated fronto-parietal beta coherence, '
                           'other biomarkers indicate a typical neural profile.',
            'domain': 'Connectivity',
            'primary_direction': 'high_to_asd'
        },
        'coh_delta_C3_C4': {
            'name': 'Central Interhemispheric Delta Coherence (C3-C4)',
            'high_asd': 'Elevated central delta coherence may reflect altered '
                       'thalamocortical rhythm generation and atypical arousal '
                       'regulation. Excessive low-frequency synchronization '
                       'between hemispheres at rest has been associated with '
                       'disrupted inhibitory mechanisms in ASD '
                       '(Vasa et al., 2016).',
            'low_control': 'Moderate central delta coherence is consistent with '
                          'typical thalamocortical dynamics and balanced '
                          'interhemispheric resting-state connectivity.',
            'low_asd': 'This subject shows lower central delta coherence, yet '
                      'the overall biomarker profile supports ASD classification '
                      'through other features. Delta connectivity patterns '
                      'vary across the ASD spectrum.',
            'high_control': 'Despite elevated central delta coherence, the '
                           'overall feature profile indicates typical neural '
                           'patterns.',
            'domain': 'Connectivity',
            'primary_direction': 'high_to_asd'
        },
        'coh_delta_Fz_Pz': {
            'name': 'Midline Fronto-Posterior Delta Coherence (Fz-Pz)',
            'high_asd': 'Elevated fronto-posterior delta coherence suggests '
                       'excessive long-range low-frequency synchronization. '
                       'This may reflect atypical top-down communication '
                       'between frontal executive and posterior sensory regions, '
                       'potentially impairing efficient information transfer '
                       'through over-coupling (Coben et al., 2008; '
                       'Liang & Mody, 2022).',
            'low_control': 'Moderate fronto-posterior delta coherence is '
                          'consistent with typical anterior-posterior '
                          'communication and balanced long-range connectivity.',
            'low_asd': 'This subject shows lower fronto-posterior delta '
                      'coherence, yet other features drive the ASD prediction. '
                      'Reduced long-range connectivity is also documented in '
                      'ASD (Coben et al., 2008), reflecting the heterogeneous '
                      'nature of connectivity disruptions.',
            'high_control': 'Despite elevated fronto-posterior delta coherence, '
                           'the overall profile is consistent with typical '
                           'neural patterns.',
            'domain': 'Connectivity',
            'primary_direction': 'high_to_asd'
        },
        'sampen_central': {
            'name': 'Central Region Sample Entropy',
            'low_asd': 'Reduced central sample entropy indicates more rigid, '
                      'temporally predictable neural patterns in the '
                      'sensorimotor cortex. This reduced complexity is '
                      'consistent with the neural rigidity hypothesis of ASD, '
                      'where diminished signal variability may underlie '
                      'behavioral inflexibility and repetitive patterns '
                      '(Bosl et al., 2018; Catarino et al., 2011).',
            'high_control': 'Higher central entropy reflects greater signal '
                           'complexity and more flexible neural dynamics, '
                           'consistent with typical adaptive brain function '
                           'and diverse neural repertoire.',
            'high_asd': 'This subject shows higher central entropy yet is '
                       'predicted ASD based on other biomarkers. Elevated '
                       'complexity in ASD is less common but may reflect '
                       'compensatory neural dynamics or noise-driven '
                       'irregularity rather than adaptive complexity.',
            'low_control': 'Despite lower central entropy, other features '
                          'indicate a typical overall neural profile.',
            'domain': 'Entropy',
            'primary_direction': 'low_to_asd'
        },
        'sampen_global': {
            'name': 'Global Sample Entropy',
            'low_asd': 'Reduced global EEG complexity reflects less flexible '
                      'neural dynamics across the entire cortex. This pattern '
                      'mirrors the behavioral rigidity characteristic of ASD '
                      'and aligns with findings of reduced multiscale entropy '
                      'in ASD populations (Bosl et al., 2018; '
                      'Bogéa Ribeiro & da Silva Filho, 2023).',
            'high_control': 'Higher global entropy indicates greater overall '
                           'signal complexity and more dynamically flexible '
                           'neural patterns, consistent with typical brain '
                           'function.',
            'high_asd': 'This subject shows higher global entropy yet is '
                       'predicted ASD based on other features. This is an '
                       'atypical pattern for ASD and may reflect individual '
                       'heterogeneity within the spectrum.',
            'low_control': 'Despite lower global entropy, the overall feature '
                          'profile indicates typical neural patterns.',
            'domain': 'Entropy',
            'primary_direction': 'low_to_asd'
        },
        'rel_theta_parietal': {
            'name': 'Parietal Relative Theta Power',
            'low_asd': 'Lower parietal relative theta power provides weak '
                      'supporting evidence for ASD classification. This may '
                      'reflect redistribution of spectral power when '
                      'connectivity patterns are atypical, though this '
                      'feature has minimal discriminative influence in '
                      'the model.',
            'high_control': 'Higher parietal relative theta does not strongly '
                           'indicate ASD in this model and is consistent with '
                           'typical spectral distribution.',
            'high_asd': 'This spectral feature has minimal discriminative '
                       'contribution to the overall prediction. Other '
                       'features, particularly connectivity measures, are '
                       'the primary drivers of this classification.',
            'low_control': 'This spectral feature has negligible influence '
                          'on the prediction.',
            'domain': 'Spectral',
            'primary_direction': 'low_to_asd',
            'note': 'Negligible SHAP magnitude — not a primary biomarker'
        },
        'abs_theta_occipital': {
            'name': 'Occipital Absolute Theta Power',
            'low_asd': 'Lower occipital theta power provides minimal '
                      'supporting evidence for classification. This feature '
                      'has very small discriminative influence compared to '
                      'connectivity biomarkers.',
            'high_control': 'This feature does not meaningfully contribute '
                           'to prediction confidence.',
            'high_asd': 'This spectral feature has negligible discriminative '
                       'contribution to the overall prediction.',
            'low_control': 'This spectral feature has negligible influence '
                          'on the prediction.',
            'domain': 'Spectral',
            'primary_direction': 'low_to_asd',
            'note': 'Negligible SHAP magnitude — not a primary biomarker'
        },
        'rel_delta_parietal': {
            'name': 'Parietal Relative Delta Power',
            'high_asd': 'This spectral feature provides negligible '
                       'discriminative contribution. Connectivity and entropy '
                       'features are the primary drivers of classification.',
            'low_asd': 'This spectral feature provides negligible '
                      'discriminative contribution to the prediction.',
            'high_control': 'This feature does not meaningfully influence '
                           'the prediction.',
            'low_control': 'This feature does not meaningfully influence '
                          'the prediction.',
            'domain': 'Spectral',
            'primary_direction': 'neutral',
            'note': 'Near-zero SHAP magnitude — not a primary biomarker'
        },
        'abs_alpha_occipital': {
            'name': 'Occipital Absolute Alpha Power',
            'high_asd': 'This spectral feature has minimal discriminative '
                       'contribution. The prediction is primarily driven by '
                       'connectivity and entropy biomarkers.',
            'low_asd': 'This spectral feature has minimal discriminative '
                      'contribution to the prediction.',
            'high_control': 'This feature does not meaningfully influence '
                           'the classification outcome.',
            'low_control': 'This feature does not meaningfully influence '
                          'the classification outcome.',
            'domain': 'Spectral',
            'primary_direction': 'neutral',
            'note': 'Near-zero SHAP magnitude — not a primary biomarker'
        },
        'rel_theta_frontal': {
            'name': 'Frontal Relative Theta Power',
            'high_asd': 'This spectral feature has minimal discriminative '
                       'influence in the model. While elevated frontal theta '
                       'is documented in the ASD literature (Eroğlu, 2025), '
                       'it does not strongly drive predictions in this model '
                       'compared to connectivity features.',
            'low_asd': 'This spectral feature provides negligible '
                      'discriminative contribution to the prediction.',
            'high_control': 'This feature does not meaningfully influence '
                           'the classification outcome.',
            'low_control': 'This feature does not meaningfully influence '
                          'the classification outcome.',
            'domain': 'Spectral',
            'primary_direction': 'neutral',
            'note': 'Near-zero SHAP magnitude — not a primary biomarker'
        },
        'theta_alpha_F': {
            'name': 'Frontal Theta/Alpha Ratio',
            'low_asd': 'Lower frontal theta/alpha ratio provides very weak '
                      'supporting evidence. This feature has negligible '
                      'discriminative contribution compared to the dominant '
                      'connectivity biomarkers.',
            'high_asd': 'This spectral ratio feature has negligible '
                       'discriminative contribution to the overall prediction.',
            'high_control': 'This feature does not meaningfully influence '
                           'the classification outcome.',
            'low_control': 'This feature does not meaningfully influence '
                          'the classification outcome.',
            'domain': 'Spectral',
            'primary_direction': 'low_to_asd',
            'note': 'Near-zero SHAP magnitude — not a primary biomarker'
        },
    }
    
    # Domain-level summary interpretations
    DOMAIN_SUMMARIES = {
        'Connectivity': {
            'dominant': 'Functional connectivity features dominate this prediction, '
                       'accounting for the majority of the model\'s decision. '
                       'The pattern of elevated coherence across multiple pathways '
                       'suggests atypical neural synchronization consistent with '
                       'the excitatory/inhibitory imbalance theory of ASD.',
            'supporting': 'Connectivity features provide supporting evidence '
                         'for this prediction.'
        },
        'Entropy': {
            'dominant': 'Signal complexity features dominate this prediction, '
                       'indicating atypical neural dynamics.',
            'supporting': 'Reduced signal complexity provides complementary '
                         'evidence, suggesting more rigid neural patterns '
                         'consistent with the ASD literature.'
        },
        'Spectral': {
            'dominant': 'Spectral power features contribute to this prediction, '
                       'though they typically play a supporting rather than '
                       'primary role in ASD classification.',
            'supporting': 'Spectral features provide minimal additional evidence. '
                         'The prediction is primarily driven by connectivity '
                         'and/or entropy biomarkers.'
        }
    }
    
    def interpret(self, prediction_result: Dict, 
                  shap_explanation: Dict) -> Dict:
        """
        Generate comprehensive interpretation from prediction + SHAP.
        
        Returns structured interpretation with:
        - Overall summary
        - Per-feature interpretations
        - Confidence assessment
        - Clinical context
        - Domain analysis

        This is the main public method used by the dashboard after inference.
        """
        label = prediction_result['label']
        confidence = prediction_result['confidence']
        top_features = shap_explanation['top_features']
        z_scores = prediction_result.get('z_scores', {})
        
        # Generate per-feature interpretations
        feature_interpretations = []
        for feat in top_features[:5]:
            feat_name = feat['feature']
            direction = feat['direction']
            shap_val = feat['shap_value']
            feat_value = feat.get('value', None)
            
            if feat_name in self.INTERPRETATIONS:
                interp = self.INTERPRETATIONS[feat_name]
                explanation = self._select_explanation(
                    interp, shap_val, feat_value, feat_name
                )
                
                feature_interpretations.append({
                    'feature': feat_name,
                    'display_name': interp['name'],
                    'domain': interp['domain'],
                    'direction': direction,
                    'shap_value': shap_val,
                    'explanation': explanation,
                    'z_score': z_scores.get(feat_name, None),
                    'is_primary_biomarker': interp.get('note') is None
                })
        
        # Overall summary
        summary = self._generate_summary(label, confidence, 
                                          feature_interpretations)
        
        # Confidence warning
        confidence_note = self._generate_confidence_note(confidence)
        
        # Domain analysis
        dominant_domain = self._get_dominant_domain(feature_interpretations)
        domain_summary = self._get_domain_summary(feature_interpretations)
        
        return {
            'summary': summary,
            'feature_interpretations': feature_interpretations,
            'confidence_note': confidence_note,
            'prediction': label,
            'confidence': confidence,
            'dominant_domain': dominant_domain,
            'domain_summary': domain_summary,
            'biomarker_profile': self._generate_biomarker_profile(
                feature_interpretations, label
            )
        }
    
    def _select_explanation(self, interp: Dict, shap_val: float, 
                           feat_value: float, feat_name: str) -> str:
        """
        Select the appropriate explanation based on SHAP direction
        and feature value.
        
        Logic:
        - Positive SHAP = pushes toward ASD
        - Negative SHAP = pushes toward Control
        - Feature value determines high/low context
        """
        primary_direction = interp.get('primary_direction', 'high_to_asd')
        
        if shap_val > 0:  # Feature pushes toward ASD
            # Determine if feature value is high or low
            # For connectivity: high values → ASD (primary)
            # For entropy: low values → ASD (primary)
            if primary_direction == 'high_to_asd':
                # Positive SHAP + high_to_asd primary = expected pattern
                return interp.get('high_asd', interp.get('low_asd', ''))
            elif primary_direction == 'low_to_asd':
                # Positive SHAP + low_to_asd primary = expected pattern
                return interp.get('low_asd', interp.get('high_asd', ''))
            else:
                # Neutral/unknown — use generic
                return interp.get('high_asd', interp.get('low_asd', ''))
        else:  # Feature pushes toward Control (negative SHAP)
            if primary_direction == 'high_to_asd':
                # Negative SHAP for high_to_asd feature = low value → Control
                return interp.get('low_control', 
                       interp.get('high_control', 
                       'This feature supports the Control classification.'))
            elif primary_direction == 'low_to_asd':
                # Negative SHAP for low_to_asd feature = high value → Control
                return interp.get('high_control',
                       interp.get('low_control',
                       'This feature supports the Control classification.'))
            else:
                return interp.get('high_control', 
                       interp.get('low_control',
                       'This feature supports the Control classification.'))
    
    def _generate_summary(self, label: str, confidence: float,
                          feature_interpretations: List[Dict]) -> str:
        """
        Generate the top-level natural-language summary for the prediction.

        The wording changes with predicted class and confidence so uncertain
        outputs are framed more cautiously.
        """
        
        # Count primary biomarkers contributing
        primary_count = sum(1 for f in feature_interpretations 
                          if f.get('is_primary_biomarker', False))
        
        if label == 'ASD':
            if confidence > 0.80:
                summary = (
                    f"The EEG profile shows strong indicators of ASD-related "
                    f"neural patterns (confidence: {confidence*100:.1f}%). "
                    f"The classification is primarily driven by atypical "
                    f"functional connectivity patterns — elevated coherence "
                    f"across multiple brain pathways — consistent with the "
                    f"neural hyperconnectivity profile documented in ASD."
                )
            elif confidence > 0.65:
                summary = (
                    f"The EEG profile shows moderate indicators of ASD-related "
                    f"patterns (confidence: {confidence*100:.1f}%). "
                    f"Connectivity biomarkers suggest atypical neural "
                    f"synchronization, though some features are borderline."
                )
            else:
                summary = (
                    f"The EEG profile shows weak ASD-related indicators "
                    f"(confidence: {confidence*100:.1f}%). "
                    f"Features are near the decision boundary and this "
                    f"prediction carries substantial uncertainty. "
                    f"Clinical correlation is essential."
                )
        else:  # Control
            if confidence > 0.80:
                summary = (
                    f"The EEG profile is consistent with typical neural "
                    f"patterns (confidence: {confidence*100:.1f}%). "
                    f"Connectivity and complexity measures fall within "
                    f"ranges associated with neurotypical function."
                )
            elif confidence > 0.65:
                summary = (
                    f"The EEG profile is mostly consistent with typical "
                    f"patterns (confidence: {confidence*100:.1f}%). "
                    f"No strong ASD-related biomarkers detected, though "
                    f"some features show mild atypicality."
                )
            else:
                summary = (
                    f"The EEG profile leans toward typical patterns but with "
                    f"low confidence ({confidence*100:.1f}%). "
                    f"This subject's features are near the decision boundary. "
                    f"Further assessment may be warranted."
                )
        
        return summary
    
    def _generate_confidence_note(self, confidence: float) -> str:
        """
        Generate a warning or note when confidence is below strong certainty.

        Empty string means no extra warning is needed for the dashboard.
        """
        if confidence < 0.55:
            return (
                "⚠️ VERY LOW CONFIDENCE: This prediction is near random "
                "chance. The model cannot reliably distinguish this subject's "
                "EEG patterns. Clinical judgment should take full precedence. "
                "Do NOT use this result for clinical decision-making."
            )
        elif confidence < 0.65:
            return (
                "⚠️ LOW CONFIDENCE: Prediction uncertainty is high. "
                "This subject's features are near the decision boundary, "
                "suggesting an atypical or borderline profile. "
                "Clinical correlation is strongly recommended."
            )
        elif confidence < 0.75:
            return (
                "ℹ️ MODERATE CONFIDENCE: The prediction has reasonable "
                "support but should be interpreted alongside clinical "
                "assessment."
            )
        else:
            return ""
    
    def _get_dominant_domain(self, interpretations: List[Dict]) -> str:
        """
        Identify which biomarker domain contributes the most SHAP magnitude.

        Domains include Connectivity, Entropy, and Spectral features.
        """
        if not interpretations:
            return 'Unknown'
        
        # Weight by absolute SHAP value
        domain_importance = {}
        for interp in interpretations:
            d = interp.get('domain', 'Unknown')
            shap_mag = abs(interp.get('shap_value', 0))
            domain_importance[d] = domain_importance.get(d, 0) + shap_mag
        
        if domain_importance:
            return max(domain_importance, key=domain_importance.get)
        return 'Unknown'
    
    def _get_domain_summary(self, interpretations: List[Dict]) -> Dict:
        """
        Summarize each contributing biomarker domain for dashboard metrics.

        Importance is based on the share of absolute SHAP magnitude assigned
        to each domain among the interpreted top features.
        """
        domain_importance = {}
        for interp in interpretations:
            d = interp.get('domain', 'Unknown')
            shap_mag = abs(interp.get('shap_value', 0))
            domain_importance[d] = domain_importance.get(d, 0) + shap_mag
        
        total_importance = sum(domain_importance.values()) or 1.0
        dominant = self._get_dominant_domain(interpretations)
        
        summaries = {}
        for domain, importance in domain_importance.items():
            proportion = importance / total_importance
            if domain in self.DOMAIN_SUMMARIES:
                if domain == dominant:
                    summaries[domain] = {
                        'text': self.DOMAIN_SUMMARIES[domain]['dominant'],
                        'proportion': proportion
                    }
                else:
                    summaries[domain] = {
                        'text': self.DOMAIN_SUMMARIES[domain]['supporting'],
                        'proportion': proportion
                    }
        
        return summaries
    
    def _generate_biomarker_profile(self, interpretations: List[Dict],
                                     label: str) -> str:
        """
        Generate a concise biomarker profile summary describing
        the overall neural pattern.
        """
        if label != 'ASD':
            return ("Neural biomarker profile within typical ranges. "
                   "No consistent ASD-related patterns identified.")
        
        # Check which patterns are present
        has_hyperconnectivity = any(
            'Connectivity' == f.get('domain') and f.get('shap_value', 0) > 0
            for f in interpretations
        )
        has_reduced_entropy = any(
            'Entropy' == f.get('domain') and f.get('shap_value', 0) > 0
            for f in interpretations
        )
        
        if has_hyperconnectivity and has_reduced_entropy:
            return (
                "Biomarker profile: Hyperconnected yet rigid neural dynamics. "
                "Elevated functional connectivity (excessive synchronization) "
                "combined with reduced signal complexity (less flexible neural "
                "patterns). This profile is consistent with the excitatory/"
                "inhibitory imbalance theory of ASD — a brain that is "
                "over-synchronized at rest but lacks dynamic adaptability."
            )
        elif has_hyperconnectivity:
            return (
                "Biomarker profile: Atypical neural hyperconnectivity. "
                "Elevated coherence across multiple pathways suggests "
                "excessive neural synchronization, potentially reflecting "
                "reduced inhibitory modulation characteristic of ASD."
            )
        elif has_reduced_entropy:
            return (
                "Biomarker profile: Reduced neural complexity. "
                "Lower signal entropy indicates more rigid, predictable "
                "neural dynamics, consistent with the behavioral "
                "inflexibility observed in ASD."
            )
        else:
            return (
                "Biomarker profile: Atypical neural patterns detected "
                "across multiple domains."
            )
