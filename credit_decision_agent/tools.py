"""
Agent tools module.
Defines the tools the LLM agent can call for credit decision making.
"""

import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json


class CreditDecisionTools:
    """Collection of tools for the credit decision agent."""

    def __init__(self, model, data_dict):
        self.model = model
        self.data_dict = data_dict
        self.explainer = None
        self._initialize_shap()

        # Policy thresholds
        self.policy = {
            "high_risk_threshold": 0.6,
            "medium_risk_threshold": 0.3,
            "max_missed_payments_for_approval": 2,
            "min_credit_utilization_concern": 0.8,
            "max_debt_ratio_for_approval": 0.5,
        }

    def _initialize_shap(self):
        """Initialize SHAP explainer with training data."""
        print("Initializing SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print("SHAP explainer ready.")

    def tool_predict_risk(self, record_scaled: pd.DataFrame) -> dict:
        """
        Tool 1: Apply XGBoost risk model to get default probability.
        Args:
            record_scaled: Single scaled record as DataFrame

        Returns:
            dict with prediction details
        """

        prob = self.model.predict_proba(record_scaled)[:, 1][0]
        prediction = int(prob >= self.policy["medium_risk_threshold"])

        # Risk category
        if prob >= self.policy["high_risk_threshold"]:
            risk_category = "HIGH"
        elif prob >= self.policy["medium_risk_threshold"]:
            risk_category = "MEDIUM"
        else:
            risk_category = "LOW"

        result = {
            "tool": "predict_risk",
            "timestamp": datetime.now().isoformat(),
            "default_probability": round(float(prob), 4),
            "risk_category": risk_category,
            "binary_prediction": prediction,
            "model_type": "XGBoost",
            "confidence": round(float(max(prob, 1 - prob)), 4)
        }
        return result


    def tool_shap_explanation(self, record_scaled: pd.DataFrame) -> dict:
        """
        Tool 2: Generate SHAP-based feature importance explanation.
        Args:
            record_scaled: Single scaled record as DataFrame
        
        Returns:

            dict with SHAP values and top contributing features
        """

        shap_values = self.explainer.shap_values(record_scaled)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For binary classification, take class 1 (default)
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]

        feature_names = self.data_dict["feature_names"]
        # Create feature importance ranking

        feature_impacts = []
        for i, (name, value) in enumerate(zip(feature_names, sv)):
            feature_impacts.append({
                "feature": name,
                "shap_value": round(float(value), 4),
                "direction": "increases_risk" if value > 0 else "decreases_risk",
                "abs_impact": round(abs(float(value)), 4)
            })

        # Sort by absolute impact
        feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)

        # Base value
        base_value = float(self.explainer.expected_value)
        if isinstance(self.explainer.expected_value, np.ndarray):
            base_value = float(self.explainer.expected_value[1]) if len(self.explainer.expected_value) > 1 else float(self.explainer.expected_value[0])

        result = {
            "tool": "shap_explanation",
            "timestamp": datetime.now().isoformat(),
            "base_value": round(base_value, 4),
            "top_risk_factors": feature_impacts[:5],
            "top_protective_factors": [f for f in feature_impacts if f["direction"] == "decreases_risk"][:3],
            "all_shap_values": {name: round(float(v), 4) for name, v in zip(feature_names, sv)},
            "feature_names": feature_names,
            "raw_shap_values": [round(float(v), 4) for v in sv]
        }
        return result


    def tool_policy_check(self, customer_profile: dict, risk_result: dict) -> dict:
        """
        Tool 3: Check against institutional policy rules.
       
        Args:
            customer_profile: Human-readable customer profile
            risk_result: Output from predict_risk tool

        Returns:
            dict with policy compliance details
        """

        violations = []
        passes = []

        # Check missed payments
        if customer_profile["missed_payments_count"] > self.policy["max_missed_payments_for_approval"]:
            violations.append({
                "rule": "Maximum missed payments exceeded",
                "threshold": self.policy["max_missed_payments_for_approval"],
                "actual": customer_profile["missed_payments_count"],
                "severity": "HIGH"
            })

        else:
            passes.append({
                "rule": "Missed payments within limit",
                "threshold": self.policy["max_missed_payments_for_approval"],
                "actual": customer_profile["missed_payments_count"]
            })

        # Check credit utilization
        if customer_profile["credit_utilization"] > self.policy["min_credit_utilization_concern"]:
            violations.append({
                "rule": "Credit utilization too high",
                "threshold": self.policy["min_credit_utilization_concern"],
                "actual": round(customer_profile["credit_utilization"], 4),
                "severity": "MEDIUM"
            })

        else:
            passes.append({
                "rule": "Credit utilization acceptable",
                "threshold": self.policy["min_credit_utilization_concern"],
                "actual": round(customer_profile["credit_utilization"], 4)
            })

        # Check risk threshold
        if risk_result["default_probability"] > self.policy["high_risk_threshold"]:
            violations.append({
                "rule": "Risk score exceeds high-risk threshold",
                "threshold": self.policy["high_risk_threshold"],
                "actual": risk_result["default_probability"],
                "severity": "HIGH"
            })

        else:
            passes.append({
                "rule": "Risk score within acceptable range",
                "threshold": self.policy["high_risk_threshold"],
                "actual": risk_result["default_probability"]
            })

        # Decision recommendation based on policy
        if any(v["severity"] == "HIGH" for v in violations):
            policy_recommendation = "REJECT"
        elif len(violations) > 0:
            policy_recommendation = "REVIEW"
        else:
            policy_recommendation = "APPROVE"

        result = {
            "tool": "policy_check",
            "timestamp": datetime.now().isoformat(),
            "policy_recommendation": policy_recommendation,
            "violations": violations,
            "passes": passes,
            "total_violations": len(violations),
            "total_passes": len(passes),
            "policy_thresholds": self.policy
        }
        return result


    def tool_fairness_check(self, record_scaled: pd.DataFrame, record_raw: pd.Series) -> dict:
        """
        Tool 4: Counterfactual fairness test.
        Changes protected attributes (SEX) and checks if decision flips.

        Args:
            record_scaled: Scaled record
            record_raw: Raw record

        Returns:
            dict with fairness diagnostic
        """

        original_prob = self.model.predict_proba(record_scaled)[:, 1][0]
        # Counterfactual: flip gender (SEX: 1->2 or 2->1)
        counterfactual_scaled = record_scaled.copy()
        original_sex = int(record_raw["SEX"])
        counterfactual_sex = 2 if original_sex == 1 else 1
        counterfactual_scaled["SEX"] = counterfactual_sex
        counterfactual_prob = self.model.predict_proba(counterfactual_scaled)[:, 1][0]

        # Check if decision flips
        threshold = self.policy["medium_risk_threshold"]
        original_decision = "REJECT" if original_prob >= threshold else "APPROVE"
        counterfactual_decision = "REJECT" if counterfactual_prob >= threshold else "APPROVE"
        decision_flipped = original_decision != counterfactual_decision
        probability_difference = abs(float(original_prob - counterfactual_prob))

        # Fairness assessment
        if decision_flipped:
            fairness_flag = "FAIL - Decision is gender-biased"
            bias_detected = True

        elif probability_difference > 0.05:
            fairness_flag = "WARNING - Notable probability difference by gender"
            bias_detected = False

        else:
            fairness_flag = "PASS - No significant gender bias detected"
            bias_detected = False

        result = {
            "tool": "fairness_check",
            "timestamp": datetime.now().isoformat(),
            "protected_attribute": "SEX",
            "original_value": "Male" if original_sex == 1 else "Female",
            "counterfactual_value": "Male" if counterfactual_sex == 1 else "Female",
            "original_probability": round(float(original_prob), 4),
            "counterfactual_probability": round(float(counterfactual_prob), 4),
            "probability_difference": round(probability_difference, 4),
            "original_decision": original_decision,
            "counterfactual_decision": counterfactual_decision,
            "decision_flipped": decision_flipped,
            "fairness_flag": fairness_flag,
            "bias_detected": bias_detected
        }

        return result

 

    def tool_historical_comparison(self, record_raw: pd.Series) -> dict:
        """
        Tool 5: Compare with similar historical cases.
        Finds similar customers in training data and reports their outcomes.

        Args:
            record_raw: Raw (unscaled) record

        Returns:
            dict with historical comparison
        """

        raw_train = self.data_dict["X_train_raw"].copy()
        raw_train["default"] = self.data_dict["y_train"]

        # Find similar customers based on key features
        age = int(record_raw["AGE"])
        limit_bal = float(record_raw["LIMIT_BAL"])
        education = int(record_raw["EDUCATION"])
        pay_0 = int(record_raw["PAY_0"])

        # Filter similar customers
        similar = raw_train[
            (raw_train["AGE"].between(age - 5, age + 5)) &
            (raw_train["LIMIT_BAL"].between(limit_bal * 0.7, limit_bal * 1.3)) &
            (raw_train["EDUCATION"] == education)
        ]

        if len(similar) < 5:
            # Relax constraints if too few matches
            similar = raw_train[
                (raw_train["AGE"].between(age - 10, age + 10)) &
                (raw_train["LIMIT_BAL"].between(limit_bal * 0.5, limit_bal * 1.5))
            ]

        # Further filter by payment behavior
        similar_with_same_pay = similar[similar["PAY_0"] == pay_0]
        if len(similar_with_same_pay) >= 5:
            comparison_group = similar_with_same_pay

        else:
            comparison_group = similar

        total_similar = len(comparison_group)
        default_count = int(comparison_group["default"].sum())
        default_rate = float(comparison_group["default"].mean()) if total_similar > 0 else 0.0

        # Summary statistics of similar customers
        avg_limit = float(comparison_group["LIMIT_BAL"].mean()) if total_similar > 0 else 0
        avg_age = float(comparison_group["AGE"].mean()) if total_similar > 0 else 0

        result = {
            "tool": "historical_comparison",
            "timestamp": datetime.now().isoformat(),
            "similar_customers_found": total_similar,
            "default_count": default_count,
            "non_default_count": total_similar - default_count,
            "historical_default_rate": round(default_rate, 4),
            "comparison_criteria": {
                "age_range": f"{age - 5} to {age + 5}",
                "credit_limit_range": f"{limit_bal * 0.7:.0f} to {limit_bal * 1.3:.0f}",
                "education": education,
                "payment_status_match": pay_0
            },

            "group_statistics": {
                "avg_credit_limit": round(avg_limit, 2),
                "avg_age": round(avg_age, 1),
            },
            "interpretation": (
                f"Among {total_similar} similar historical customers, "
                f"{default_rate:.1%} defaulted. "
                f"{'This suggests elevated risk.' if default_rate > 0.3 else 'This suggests moderate to low risk.'}"
            )
        }
        return result


    def generate_shap_plot(self, record_scaled: pd.DataFrame, save_path="shap_waterfall.html") -> str:
        """
        Generate a SHAP waterfall plot using Plotly.

        Args:
            record_scaled: Single scaled record
            save_path: Path to save the HTML plot

        Returns:
            Path to saved plot
        """

        shap_values = self.explainer.shap_values(record_scaled)

        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]

    
        feature_names = self.data_dict["feature_names"]
        # Sort by absolute value
        indices = np.argsort(np.abs(sv))[::-1][:10]  # Top 10

        top_features = [feature_names[i] for i in indices]
        top_values = [sv[i] for i in indices]
        colors = ["red" if v > 0 else "blue" for v in top_values]

        fig = go.Figure(go.Bar(
            x=top_values,
            y=top_features,
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.4f}" for v in top_values],
            textposition="outside"
        ))

        fig.update_layout(
            title="SHAP Feature Contributions to Default Risk",
            xaxis_title="SHAP Value (impact on default probability)",
            yaxis_title="Feature",
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=500,
            width=800,
            annotations=[
                dict(
                    text="Red = increases risk | Blue = decreases risk",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.15,
                    showarrow=False,
                    font=dict(size=11)
                )
            ]
        )
       
        fig.write_html(save_path)
        print(f"SHAP plot saved to: {save_path}")
        return save_path

    def generate_fairness_plot(self, fairness_result: dict, save_path="fairness_plot.html") -> str:
        """
        Generate a fairness comparison plot using Plotly.

        Args:
            fairness_result: Output from fairness_check tool
            save_path: Path to save the HTML plot

        Returns:
            Path to saved plot
        """

        categories = [
            f"Original ({fairness_result['original_value']})",
            f"Counterfactual ({fairness_result['counterfactual_value']})"
        ]

        probabilities = [
            fairness_result["original_probability"],
            fairness_result["counterfactual_probability"]
        ]

        colors = ["#2196F3", "#FF9800"]       

        fig = go.Figure(go.Bar(
            x=categories,
            y=probabilities,
            marker_color=colors,
            text=[f"{p:.4f}" for p in probabilities],
            textposition="outside"
        ))

        # Add threshold line
        fig.add_hline(
            y=self.policy["medium_risk_threshold"],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Decision Threshold ({self.policy['medium_risk_threshold']})"
        )

        fig.update_layout(
            title=f"Fairness Check: Counterfactual Analysis (Protected: {fairness_result['protected_attribute']})",
            yaxis_title="Default Probability",
            xaxis_title="Scenario",
            template="plotly_white",
            height=400,
            width=600,
            yaxis=dict(range=[0, max(probabilities) * 1.3])
        )


        fig.write_html(save_path)
        print(f"Fairness plot saved to: {save_path}")
        return save_path