"""
Extended fairness analysis module.
Provides batch-level fairness diagnostics beyond individual counterfactual checks.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def batch_fairness_analysis(model, X_test_scaled, X_test_raw, feature_names,
                             threshold=0.3, save_path="batch_fairness_report.html"):
    """
    Run fairness analysis across the entire test set.
    Compares default probability distributions by gender.

    Args:
        model: Trained model
        X_test_scaled: Scaled test features
        X_test_raw: Raw test features
        feature_names: List of feature names
        threshold: Decision threshold
        save_path: Path to save the report

    Returns:
        dict with batch fairness metrics
    """

    probs = model.predict_proba(X_test_scaled)[:, 1]
    decisions = (probs >= threshold).astype(int)

    # Gender groups
    sex_raw = X_test_raw["SEX"].values
    male_mask = sex_raw == 1
    female_mask = sex_raw == 2

    male_probs = probs[male_mask]
    female_probs = probs[female_mask]

    male_approval_rate = 1 - decisions[male_mask].mean()
    female_approval_rate = 1 - decisions[female_mask].mean()


    # Disparate impact ratio (4/5ths rule)
    if max(male_approval_rate, female_approval_rate) > 0:
        disparate_impact = min(male_approval_rate, female_approval_rate) / max(male_approval_rate, female_approval_rate)
    else:
        disparate_impact = 1.0


    # Statistical parity difference
    stat_parity_diff = abs(male_approval_rate - female_approval_rate)

    metrics = {
        "male_count": int(male_mask.sum()),
        "female_count": int(female_mask.sum()),
        "male_avg_probability": round(float(male_probs.mean()), 4),
        "female_avg_probability": round(float(female_probs.mean()), 4),
        "male_approval_rate": round(float(male_approval_rate), 4),
        "female_approval_rate": round(float(female_approval_rate), 4),
        "disparate_impact_ratio": round(float(disparate_impact), 4),
        "statistical_parity_difference": round(float(stat_parity_diff), 4),
        "four_fifths_rule_pass": disparate_impact >= 0.8,
        "threshold_used": threshold
    }

    # Generate plots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Default Probability Distribution by Gender",
                        "Approval Rates by Gender")
    )


    # Distribution plot
    fig.add_trace(
        go.Histogram(x=male_probs, name="Male", opacity=0.7,
                     marker_color="#2196F3", nbinsx=30),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=female_probs, name="Female", opacity=0.7,
                     marker_color="#FF9800", nbinsx=30),
        row=1, col=1
    )


    # Approval rate bar chart
    fig.add_trace(
        go.Bar(
            x=["Male", "Female"],
            y=[male_approval_rate, female_approval_rate],
            marker_color=["#2196F3", "#FF9800"],
            text=[f"{male_approval_rate:.2%}", f"{female_approval_rate:.2%}"],
            textposition="outside",
            showlegend=False
        ),
        row=1, col=2
    )

    fig.add_hline(
        y=0.8 * max(male_approval_rate, female_approval_rate),
        line_dash="dash", line_color="red",
        annotation_text="4/5ths Rule Threshold",
        row=1, col=2
    )

    fig.update_layout(
        title_text="Batch Fairness Analysis Report",
        template="plotly_white",
        height=450,
        width=1000,
        barmode="overlay"
    )


    fig.update_xaxes(title_text="Default Probability", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Gender", row=1, col=2)
    fig.update_yaxes(title_text="Approval Rate", row=1, col=2)
    fig.write_html(save_path)
    print(f"Batch fairness report saved to: {save_path}")

    return metrics