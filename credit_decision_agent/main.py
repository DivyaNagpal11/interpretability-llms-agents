"""
Main entry point for the Credit Decision Explainability Agent.
Usage:
    python main.py                  # Process a default sample record
    python main.py --record 42      # Process test record at index 42
    python main.py --record 42 100 7  # Process multiple records
    python main.py --batch-fairness # Run batch fairness analysis
"""

import argparse
import json
import sys
import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

from data_loader import load_and_prepare_data
from model_trainer import train_model, save_model, load_model
from tools import CreditDecisionTools
from agent import CreditDecisionAgent
from fairness import batch_fairness_analysis

def setup_pipeline():
    """
    Initialize the full pipeline: data loading, model training, tools, and agent.

    Returns:
        tuple: (agent, data_dict, tools)
    """

    # Step 1: Load and prepare data
    print("=" * 70)
    print("CREDIT DECISION EXPLAINABILITY AGENT - SETUP")
    print("=" * 70)

    data_dict = load_and_prepare_data()
    # Step 2: Train or load model
    model_path = "xgboost_credit_model.pkl"
    if os.path.exists(model_path):
        print(f"\nLoading existing model from {model_path}...")
        model = load_model(model_path)

    else:
        model = train_model(
            data_dict["X_train"], data_dict["y_train"],
            data_dict["X_test"], data_dict["y_test"]
        )
        save_model(model, model_path)

    # Step 3: Initialize tools
    print("\nInitializing agent tools...")
    tools = CreditDecisionTools(model, data_dict)

    # Step 4: Initialize agent
    print("Initializing LLM agent...")
    agent = CreditDecisionAgent(tools)

    print("\nSetup complete. Ready to process applications.\n")
    return agent, data_dict, tools


def process_single_record(agent, record_index):
    """
    Process a single test record through the agent pipeline.

    Args:
        agent: CreditDecisionAgent instance
        record_index: Index into the test set

    Returns:
        Final report dict
    """

    report = agent.process_application(record_index)

    # Save audit trail
    agent.save_trajectory(record_index)

    # Save report summary
    report_path = f"decision_report_record_{record_index}.json"

    # Create a serializable summary (exclude very long fields)
    summary = {
        "record_index": report["record_index"],
        "actual_default": report["actual_default"],
        "customer_profile": report["customer_profile"],
        "risk_assessment": report["risk_assessment"],
        "shap_top_factors": report["shap_explanation"]["top_risk_factors"],
        "policy_recommendation": report["policy_check"]["policy_recommendation"],
        "policy_violations": report["policy_check"]["violations"],
        "fairness_flag": report["fairness_diagnostic"]["fairness_flag"],
        "fairness_bias_detected": report["fairness_diagnostic"]["bias_detected"],
        "historical_default_rate": report["historical_comparison"]["historical_default_rate"],
        "plots": report["plots"]
    }

    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nDecision report saved to: {report_path}")
    return report


def run_batch_fairness(data_dict, tools):
    """Run batch-level fairness analysis."""
    print("\n" + "=" * 70)
    print("BATCH FAIRNESS ANALYSIS")
    print("=" * 70)
    metrics = batch_fairness_analysis(
        model=tools.model,
        X_test_scaled=data_dict["X_test"],
        X_test_raw=data_dict["X_test_raw"],
        feature_names=data_dict["feature_names"],
        threshold=tools.policy["medium_risk_threshold"],
        save_path="batch_fairness_report.html"
    )

    print("\nBatch Fairness Metrics:")
    print(f"  Male count: {metrics['male_count']}")
    print(f"  Female count: {metrics['female_count']}")
    print(f"  Male avg default probability: {metrics['male_avg_probability']:.4f}")
    print(f"  Female avg default probability: {metrics['female_avg_probability']:.4f}")
    print(f"  Male approval rate: {metrics['male_approval_rate']:.2%}")
    print(f"  Female approval rate: {metrics['female_approval_rate']:.2%}")
    print(f"  Disparate Impact Ratio: {metrics['disparate_impact_ratio']:.4f}")
    print(f"  Statistical Parity Difference: {metrics['statistical_parity_difference']:.4f}")
    print(f"  4/5ths Rule Pass: {'PASS' if metrics['four_fifths_rule_pass'] else 'FAIL'}")

    # Save metrics
    with open("batch_fairness_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nBatch fairness metrics saved to: batch_fairness_metrics.json")
    return metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Credit Decision Explainability Agent - MVP POC"
    )
    parser.add_argument(
        "--record",
        type=int,
        nargs="+",
        default=None,
        help="Test record index(es) to process. Default: processes record 0."
    )

    parser.add_argument(
        "--batch-fairness",
        action="store_true",
        help="Run batch fairness analysis on the entire test set."
    )

    parser.add_argument(
        "--random",
        type=int,
        default=None,
        help="Process N randomly selected test records."
    )

    args = parser.parse_args()

    # Setup pipeline
    agent, data_dict, tools = setup_pipeline()
    test_size = len(data_dict["X_test"])
    print(f"Test set size: {test_size} records available (indices 0 to {test_size - 1})")

    # Determine which records to process
    records_to_process = []
    if args.random:
        import numpy as np
        np.random.seed(42)
        records_to_process = np.random.choice(test_size, size=min(args.random, test_size), replace=False).tolist()
        print(f"\nRandomly selected {len(records_to_process)} records: {records_to_process}")

    elif args.record is not None:
        records_to_process = args.record

    else:
        # Default: process record 0
        records_to_process = [0]

    # Validate indices
    valid_records = []
    for idx in records_to_process:
        if 0 <= idx < test_size:
            valid_records.append(idx)
        else:
            print(f"WARNING: Record index {idx} is out of range (0-{test_size-1}). Skipping.")

    # Process records
    all_reports = []
    for idx in valid_records:
        try:
            report = process_single_record(agent, idx)
            all_reports.append(report)
        except Exception as e:
            print(f"\nERROR processing record {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Run batch fairness if requested
    if args.batch_fairness:
        run_batch_fairness(data_dict, tools)

    # Print summary
    if all_reports:
        print("\n" + "=" * 70)
        print("PROCESSING SUMMARY")
        print("=" * 70)
        print(f"\nRecords processed: {len(all_reports)}")

        for report in all_reports:
            actual = "DEFAULT" if report["actual_default"] == 1 else "NO DEFAULT"
            decision = report["policy_check"]["policy_recommendation"]
            risk = report["risk_assessment"]["risk_category"]
            prob = report["risk_assessment"]["default_probability"]
            fair = "✓" if not report["fairness_diagnostic"]["bias_detected"] else "✗ BIAS"

            print(f"  Record {report['record_index']:>5}: "
                  f"Decision={decision:<8} | Risk={risk:<6} | "
                  f"P(default)={prob:.4f} | "
                  f"Fairness={fair} | "
                  f"Actual={actual}")

    print("\nDone. All outputs saved to current directory.")
    print("Files generated:")
    print("valid_records", valid_records)
    print("  - xgboost_credit_model.pkl (trained model)")
    for idx in valid_records:
        print(f"  - shap_plot_record_{idx}.html (SHAP visualization)")
        print(f"  - fairness_plot_record_{idx}.html (fairness visualization)")
        print(f"  - decision_report_record_{idx}.json (decision report)")
        print(f"  - audit_trail_record_{idx}.json (full audit trail)")
    if args.batch_fairness:
        print("  - batch_fairness_report.html (batch fairness plots)")
        print("  - batch_fairness_metrics.json (batch fairness metrics)")
 

if __name__ == "__main__":
    main()