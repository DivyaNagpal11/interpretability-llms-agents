"""
LLM Agent module.
Orchestrates tool calls and generates natural language explanations using Gemini.
Logs all steps to Langfuse for audit trail.
"""


import os
import json
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from langfuse import Langfuse
# from langfuse.decorators import observe, langfuse_context
import pandas as pd

load_dotenv() 

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Configure Langfuse

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)


class CreditDecisionAgent:
    """
    Agent that orchestrates credit decision tools and generates
    natural language explanations using Gemini.
    """

    def __init__(self, tools):
        """
        Args:
            tools: CreditDecisionTools instance
        """
        self.tools = tools
        self.model = genai.GenerativeModel("gemini-2.5-pro")
        self.trajectory = []  # Audit log

    def _log_step(self, step_type: str, content: dict):
        """Log a step in the agent trajectory."""

        entry = {
            "step_type": step_type,
            "timestamp": datetime.now().isoformat(),
            "content": content
        }
        self.trajectory.append(entry)
        return entry


    def _call_gemini(self, prompt: str, trace_name: str = "gemini_call") -> str:
        """
        Call Gemini API and log to Langfuse.

        Args:
            prompt: The prompt to send
            trace_name: Name for Langfuse trace

        Returns:
            Generated text response
        """

        generation = langfuse.generation(
            name=trace_name,
            input=prompt,
            model="gemini-1.5-flash"
        )

        try:
            response = self.model.generate_content(prompt)
            output_text = response.text

            generation.end(
                output=output_text,
                metadata={"status": "success"}
            )
            return output_text

        except Exception as e:
            generation.end(
                output=str(e),
                metadata={"status": "error"}
            )
            raise e


    def process_application(self, record_index: int) -> dict:
        """
        Main agent workflow: process a credit application.

        Args:
            record_index: Index of the test record to process

        Returns:
            Complete decision report with explanation
        """
        # Reset trajectory for new application
        self.trajectory = []

        # Create Langfuse trace for the entire application processing
        trace = langfuse.trace(
            name="credit_decision_process",
            metadata={
                "record_index": record_index,
                "timestamp": datetime.now().isoformat()
            }
        )

        print(f"\n{'='*70}")
        print(f"PROCESSING CREDIT APPLICATION - Record Index: {record_index}")
        print(f"{'='*70}\n")

        # ---- Step 1: Fetch customer data ----
        print("[Step 1] Fetching customer profile...")

        record_scaled = self.tools.data_dict["X_test"].iloc[[record_index]]
        record_raw = self.tools.data_dict["X_test_raw"].iloc[record_index]
        actual_label = self.tools.data_dict["y_test"][record_index]


        from data_loader import get_customer_profile
        customer_profile = get_customer_profile(record_raw)

        self._log_step("input", {
            "record_index": record_index,
            "customer_profile": customer_profile,
            "actual_default": int(actual_label)
        })


        span_fetch = trace.span(
            name="fetch_customer_profile",
            input={"record_index": record_index},
            output=customer_profile
        )


        print(f"  Customer: Age {customer_profile['age']}, "
              f"{customer_profile['sex']}, {customer_profile['education']}, "
              f"Credit Limit: ${customer_profile['credit_limit']:,.0f}")

       
        # ---- Step 2: Risk Prediction ----
        print("\n[Step 2] Running risk model (XGBoost)...")
        risk_result = self.tools.tool_predict_risk(record_scaled)
        self._log_step("tool_call", risk_result)

        span_risk = trace.span(
            name="tool_predict_risk",
            input={"record_index": record_index},
            output=risk_result
        )

        print(f"  Default Probability: {risk_result['default_probability']:.4f}")
        print(f"  Risk Category: {risk_result['risk_category']}")
        print(f"  Model Confidence: {risk_result['confidence']:.4f}")


        # ---- Step 3: SHAP Explanation ----
        print("\n[Step 3] Generating SHAP explanation...")

        shap_result = self.tools.tool_shap_explanation(record_scaled)
        self._log_step("tool_call", shap_result)

        span_shap = trace.span(
            name="tool_shap_explanation",
            input={"record_index": record_index},
            output={k: v for k, v in shap_result.items() if k != "raw_shap_values"}
        )


        print(f"  Top Risk Factors:")
        for factor in shap_result["top_risk_factors"][:3]:
            print(f"    - {factor['feature']}: {factor['shap_value']:+.4f} ({factor['direction']})")

       
        # ---- Step 4: Policy Check ----
        print("\n[Step 4] Checking policy compliance...")

        policy_result = self.tools.tool_policy_check(customer_profile, risk_result)
        self._log_step("tool_call", policy_result)

        span_policy = trace.span(
            name="tool_policy_check",
            input={"customer_profile": customer_profile, "risk_result": risk_result},
            output=policy_result
        )

        print(f"  Policy Recommendation: {policy_result['policy_recommendation']}")
        print(f"  Violations: {policy_result['total_violations']}, Passes: {policy_result['total_passes']}")

        # ---- Step 5: Fairness Check ----
        print("\n[Step 5] Running fairness diagnostic...")
        fairness_result = self.tools.tool_fairness_check(record_scaled, record_raw)
        self._log_step("tool_call", fairness_result)


        span_fairness = trace.span(
            name="tool_fairness_check",
            input={"record_index": record_index},
            output=fairness_result
        )

        print(f"  Fairness Flag: {fairness_result['fairness_flag']}")
        print(f"  Probability Difference: {fairness_result['probability_difference']:.4f}")

        # ---- Step 6: Historical Comparison ----
        print("\n[Step 6] Comparing with historical cases...")

        historical_result = self.tools.tool_historical_comparison(record_raw)
        self._log_step("tool_call", historical_result)

        span_historical = trace.span(
            name="tool_historical_comparison",
            input={"record_index": record_index},
            output=historical_result
        )

        print(f"  Similar Customers Found: {historical_result['similar_customers_found']}")
        print(f"  Historical Default Rate: {historical_result['historical_default_rate']:.2%}")

        # ---- Step 7: Generate Plots ----
        print("\n[Step 7] Generating visualizations...")
        shap_plot_path = self.tools.generate_shap_plot(
            record_scaled,
            save_path=f"shap_plot_record_{record_index}.html"
        )

        fairness_plot_path = self.tools.generate_fairness_plot(
            fairness_result,
            save_path=f"fairness_plot_record_{record_index}.html"
        )

        # ---- Step 8: LLM Reasoning - Generate Natural Language Explanation ----
        print("\n[Step 8] Generating natural language explanation via Gemini...")

        explanation_prompt = self._build_explanation_prompt(
            customer_profile, risk_result, shap_result,
            policy_result, fairness_result, historical_result
        )


        self._log_step("llm_prompt", {"prompt": explanation_prompt})
        explanation = self._call_gemini(explanation_prompt, trace_name="generate_explanation")
        self._log_step("llm_response", {"explanation": explanation})


        # ---- Step 9: Generate Audit Summary ----
        print("\n[Step 9] Generating audit summary...")

        audit_prompt = self._build_audit_prompt(
            customer_profile, risk_result, shap_result,
            policy_result, fairness_result, historical_result
        )

    
        audit_summary = self._call_gemini(audit_prompt, trace_name="generate_audit_summary")   
        self._log_step("llm_response", {"audit_summary": audit_summary})

        # ---- Compile Final Report ----
        final_report = {
            "record_index": record_index,
            "actual_default": int(actual_label),
            "customer_profile": customer_profile,
            "risk_assessment": risk_result,
            "shap_explanation": {k: v for k, v in shap_result.items()
                                if k not in ["raw_shap_values", "all_shap_values", "feature_names"]},
            "policy_check": policy_result,
            "fairness_diagnostic": fairness_result,
            "historical_comparison": historical_result,
            "natural_language_explanation": explanation,
            "audit_summary": audit_summary,
            "plots": {
                "shap_plot": shap_plot_path,
                "fairness_plot": fairness_plot_path
            },
            "agent_trajectory": self.trajectory
        }

        # Log final report to Langfuse
        trace.span(
            name="final_report",
            input={"record_index": record_index},
            output={
                "decision": policy_result["policy_recommendation"],
                "risk_category": risk_result["risk_category"],
                "fairness_flag": fairness_result["fairness_flag"],
                "actual_default": int(actual_label)
            }
        )


        # Flush Langfuse
        langfuse.flush()

        # Print final explanation
        print(f"\n{'='*70}")
        print("DECISION EXPLANATION")
        print(f"{'='*70}")
        print(explanation)
        print(f"\n{'='*70}")
        print("AUDIT SUMMARY")
        print(f"{'='*70}")
        print(audit_summary)
        print(f"\n{'='*70}")
        print(f"GROUND TRUTH: {'DEFAULT' if actual_label == 1 else 'NO DEFAULT'}")
        print(f"{'='*70}\n")

        return final_report
   

    def _build_explanation_prompt(self, customer_profile, risk_result, shap_result,
                                   policy_result, fairness_result, historical_result):
        """
        Build the prompt for Gemini to generate a natural language explanation.
        """
        # Format top SHAP factors nicely
        top_factors_text = ""
        for i, factor in enumerate(shap_result["top_risk_factors"][:5], 1):
            direction = "↑ increases risk" if factor["direction"] == "increases_risk" else "↓ decreases risk"
            top_factors_text += f"  {i}. {factor['feature']}: SHAP value = {factor['shap_value']:+.4f} ({direction})\n"

        protective_text = ""
        for i, factor in enumerate(shap_result.get("top_protective_factors", [])[:3], 1):
            protective_text += f"  {i}. {factor['feature']}: SHAP value = {factor['shap_value']:+.4f} (decreases risk)\n"

        # Format policy violations
        violations_text = ""
        if policy_result["violations"]:
            for v in policy_result["violations"]:
                violations_text += f"  - {v['rule']} (Severity: {v['severity']}, Threshold: {v['threshold']}, Actual: {v['actual']})\n"
        else:
            violations_text = "  None\n"
       
        passes_text = ""
        if policy_result["passes"]:
            for p in policy_result["passes"]:
                passes_text += f"  - {p['rule']} (Threshold: {p['threshold']}, Actual: {p['actual']})\n"
        else:
            passes_text = "  None\n"

        prompt = f"""You are a credit risk analyst AI assistant. Your job is to explain a credit decision

in clear, professional natural language that would satisfy both the customer and a regulatory auditor.

 

Based on the following tool outputs, generate a comprehensive decision explanation.

 

=== CUSTOMER PROFILE ===

- Age: {customer_profile['age']}

- Sex: {customer_profile['sex']}

- Education: {customer_profile['education']}

- Marital Status: {customer_profile['marriage']}

- Credit Limit: ${customer_profile['credit_limit']:,.0f}

- Credit Utilization: {customer_profile['credit_utilization']:.2%}

- Recent Payment Status: {customer_profile['recent_payment_status']}

- Missed Payments (last 6 months): {customer_profile['missed_payments_count']}

- Latest Bill Amount: ${customer_profile['latest_bill_amount']:,.0f}

- Latest Payment Amount: ${customer_profile['latest_payment_amount']:,.0f}

- Total Bills (6 months): ${customer_profile['total_bill_amounts']:,.0f}

- Total Payments (6 months): ${customer_profile['total_payment_amounts']:,.0f}

 

=== RISK MODEL OUTPUT ===

- Model: {risk_result['model_type']}

- Default Probability: {risk_result['default_probability']:.4f} ({risk_result['default_probability']:.2%})

- Risk Category: {risk_result['risk_category']}

- Model Confidence: {risk_result['confidence']:.4f}

 

=== SHAP INTERPRETABILITY (Feature Contributions) ===

Base value (average prediction): {shap_result['base_value']:.4f}

 

Top contributing features to the prediction:

{top_factors_text}

 

Top protective features (reducing risk):

{protective_text if protective_text.strip() else "  None identified"}

 

=== POLICY COMPLIANCE CHECK ===

- Policy Recommendation: {policy_result['policy_recommendation']}

- Total Violations: {policy_result['total_violations']}

- Total Passes: {policy_result['total_passes']}

 

Policy Violations:

{violations_text}

 

Policy Passes:

{passes_text}

 

=== FAIRNESS DIAGNOSTIC ===

- Protected Attribute Tested: {fairness_result['protected_attribute']}

- Original ({fairness_result['original_value']}): Probability = {fairness_result['original_probability']:.4f}

- Counterfactual ({fairness_result['counterfactual_value']}): Probability = {fairness_result['counterfactual_probability']:.4f}

- Probability Difference: {fairness_result['probability_difference']:.4f}

- Decision Flipped: {fairness_result['decision_flipped']}

- Fairness Assessment: {fairness_result['fairness_flag']}

 

=== HISTORICAL COMPARISON ===

- Similar Customers Found: {historical_result['similar_customers_found']}

- Historical Default Rate: {historical_result['historical_default_rate']:.2%}

- Interpretation: {historical_result['interpretation']}

 

=== INSTRUCTIONS ===

Please generate a structured explanation with the following sections:

 

1. **Decision Summary**: State the recommendation (APPROVE/REVIEW/REJECT) and the key reason in 1-2 sentences.

 

2. **Risk Assessment Explanation**: Explain the default probability and risk category in plain language.

   What does this mean for the customer?

 

3. **Key Factors Driving the Decision**: Using the SHAP values, explain in natural language which

   specific customer attributes most influenced the prediction and why. Do NOT just list feature names -

   translate them into meaningful explanations (e.g., "PAY_0" should be explained as "most recent payment status").

 

4. **Policy Compliance**: Summarize which policy rules were met and which were violated.

   Explain the implications.

 

5. **Fairness Assessment**: Explain the counterfactual fairness test results. Was the decision

   fair with respect to gender? Provide reassurance or flag concerns.

 

6. **Historical Context**: How does this customer compare to similar historical applicants?

 

7. **Confidence & Caveats**: Note the model's confidence level and any caveats or limitations.

 

Write in a professional but accessible tone. This explanation should be suitable for:

- A customer asking "why was my application rejected/approved?"

- A regulatory auditor checking for compliance and fairness

- An internal reviewer validating the decision process

"""

        return prompt

   

    def _build_audit_prompt(self, customer_profile, risk_result, shap_result,
                             policy_result, fairness_result, historical_result):
        """
        Build the prompt for Gemini to generate a concise audit trail summary.
        """
        # Summarize trajectory steps

        steps_summary = ""
        for i, step in enumerate(self.trajectory, 1):
            step_type = step["step_type"]
            timestamp = step["timestamp"]
            if step_type == "tool_call":
                tool_name = step["content"].get("tool", "unknown")
                steps_summary += f"  Step {i} [{timestamp}]: Tool Call - {tool_name}\n"
            elif step_type == "input":
                steps_summary += f"  Step {i} [{timestamp}]: Input - Customer record loaded\n"
            elif step_type == "llm_prompt":
                steps_summary += f"  Step {i} [{timestamp}]: LLM Prompt - Explanation prompt constructed\n"
            elif step_type == "llm_response":
                steps_summary += f"  Step {i} [{timestamp}]: LLM Response - Generated\n"

        prompt = f"""You are a regulatory compliance AI assistant. Generate a concise audit trail summary

for a credit decision. This summary will be stored for regulatory review.


=== DECISION DETAILS ===

- Risk Score: {risk_result['default_probability']:.4f}

- Risk Category: {risk_result['risk_category']}

- Policy Recommendation: {policy_result['policy_recommendation']}

- Policy Violations: {policy_result['total_violations']}

- Fairness Flag: {fairness_result['fairness_flag']}

- Bias Detected: {fairness_result['bias_detected']}

- Historical Default Rate (similar customers): {historical_result['historical_default_rate']:.2%}

- Model Used: {risk_result['model_type']}

- Model Confidence: {risk_result['confidence']:.4f}

 

=== AGENT TRAJECTORY ===

{steps_summary}

 

=== INSTRUCTIONS ===

Generate a concise audit summary with:

 

1. **Decision Record**: One-line summary of the decision and key metrics.

2. **Process Integrity**: Confirm all required steps were executed (risk model, interpretability, policy check, fairness check, historical comparison).

3. **Bias Review**: Summarize the fairness test outcome.

4. **Policy Adherence**: Were all policies followed? Any exceptions?

5. **Audit Trail Completeness**: Confirm all steps are logged with timestamps.

6. **Risk Flags**: Any concerns that require human review?

 

Keep it concise and structured. Use bullet points. This is for regulatory filing.

"""

        return prompt


    def save_trajectory(self, record_index: int, output_dir: str = "."):
        """
        Save the full agent trajectory to a JSON file for audit purposes.

        Args:
            record_index: The record index processed
            output_dir: Directory to save the file
        """

        filepath = os.path.join(output_dir, f"audit_trail_record_{record_index}.json")

        # Make trajectory JSON-serializable
        serializable_trajectory = []
        for step in self.trajectory:
            serializable_step = {
                "step_type": step["step_type"],
                "timestamp": step["timestamp"],
                "content": self._make_serializable(step["content"])
            }
            serializable_trajectory.append(serializable_step)

        with open(filepath, "w", indent=2) as f:
            json.dump(serializable_trajectory, f, indent=2, default=str)

        print(f"Audit trail saved to: {filepath}")
        return filepath

    def _make_serializable(self, obj):
        """Recursively convert numpy types and other non-serializable types."""
        import numpy as np

        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        else:
            return obj