"""
Evaluation script for Activity 18.2 — Local Ollama (Mistral).
Measures classification accuracy and latency (average, min, max) across 10 benchmark complaints.
"""

import sys
import csv
import time
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Force UTF-8 stdout for clean console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

ALLOWED_CATEGORIES = ["billing", "loan", "fraud", "app_issue"]

def normalize_category(raw_output: str) -> str:
    cleaned = raw_output.strip().lower()
    if cleaned in ALLOWED_CATEGORIES:
        return cleaned
    for cat in ALLOWED_CATEGORIES:
        if cat in cleaned:
            return cat
    return "billing"

def run_evaluation():
    csv_path = Path(__file__).parent / "test_cases.csv"
    if not csv_path.exists():
        print(f"ERROR: Test file not found at {csv_path}")
        sys.exit(1)

    print("Initializing ChatOllama(model='mistral', temperature=0.3)...")
    llm = ChatOllama(
        model="mistral",
        temperature=0.3,
        base_url="http://localhost:11434"
    )

    classification_prompt = ChatPromptTemplate.from_template(
        """Classify this customer message into exactly ONE of these four categories:
billing
loan
fraud
app_issue

Few-shot examples:

BILLING:
Complaint: "I was charged an extra fee on my EMI." -> billing
Complaint: "My account was debited twice." -> billing
Complaint: "I was charged a late payment fee." -> billing

LOAN:
Complaint: "How can I apply for a personal loan?" -> loan
Complaint: "What is the interest rate for a loan?" -> loan
Complaint: "My loan application is still pending." -> loan
Complaint: "Can I increase my sanctioned loan amount?" -> loan

FRAUD:
Complaint: "I don't recognize this transaction." -> fraud
Complaint: "Someone used my debit card without permission." -> fraud
Complaint: "I received an OTP for a purchase I did not make." -> fraud
Complaint: "Someone changed my net banking password." -> fraud

APP_ISSUE:
Complaint: "The mobile app crashes." -> app_issue
Complaint: "The app is stuck on the loading screen." -> app_issue
Complaint: "Biometric login is not working." -> app_issue
Complaint: "I cannot download my statement from the app." -> app_issue

Important classification rules:
- Loan application, loan eligibility, loan interest rate, loan status, foreclosure, or sanctioned amount -> loan.
- Unauthorized transaction, stolen credentials, suspicious activity, unknown third-party access, OTP for an unrecognized purchase -> fraud.
- Technical problems with the mobile/web application -> app_issue.
- Charges, fees, deductions, duplicate debits, billing amounts -> billing.
- Context & Follow-ups: If the message provides an account number, customer ID, or detail following up on an ongoing inquiry, retain the relevant category of the ongoing inquiry (e.g., app_issue, billing, or loan). Merely providing an account number, card number, or verification detail is NEVER fraud unless unauthorized activity, theft, or suspicious access is explicitly reported.

Return ONLY:
billing
loan
fraud
app_issue

Do not explain the classification.

Previous Context (if any): {context}
Customer Message: {text}
Category:"""
    )

    chain = classification_prompt | llm | StrOutputParser()

    total = 0
    correct = 0
    incorrect = 0
    errors = 0
    latencies = []
    results = []

    print("Running classification evaluation on Local Ollama (mistral)...")
    print("-" * 75)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            expected = row["category"].strip().lower()
            complaint = row["complaint"].strip()

            start_time = time.perf_counter()
            try:
                raw_prediction = chain.invoke({
                    "context": "None (Benchmark test case)",
                    "text": complaint
                })
                elapsed = time.perf_counter() - start_time
                latencies.append(elapsed)

                predicted = normalize_category(raw_prediction)
                is_correct = (predicted == expected)

                if is_correct:
                    correct += 1
                    status = "[PASS]"
                else:
                    incorrect += 1
                    status = "[FAIL]"

                print(
                    f"[{total:2d}] {status} Expected: {expected:<10} | Predicted: {predicted:<10} | "
                    f"Latency: {elapsed:5.2f}s | Complaint: {complaint[:30]}...",
                    flush=True
                )
                results.append({
                    "id": total,
                    "expected": expected,
                    "predicted": predicted,
                    "status": "PASS" if is_correct else "FAIL",
                    "latency": elapsed,
                    "complaint": complaint
                })

            except Exception as e:
                elapsed = time.perf_counter() - start_time
                errors += 1
                err_str = str(e).replace("\n", " ")[:40]
                print(f"[{total:2d}] [ERROR] Expected: {expected:<10} | Error: {err_str}... | Latency: {elapsed:5.2f}s", flush=True)
                results.append({
                    "id": total,
                    "expected": expected,
                    "predicted": "ERROR",
                    "status": "ERROR",
                    "latency": elapsed,
                    "complaint": complaint
                })

    avg_latency = (sum(latencies) / len(latencies)) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0
    acc = (correct / (correct + incorrect)) * 100.0 if (correct + incorrect) > 0 else 0.0

    print("=" * 40)
    print("Ollama Mistral Evaluation")
    print("=" * 40)
    print(f"Model: mistral")
    print(f"Provider: Ollama Local")
    print(f"Total test cases: {total}")
    print()
    print(f"Correct classifications: {correct}")
    print(f"Incorrect classifications: {incorrect}")
    print(f"Accuracy: {acc:.2f}%")
    print()
    print(f"Average latency: {avg_latency:.2f} seconds")
    print(f"Minimum latency: {min_latency:.2f} seconds")
    print(f"Maximum latency: {max_latency:.2f} seconds")
    print("=" * 40)

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "errors": errors,
        "accuracy": acc,
        "avg_latency": avg_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "results": results
    }

if __name__ == "__main__":
    run_evaluation()
