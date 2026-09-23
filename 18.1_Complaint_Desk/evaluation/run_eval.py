import os
import csv
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA

ALLOWED_CATEGORIES = ["billing", "loan", "fraud", "app_issue"]

def get_api_key():
    # 1. Check existing environment variable
    if os.environ.get("NVIDIA_API_KEY"):
        return os.environ.get("NVIDIA_API_KEY")

    # 2. Check .env file in project root
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_file)
            if os.environ.get("NVIDIA_API_KEY"):
                return os.environ.get("NVIDIA_API_KEY")
        except ImportError:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("NVIDIA_API_KEY="):
                        val = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if val:
                            return val

    # 3. Check .streamlit/secrets.toml
    secrets_file = project_root / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        try:
            import tomllib
            with open(secrets_file, "rb") as f:
                secrets = tomllib.load(f)
                if "NVIDIA_API_KEY" in secrets and secrets["NVIDIA_API_KEY"]:
                    return secrets["NVIDIA_API_KEY"]
        except Exception:
            pass

    return None

def normalize_category(raw_output: str) -> str:
    cleaned = raw_output.strip().lower()
    if cleaned in ALLOWED_CATEGORIES:
        return cleaned
    for cat in ALLOWED_CATEGORIES:
        if cat in cleaned:
            return cat
    return "billing"

def run_evaluation():
    api_key = get_api_key()
    if not api_key:
        print("ERROR: NVIDIA_API_KEY could not be found.")
        print("Please ensure your key is set in .env, .streamlit/secrets.toml, or the environment.")
        print("Example: set NVIDIA_API_KEY=your_key (Windows) or export NVIDIA_API_KEY='your_key' (Linux/Mac)")
        sys.exit(1)

    csv_path = Path(__file__).parent / "test_cases.csv"
    if not csv_path.exists():
        print(f"ERROR: Test file not found at {csv_path}")
        sys.exit(1)

    raw_model = os.getenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct").strip()
    model_name = raw_model

    llm = ChatNVIDIA(
        model=model_name,
        temperature=0.3,
        nvidia_api_key=api_key,
        timeout=30
    )

    classification_prompt = ChatPromptTemplate.from_template(
        """Classify this customer complaint into exactly ONE of these four categories:
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

Complaint: {text}
Category:"""
    )

    chain = classification_prompt | llm | StrOutputParser()

    total = 0
    correct = 0
    incorrect = 0
    errors = 0
    results = []

    print(f"Running classification evaluation on NVIDIA Hosted NIM ({model_name})...")
    print("-" * 75)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            cid = row["id"]
            complaint = row["complaint"]
            expected = row["expected_category"].strip().lower()

            try:
                raw_pred = chain.invoke({"text": complaint})
                predicted = normalize_category(raw_pred)
                is_correct = (predicted == expected)
                if is_correct:
                    correct += 1
                    status = "PASS"
                else:
                    incorrect += 1
                    status = "FAIL"
                print(f"[{cid:>2}] [{status}] Expected: {expected:<10} | Predicted: {predicted:<10} | Complaint: {complaint[:35]}...", flush=True)
            except Exception as e:
                errors += 1
                predicted = f"ERROR: {e}"
                status = "ERROR"
                print(f"[{cid:>2}] [{status}] Expected: {expected:<10} | API Error: {str(e)[:40]}... | Complaint: {complaint[:35]}...", flush=True)

            results.append({
                "id": cid,
                "complaint": complaint,
                "expected": expected,
                "predicted": predicted,
                "status": status
            })

    # Accuracy computed strictly among successfully evaluated cases (excluding API provider errors)
    evaluated_count = correct + incorrect
    accuracy = (correct / evaluated_count) * 100 if evaluated_count > 0 else 0.0

    print("-" * 75)
    print(f"Evaluation Summary:")
    print(f"  Model:                                  {model_name}")
    print(f"  Total test cases:                       {total}")
    print(f"  Successfully evaluated:                 {evaluated_count}")
    print(f"  Correct classifications:                {correct}")
    print(f"  Incorrect classifications:              {incorrect}")
    print(f"  API/Provider errors:                    {errors}")
    if evaluated_count > 0:
        print(f"  Accuracy (among evaluated cases):       {accuracy:.2f}%")
    else:
        print(f"  Accuracy:                               N/A (All cases encountered API errors)")
    print("-" * 75)

if __name__ == "__main__":
    run_evaluation()
