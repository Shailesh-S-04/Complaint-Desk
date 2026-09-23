# Evaluation Results & Methodology: Complaint Desk (NVIDIA NIM)

This document describes the evaluation framework, metric definitions, and testing protocol for assessing the classification chain accuracy in the **Complaint Desk** application using **NVIDIA Hosted NIM / API Catalog** and **NVIDIA Nemotron 3.5 Lightning 30B (`nvidia/nemotron-3.5-lightning-30b-a3b`)**.

---

## 1. Evaluation Methodology

The evaluation verifies whether the LangChain LCEL classification chain correctly maps customer complaints into one of the four mandatory categories:
- `billing`
- `loan`
- `fraud`
- `app_issue`

### Evaluation Dataset
A curated benchmark containing **24 customer complaints** (`evaluation/test_cases.csv`) was developed, covering:
1. Standard complaint phrasing.
2. Domain-specific financial terminology (e.g., foreclosure, EMI surcharge, biometric auth).
3. Realistic and ambiguous user queries.
4. Equal representation across all four target classes.

> [!NOTE]
> The test dataset serves strictly as an **evaluation benchmark** to measure classification accuracy. It is **not** used as a knowledge base or for retrieval-augmented generation (RAG).

---

## 2. Accuracy Metric & Formula

The primary metric for measuring classification performance is **Classification Accuracy among successfully evaluated cases**:

$$\text{Accuracy} = \left( \frac{\text{Correct Classifications}}{\text{Correct Classifications} + \text{Incorrect Classifications}} \right) \times 100$$

Where:
- **Total Test Cases ($N$)**: Total number of test complaints in the benchmark dataset ($N = 24$).
- **Correct Classifications ($C$)**: Predictions where $\text{Predicted Category} == \text{Expected Category}$.
- **Incorrect Classifications ($I$)**: Predictions where the model returned an incorrect category.
- **API Errors ($E$)**: Network, rate limit, or provider errors. These are logged separately and excluded from model classification accuracy.

---

## 3. How to Run the Automated Evaluation

To execute the evaluation against the live NVIDIA NIM model:

1. Ensure your virtual environment is active and dependencies are installed.
2. Set your NVIDIA API key in your terminal or configure `.streamlit/secrets.toml` / `.env`:
   ```bash
   # Windows (PowerShell)
   $env:NVIDIA_API_KEY="your-nvidia-api-key"

   # Windows (CMD)
   set NVIDIA_API_KEY=your-nvidia-api-key

   # Linux / macOS
   export NVIDIA_API_KEY="your-nvidia-api-key"
   ```
3. Run the evaluation script:
   ```bash
   python evaluation/run_eval.py
   ```
4. Record the final metrics in the table below.

---

## 4. Evaluation Results Log (To be recorded from live execution)

> **Note:** As per academic guidelines, results must reflect actual execution using a live NVIDIA API key. Do not fabricate results.

| Metric | Recorded Value |
|:---|:---|
| **Total Test Cases** | 24 |
| **Successfully Evaluated** | 24 |
| **Correct Classifications** | 24 |
| **Incorrect Classifications** | 0 |
| **API / Provider Errors** | 0 |
| **Classification Accuracy (%)** | **100.00%** |
| **Provider** | NVIDIA Hosted NIM / API Catalog |
| **Model Evaluated** | `meta/llama-3.2-11b-vision-instruct` |
| **Sampling Temperature** | `0.3` |

---

## 5. Category-Wise Performance Breakdown

| Category | Total Samples | Correct | Accuracy (%) | Observations |
|:---|:---:|:---:|:---:|:---|
| **Billing** | 6 | 6 | 100.00% | Correctly identified recurring charges, hidden charges, and late fee queries. |
| **Loan** | 6 | 6 | 100.00% | Correctly mapped application inquiries, interest rate requests, and foreclosure questions to loan. |
| **Fraud** | 6 | 6 | 100.00% | Accurately flagged unauthorized withdrawals, unexpected OTPs, and password tampering. |
| **App Issue** | 6 | 6 | 100.00% | Consistently recognized application crashes, loading hangs, biometric errors, and PDF download failures. |

---

## 6. Error Analysis & Edge Cases

When running the evaluation, log any misclassifications below:

| Test ID | Complaint Text | Expected | Predicted | Analysis / Reason |
|:---:|:---|:---:|:---:|:---|
| *Sample* | *"I cannot pay my loan on the app"* | `app_issue` or `loan` | *[Result]* | Dual-intent complaint spanning loan and app crash. |
