# Activity 18.2 — The OpenAI ↔ Ollama Swap: Evaluation & Comparison Report

**Course Module:** Module 8: GenAI Application Engineering  
**Activity:** 18.2 Activity B — The OpenAI ↔ Ollama Swap  
**Comparison:** Activity 18.1 (Cloud-Hosted NVIDIA NIM) vs. Activity 18.2 (Local Ollama Mistral)

---

## 1. Executive Summary & Comparison Table

Both versions were evaluated against the **same ten representative finance complaints** covering all four domain categories: `billing`, `loan`, `fraud`, and `app_issue`.

| Metric | Activity 18.1: Hosted NVIDIA NIM | Activity 18.2: Local Ollama Swap |
|---|---|---|
| **Model** | `meta/llama-3.2-11b-vision-instruct` (11B) | `mistral` (7B v0.3) |
| **Provider / Host** | NVIDIA Hosted API Catalog / NIM | Local Ollama Instance (`localhost:11434`) |
| **Classification Accuracy** | **100.00%** (10 / 10 PASS) | **100.00%** (10 / 10 PASS) |
| **Average Latency (Overall)** | **7.34 seconds** | **10.42 seconds** *(includes cold-start)* |
| **Warm-State Inference Latency** | **1.11s – 1.83s** | **0.29s – 0.50s** |
| **Cold-Start / Peak Latency** | **22.66 seconds** *(network + server wake)* | **100.54 seconds** *(weights loading to RAM)* |
| **API Token Cost / 1k Requests** | ~$0.20 – $0.50 (or NVIDIA dev credits) | **$0.00** (Zero external API fees) |
| **Infrastructure Cost** | Managed serverless cloud infrastructure | Client hardware, RAM (~4.4 GB), electricity |
| **Data Privacy** | Payloads transit public Internet via TLS | **100% on-device** (Zero data egress) |
| **Authentication Dependency** | `NVIDIA_API_KEY` required | **No API keys or internet connection required** |

---

## 2. Detailed Empirical Observations

### A. Classification Accuracy
Both models achieved **100% accuracy (10/10 PASS)** on the standardized benchmark dataset:
- `billing` complaints (extra fee on EMI, duplicate debit) were mapped accurately by both models.
- `loan` inquiries (personal loan application, interest rate query, sanctioned amount increase) were properly identified.
- `fraud` alerts (unrecognized transactions, unauthorized debit card use) were separated from routine billing disputes.
- `app_issue` complaints (mobile app crashes, stuck loading screens, statement download failures) were cleanly identified.

### B. Latency Analysis
```text
Measured Latency Distribution (10 Shared Cases):
  18.1 Hosted NVIDIA:  Min: 1.11s | Max: 22.66s | Mean:  7.34s
  18.2 Local Ollama:   Min: 0.29s | Max: 100.54s| Mean: 10.42s
```
- **Cold-Start Phase**: Ollama's first call took **100.54s** as the 4.4 GB GGUF model weights were loaded from disk into memory. NVIDIA's cold-start was **22.66s**, governed by network connection establishment and hosted microservice warming.
- **Warm Inference Phase**: Once loaded into memory, **local Ollama significantly outperformed the cloud API in speed**, clocking between **0.29s and 0.50s per request**, compared to NVIDIA's **1.11s – 1.83s** (which is bounded by public network round-trip latency).

### C. Reply Quality
Both models received the identical reply prompt (~60 words, empathetic tone, no invented facts/policies, signed as *XYZ Finance Support*):
- **NVIDIA Llama 3.2 11B**: Generated articulate, empathetic, and strictly bounded acknowledgements. It closely adhered to the 60-word limit and maintained strong corporate polish.
- **Ollama Mistral 7B**: Produced concise, professional, and well-structured replies. While slightly more direct in phrasing than Llama 3.2 11B, it adhered cleanly to negative constraints (no invented fee amounts or promises).

### D. Cost per 1,000 Requests
- **Hosted Cloud API (NVIDIA NIM / OpenRouter / OpenAI)**:
  - Incur per-token pricing (typically ~$0.05 to $0.20 per 1M input tokens and ~$0.20 to $0.60 per 1M output tokens).
  - For 1,000 complaints (~300 prompt tokens + ~80 completion tokens = ~380k tokens total), estimated API cost is **~$0.10 to $0.25 per 1,000 requests**.
- **Local Ollama Inference**:
  - Direct API cost is **$0.00**.
  - However, local inference is not free in total cost of ownership (TCO): it consumes local GPU/CPU compute, system RAM (~4.5 GB), and electricity. At enterprise scale, dedicated local inference nodes require capital expenditure (servers, GPUs, cooling, maintenance).

### E. Data Privacy & Compliance
- **Hosted Cloud API**:
  - Customer complaints contain sensitive Financial Personally Identifiable Information (PII), such as transaction references, debit dispute contexts, and EMI numbers.
  - Data leaves the corporate firewall and transits the public Internet to a third-party datacenter. Requires strict Business Associate Agreements (BAA), SOC2 certification, and GDPR/CCPA data processing compliance.
- **Local Ollama**:
  - The entire complaint payload remains strictly within the local host memory.
  - **Zero network egress**: No external API calls, eliminating the risk of third-party eavesdropping or provider logging.

---

## 3. Which Would I Ship for a Bank, and Why?

> ### **Decision: I Would Ship the Local Ollama Architecture (Activity 18.2) for a Bank.**

### Justification Across the Four Rubric Dimensions:

#### 1. Data Privacy & Regulatory Compliance (Primary Driver)
Banking is one of the most strictly regulated industries in the world (governed by PCI-DSS, GDPR, RBI Master Directions, GLBA, and FINRA). Customer complaints frequently contain account dispute numbers, transaction amounts, and fraud allegations. Sending this telemetry to a public cloud API—even over TLS—introduces third-party vendor risk, data sovereignty hurdles, and audit exposure. Running **Ollama on-premise in a private banking datacenter or on air-gapped branch workstations guarantees that customer financial data never leaves the institution's security perimeter**.

#### 2. Latency & Determinism
While the local engine requires an initial cold-start weight-load, its steady-state inference latency is remarkable: **0.29s to 0.50s per request**, compared to **1.11s – 22.66s for cloud API round-trips**. For high-throughput banking call centers and customer service desks, sub-second deterministic response times without dependency on external ISP bandwidth or cloud provider outages are essential.

#### 3. Cost & Predictability at Scale
A large commercial bank processes millions of customer interactions each month. At that volume, metered cloud API charges become an open-ended operational expense (OpEx) subject to price changes or sudden rate-limit throttling (HTTP 429). With Ollama, the model runs on existing on-premise enterprise infrastructure with **zero incremental token costs**, making operational budgeting completely predictable.

#### 4. Reply Quality & Model Sufficiency
Our empirical evaluation proved that **Mistral 7B achieved 100.00% classification accuracy**, matching the larger 11B cloud model across all four financial categories. Because financial complaint desk acknowledgement is a bounded classification and acknowledgement task (rather than open-ended creative writing), Mistral 7B delivers all the required linguistic capability without the security overhead of cloud APIs.
