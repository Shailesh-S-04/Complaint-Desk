"""
Module 8 Hands-On Activity 18.2: Complaint Desk (Ollama Swap)
Local LLM version using ChatOllama with the Mistral model.

Pipeline Architecture:
    Customer Complaint
           ↓
    Chain 1: Classification (billing / loan / fraud / app_issue)
           ↓
    Chain 2: Polite Acknowledgement Generation (~60 words, signed XYZ Finance Support)
           ↓
    Streamlit Chat UI + Session State
"""

import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Complaint Desk — Local Ollama (Mistral)",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Taxonomy & Badges
# -----------------------------------------------------------------------------
ALLOWED_CATEGORIES = ["billing", "loan", "fraud", "app_issue"]

CATEGORY_BADGES = {
    "billing": "💳 Billing Discrepancy",
    "loan": "💰 Loan Inquiry",
    "fraud": "🚨 Fraud Alert",
    "app_issue": "📱 App Issue",
}

# -----------------------------------------------------------------------------
# Initialize Local LLM via Ollama
# -----------------------------------------------------------------------------
# Activity 18.2 specifies ChatOllama with mistral at temperature 0.3
OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.3,
    base_url=OLLAMA_BASE_URL
)

# -----------------------------------------------------------------------------
# Chain 1: Complaint Classification Chain (LCEL)
# -----------------------------------------------------------------------------
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

classification_chain = (
    classification_prompt
    | llm
    | StrOutputParser()
)

def normalize_category(raw_output: str) -> str:
    """Normalize and validate the model classification output against allowed categories."""
    cleaned = raw_output.strip().lower()
    
    # Direct exact match
    if cleaned in ALLOWED_CATEGORIES:
        return cleaned
    
    # Substring search if model returned extra words or punctuation
    for cat in ALLOWED_CATEGORIES:
        if cat in cleaned:
            return cat
            
    # Safe fallback if ambiguous or unrecognized
    return "billing"

# -----------------------------------------------------------------------------
# Chain 2: Category-Appropriate Reply Chain (LCEL)
# -----------------------------------------------------------------------------
reply_prompt = ChatPromptTemplate.from_template(
    """You are a customer-support assistant for XYZ Finance.

Write a polite acknowledgement for the customer's message.

Category:
{cat}

Previous Conversation Context (if any):
{context}

Customer's Latest Message:
{text}

Requirements:
- Approximately 60 words.
- Be professional and empathetic.
- Address the category appropriately and maintain continuity if this is a follow-up message.
- If the customer provided account or verification details, acknowledge receipt safely without asking for passwords, PINs, or OTPs.
- Do not ask the customer to post sensitive banking passwords, OTPs, or confidential credentials in chat; inform them that support will verify their account through secure official channels.
- Do not invent company policies, fees, timelines, guarantees, refunds, investigations, or regulatory facts.
- Do not claim an action has already been completed.
- Do not invent customer/account information.
- Sign as 'XYZ Finance Support'."""
)

reply_chain = (
    reply_prompt
    | llm
    | StrOutputParser()
)

# -----------------------------------------------------------------------------
# Session State Initialization (Conversation Persistence)
# -----------------------------------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("ℹ️ Application Information")
    st.markdown("---")
    
    st.markdown("### 🏛️ Architecture (Activity 18.2)")
    st.write("**Two-Chain LangChain LCEL Pipeline**")
    st.code("Chain 1: Complaint → Category\nChain 2: Category + Text → Reply", language="text")

    st.markdown("### 🏷️ Allowed Categories")
    for cat, label in CATEGORY_BADGES.items():
        st.markdown(f"- **`{cat}`** ({label})")

    st.markdown("### ⚙️ Model Configuration")
    st.write("**Provider:** `Ollama (Local Inference)`")
    st.write(f"**Model:** `{OLLAMA_MODEL}`")
    st.write("**Temperature:** `0.3`")
    st.write(f"**Endpoint:** `{OLLAMA_BASE_URL}`")
    st.caption("Zero external API calls. 100% on-device inference for maximum data privacy.")

    st.markdown("---")
    st.markdown("### 💾 Conversation State")
    st.write(f"Total interactions in session: **{len(st.session_state.conversation)}**")
    st.caption("Persistence is maintained in Streamlit session memory.")

    if st.button("🗑️ Clear Conversation History", use_container_width=True):
        st.session_state.conversation = []
        st.rerun()

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
st.title("Complaint Desk — Local Ollama (Mistral)")
st.markdown("Activity 18.2: Local, privacy-first complaint classification and response generation powered by Ollama and Mistral.")
st.markdown("---")

# Common Pre-asked Questions / FAQs
FAQ_QUESTIONS = [
    {
        "label": "💳 Extra fee on EMI",
        "text": "I was charged an extra fee on my EMI without prior notice."
    },
    {
        "label": "💰 Loan status pending",
        "text": "My personal loan application is still pending for over two weeks."
    },
    {
        "label": "🛡️ Unrecognized transaction",
        "text": "I don't recognize this transaction on my debit card statement."
    },
    {
        "label": "📱 App crashes on launch",
        "text": "The mobile app crashes immediately on the loading screen."
    },
]

# Initial Bot Welcome Message
with st.chat_message("assistant"):
    st.markdown(
        "👋 **Hello! I am your AI Support Assistant for XYZ Finance (Local Ollama Mistral Edition).**\n\n"
        "I am here to assist you with your complaints and inquiries. Whether you are experiencing "
        "billing discrepancies, loan queries, suspected fraud, or technical issues with our mobile app, "
        "all interactions are processed completely locally on-device.\n\n"
        "You can type your complaint below or click any of the common inquiries below to get started immediately:"
    )

# Quick-select FAQ buttons
st.markdown("##### 💡 Frequently Asked Questions / Common Inquiries:")
faq_cols = st.columns(len(FAQ_QUESTIONS))
selected_faq = None
for idx, faq in enumerate(FAQ_QUESTIONS):
    with faq_cols[idx]:
        if st.button(faq["label"], key=f"faq_btn_{idx}", use_container_width=True):
            selected_faq = faq["text"]

# Render conversation history from session state
for item in st.session_state.conversation:
    with st.chat_message("user"):
        st.write(item["complaint"])

    with st.chat_message("assistant"):
        cat = item["category"]
        badge_label = CATEGORY_BADGES.get(cat, f"Category: {cat}")
        st.markdown(f"**Category:** `{cat.upper()}` — *{badge_label}*")
        st.write(item["response"])

# -----------------------------------------------------------------------------
# Chat Input & Processing
# -----------------------------------------------------------------------------
complaint_input = st.chat_input("Paste or type a customer complaint...")

# Trigger processing from either text input or quick FAQ button
active_complaint = selected_faq if selected_faq else complaint_input

if active_complaint:
    clean_complaint = active_complaint.strip()
    
    if not clean_complaint:
        st.warning("Please enter a non-empty customer complaint.")
    else:
        # 1. Display User Message Immediately
        with st.chat_message("user"):
            st.write(clean_complaint)

        # 2. Process Complaint via Dual Chains
        with st.chat_message("assistant"):
            with st.spinner("Classifying complaint and generating response via Local Ollama (Mistral)..."):
                try:
                    # Construct prior context if this is a follow-up interaction
                    if st.session_state.conversation:
                        last_turn = st.session_state.conversation[-1]
                        context_str = f"Prior inquiry: '{last_turn['complaint']}' (Category: {last_turn['category']})"
                    else:
                        context_str = "None (Initial inquiry)"

                    # Chain 1: Classify Complaint / Follow-up with Context
                    raw_category = classification_chain.invoke({
                        "context": context_str,
                        "text": clean_complaint
                    })
                    predicted_category = normalize_category(raw_category)

                    # Chain 2: Generate Category-Appropriate Reply with Context
                    reply_text = reply_chain.invoke({
                        "context": context_str,
                        "text": clean_complaint,
                        "cat": predicted_category
                    })

                    # Display Assistant Response
                    badge_label = CATEGORY_BADGES.get(predicted_category, f"Category: {predicted_category}")
                    st.markdown(f"**Category:** `{predicted_category.upper()}` — *{badge_label}*")
                    st.write(reply_text)

                    # 3. Store in Session State for Persistence
                    st.session_state.conversation.append({
                        "complaint": clean_complaint,
                        "category": predicted_category,
                        "response": reply_text
                    })

                    if selected_faq:
                        st.rerun()

                except Exception as e:
                    err_msg = str(e)
                    if "connection refused" in err_msg.lower() or "connect" in err_msg.lower():
                        st.error(
                            "❌ **Ollama Connection Error**\n\n"
                            "Could not connect to the local Ollama server at `http://localhost:11434`.\n\n"
                            "**Troubleshooting:**\n"
                            "1. Verify Ollama is running: `ollama serve`\n"
                            "2. Verify the Mistral model is downloaded: `ollama pull mistral`\n"
                            "3. Test endpoint in browser: http://localhost:11434"
                        )
                    else:
                        st.error(f"An error occurred while communicating with Ollama: {err_msg}")
