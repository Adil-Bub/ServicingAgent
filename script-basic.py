import os
import json
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
import langchain

# Optional: set to False if you want to hide background LangChain terminal logs
langchain.debug = True

# ==========================================
# 1. Setup & UI Header
# ==========================================
st.set_page_config(page_title="Bank AI Agent", page_icon="🏦", layout="wide")
st.title("🏦 AI Servicing Agent")
st.markdown("Enter a customer transcript below to see the agent research data, enforce policy, and take action.")

# ==========================================
# 2. Initialize In-Memory Database (Thread-Safe Fix)
# ==========================================
if "mock_db" not in st.session_state:
    st.session_state["mock_db"] = {
        "loans": {
            "1234": {"customer_id": "C-998", "balance": "$15,000", "status": "Current", "monthly_payment": "$450", "current_due_date": 15, "due_date_changes_ytd": 0},
            "5678": {"customer_id": "C-112", "balance": "$2,500", "status": "Current", "monthly_payment": "$120", "current_due_date": 5, "due_date_changes_ytd": 0}
        },
        "customers": {
            "C-998": {"name": "John Doe", "customer_since": "2015", "account_tier": "Gold", "risk_score": "Low"},
            "C-112": {"name": "Jane Smith", "customer_since": "2024", "account_tier": "Standard", "risk_score": "Medium"}
        },
        "notes": {
            "1234": "2026-03-20 [Agent 44]: Cust called to update contact info.",
            "5678": "2026-01-10 [Agent 09]: Routine check-in."
        }
    }

# CRITICAL FIX: We bind the session state dictionary to a global variable 
# in the main thread. LangChain's background threads can safely access and 
# mutate this object WITHOUT triggering Streamlit's context warnings!
ACTIVE_DB = st.session_state["mock_db"]

# ==========================================
# 3. Define Tools (Using ACTIVE_DB)
# ==========================================
@tool
def get_loan_details(loan_id: str) -> dict:
    """Fetches structured loan data, including current due dates and modification history."""
    return ACTIVE_DB["loans"].get(loan_id, {"error": "Loan ID not found."})

@tool
def get_customer_profile(customer_id: str) -> dict:
    """Fetches structured background profile data for a specific customer."""
    return ACTIVE_DB["customers"].get(customer_id, {"error": "Customer ID not found."})

@tool
def get_account_notes(loan_id: str) -> str:
    """Fetches unstructured, raw text notes left by previous human agents regarding the loan."""
    return ACTIVE_DB["notes"].get(loan_id, "No notes found on this account.")

@tool
def change_due_date(loan_id: str, new_due_date: int) -> dict:
    """Submits a request to change the monthly due date for a loan."""
    if not isinstance(new_due_date, int) or not (1 <= new_due_date <= 28):
        return {"status": "REJECTED", "reason": "Invalid date. Due date must be between the 1st and 28th."}
    
    loan = ACTIVE_DB["loans"].get(loan_id)
    
    if not loan:
        return {"status": "REJECTED", "reason": "Loan ID not found."}
    
    if loan["due_date_changes_ytd"] >= 1:
        return {"status": "REJECTED", "reason": "Maximum due date changes (1 per year) exceeded."}
        
    # Safely updating the state dictionary
    loan["current_due_date"] = new_due_date
    loan["due_date_changes_ytd"] += 1
    
    return {"status": "SUCCESS", "message": f"Database updated. Due date successfully changed to the {new_due_date}th."}

# ==========================================
# 4. Initialize Agent
# ==========================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0,
    api_key=st.secrets["GOOGLE_API_KEY"]
)

tools = [get_loan_details, get_customer_profile, get_account_notes, change_due_date]

system_instructions = """
You are a helpful AI Servicing Agent for a bank. Your job is to assist human agents by researching customer data and executing authorized account actions.

When a customer provides a loan ID, you must:
1. Fetch the loan details.
2. Use the customer_id to fetch the customer profile.
3. Fetch the unstructured account notes.

BANK POLICY FOR DUE DATE CHANGES:
- Customers can change their due date to any day between the 1st and the 28th of the month.
- Customers are only allowed ONE due date change per year (check 'due_date_changes_ytd').
- LOGIC CHECK: If the requested new due date is EXACTLY THE SAME as their 'current_due_date', do NOT execute the change tool. Instead, politely inform the customer that their due date is already set to that day.
- If the customer is eligible, requests a valid date, and the date is different from their current date, you MUST use the `change_due_date` tool to execute the change before concluding your response.

Output your final response in 3 clear sections. Use bullet points for the summary to ensure it is highly readable:

### SUMMARY
* **Customer Name:** [Name]
* **Loan Status:** [Status]
* **Current Due Date:** [Date]
* **Request:** [Brief 1-sentence summary of what they want]

### ACTION TAKEN
[Describe the system action you took, or why it was rejected/unnecessary. Include the API response status if a tool was used.]

### NEXT BEST ACTION
[Provide the exact script the human agent should read to the customer to wrap up the call.]
"""

agent_executor = create_agent(llm, tools, system_prompt=system_instructions)

# ==========================================
# 5. The Interactive UI Logic
# ==========================================
default_transcript = "Customer: Hi, I'm calling about loan number 5678. I recently got a new job and I get paid on the 25th now. My current payment is due on the 5th, which is really tough. Can we move my monthly due date to the 28th?"

mock_transcript = st.text_area("**Customer Request**", value=default_transcript)

if st.button("Process Request", type="primary"):
    
    final_clean_text = ""
    total_input_tokens = 0
    total_output_tokens = 0
    total_run_tokens = 0
    
    with st.status("Agent is thinking and querying databases...", expanded=True) as status:
        try:
            for event in agent_executor.stream({"messages": [("user", mock_transcript)]}):
                for node_name, node_data in event.items():
                    
                    if isinstance(node_data["messages"], list):
                        last_message = node_data["messages"][-1]
                    else:
                        last_message = node_data["messages"]
                    
                    # --- TRACK TOKEN USAGE ---
                    if last_message.type == "ai" and hasattr(last_message, "usage_metadata") and last_message.usage_metadata:
                        total_input_tokens += last_message.usage_metadata.get("input_tokens", 0)
                        total_output_tokens += last_message.usage_metadata.get("output_tokens", 0)
                        total_run_tokens += last_message.usage_metadata.get("total_tokens", 0)
                    
                    # --- CLEAN OUTPUT LOGIC ---
                    st.markdown(f"**👉 Step:** `{node_name.upper()}`")
                    
                    if last_message.type == "ai" and getattr(last_message, "tool_calls", []):
                        for tool_call in last_message.tool_calls:
                            st.info(f"🔧 **Action:** Calling `{tool_call['name']}`")
                            st.json(tool_call['args'])
                            
                    elif last_message.type == "tool":
                        st.success(f"✅ **Result from:** `{last_message.name}`")
                        try:
                            st.json(json.loads(last_message.content))
                        except json.JSONDecodeError:
                            st.write(last_message.content)
                            
                    elif last_message.type == "ai" and not getattr(last_message, "tool_calls", []):
                        st.write("🧠 **Finalizing Response...**")
                        if isinstance(last_message.content, list):
                            final_clean_text = last_message.content[0].get("text", "")
                        else:
                            final_clean_text = last_message.content
                    
                    st.divider()
            
            status.update(label="Analysis Complete!", state="complete", expanded=False)
            
        except Exception as e:
            st.error(f"❌ THE LOOP CRASHED: {e}")
            status.update(label="Error Occurred", state="error")

    st.success("✨ Final Recommendation Ready")
    st.markdown(final_clean_text)
    
    st.divider()
    st.caption("**AI Token Usage & Cost Analytics**")
    cols = st.columns(3)
    cols[0].metric(label="Input Tokens", value=total_input_tokens)
    cols[1].metric(label="Output Tokens", value=total_output_tokens)
    cols[2].metric(label="Total Tokens Used", value=total_run_tokens)

# ==========================================
# 6. Sidebar Logic (Rendered at the end)
# ==========================================
with st.sidebar:
    # 1. Database View comes first
    st.header("🗄️ Live Database View")
    st.caption("Watch these values update as the agent takes action!")
    st.subheader("🏦 Loan Records")
    st.json(ACTIVE_DB["loans"])
    
    st.divider()
    
    # 2. Reset Button comes below
    if st.button("🔄 Reset Database to Default", use_container_width=True):
        if "mock_db" in st.session_state:
            del st.session_state["mock_db"]
        st.rerun()
    st.caption("Click this to reset customer due dates and quotas between tests.")
