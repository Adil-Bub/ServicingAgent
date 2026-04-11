import os
import json
import time
import streamlit as st

from twilio.rest import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# ==========================================
# 1. Setup & UI Header
# ==========================================
st.set_page_config(page_title="Bank AI Agent", page_icon="🏦", layout="wide")
st.title("🏦 AI Servicing Agent")
st.markdown("Enter a customer transcript below or **receive a phone call** to speak to the AI.")

# ==========================================
# 2. Initialize In-Memory Database
# ==========================================
if "mock_db" not in st.session_state:
    st.session_state["mock_db"] = {
        "loans": {
            "1234": {"customer_id": "C-998", "balance": "$15,000", "status": "Current", "monthly_payment": "$450", "current_due_date": 15, "due_date_changes_ytd": 0},
            "5678": {"customer_id": "C-112", "balance": "$2,500", "status": "Current", "monthly_payment": "$120", "current_due_date": 5, "due_date_changes_ytd": 0}
        },
        "customers": {
            "C-998": {"name": "Victoria Richards", "customer_since": "2015", "account_tier": "Gold", "risk_score": "Low"},
            "C-112": {"name": "Joseph Smith", "customer_since": "2024", "account_tier": "Standard", "risk_score": "Medium"}
        }
    }

ACTIVE_DB = st.session_state["mock_db"]

# ==========================================
# 3. Define Tools 
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
def change_due_date(loan_id: str, new_due_date: int) -> dict:
    """Submits a request to change the monthly due date for a loan."""
    if not isinstance(new_due_date, int) or not (1 <= new_due_date <= 28):
        return {"status": "REJECTED", "reason": "Invalid date. Due date must be between the 1st and 28th."}
    
    loan = ACTIVE_DB["loans"].get(loan_id)
    
    if not loan:
        return {"status": "REJECTED", "reason": "Loan ID not found."}
    
    if loan["due_date_changes_ytd"] >= 1:
        return {"status": "REJECTED", "reason": "Maximum due date changes (1 per year) exceeded."}
        
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

tools = [get_loan_details, get_customer_profile, change_due_date]

system_instructions = """
You are a helpful AI Servicing Agent for a bank. Your job is to assist human agents by researching customer data and executing authorized account actions.

When a customer provides a loan ID, you must:
1. Fetch the loan details.
2. Use the customer_id to fetch the customer profile.

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
# 5. Twilio Speech-to-Text & The Interactive UI
# ==========================================

# Manage transcript state so it persists and links directly to the text area
if "transcript_text" not in st.session_state:
    st.session_state["transcript_text"] = "Customer: Hi, I'm calling about loan number 5678. I recently got a new job and I get paid on the 25th now. My current payment is due on the 5th. Can we move my monthly due date to the 28th?"

# Remember if a call has successfully finished so we can draw the ghost box
if "call_finished" not in st.session_state:
    st.session_state["call_finished"] = False

st.markdown("### 📞 Option 1: Call in your request")
user_phone = st.text_input("Enter your phone number (e.g., +919876543210):", value="+918420605026")

if st.button("Ring My Phone", type="primary"):
    if not user_phone:
        st.error("Please enter a phone number first.")
    else:
        try:
            # Reset the finished state while a new call is happening
            st.session_state["call_finished"] = False 
            
            # Initialize Twilio Client
            twilio_client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
            twilio_number = st.secrets["TWILIO_PHONE_NUMBER"]

            # TwiML instructs Twilio to speak, then record the user's response and transcribe it
            twiml_instructions = """
            <Response>
                <Say>Hello. You have reached the AI Servicing Agent. Please state your loan number and your request after the beep. Press the pound key when finished.</Say>
                <Record transcribe="true" maxLength="30" finishOnKey="#"/>
            </Response>
            """

            # THE LIVE ANIMATED BOX
            with st.status("Calling your phone...", expanded=True) as call_status:
                st.write("Initiating call...")
                call = twilio_client.calls.create(
                    twiml=twiml_instructions,
                    to=user_phone,
                    from_=twilio_number
                )
                
                st.write(f"Ringing... (Call SID: {call.sid})")
                
                # Poll until the call is finished
                while call.status not in ["completed", "failed", "busy", "no-answer", "canceled"]:
                    time.sleep(1)
                    call = twilio_client.calls(call.sid).fetch()
                
                if call.status != "completed":
                    st.error(f"Call ended with status: {call.status}")
                    st.stop()

                st.write("Call completed. Processing the recording...")
                
                # Poll to find the recording associated with this call
                recordings = []
                while not recordings:
                    time.sleep(1)
                    recordings = twilio_client.recordings.list(call_sid=call.sid)
                
                recording_sid = recordings[0].sid
                st.write("Recording processed! Transcribing speech to text...")

                # Poll to get the completed transcription by checking recent transcriptions
                final_text = ""
                while not final_text:
                    time.sleep(1)
                    # Fetch the 20 most recent transcriptions from Twilio
                    recent_transcriptions = twilio_client.transcriptions.list(limit=20)
                    
                    # Search the list for the one matching our recording SID
                    target_transcription = next(
                        (t for t in recent_transcriptions if t.recording_sid == recording_sid), None
                    )
                    
                    # Check if it exists and has finished processing
                    if target_transcription and target_transcription.status == "completed":
                        final_text = target_transcription.transcription_text
                
                # Update the session state key directly so the text box populates
                st.session_state["transcript_text"] = f"Customer: {final_text}"
                
                # Update status to complete and keep it on screen
                call_status.update(label="Transcription Complete!", state="complete", expanded=False)
                st.success("Speech successfully converted to text! You can now process the request below.")
                
                # Tell Streamlit the call is officially done!
                st.session_state["call_finished"] = True

        except Exception as e:
            st.error(f"Twilio Error: {e}")

elif st.session_state["call_finished"]:
    with st.status("Transcription Complete!", state="complete", expanded=False):
        st.write("Initiating call...")
        st.write("Ringing...")
        st.write("Call completed. Processing the recording...")
        st.write("Recording processed! Transcribing speech to text...")
    st.success("Speech successfully converted to text! You can now process the request below.")

st.markdown("### ⌨️ Option 2: Edit or Type Request")
mock_transcript = st.text_area("**Customer Request**", key="transcript_text", height=100)

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
# 6. Sidebar Logic
# ==========================================
with st.sidebar:
    st.header("🗄️ Live Database View")
    st.caption("Watch these values update as the agent takes action!")
    st.subheader("🏦 Loan Records")
    st.json(ACTIVE_DB["loans"])
    
    st.divider()
    
    if st.button("🔄 Reset Database to Default", use_container_width=True):
        if "mock_db" in st.session_state:
            del st.session_state["mock_db"]
        st.rerun()
    st.caption("Click this to reset customer due dates and quotas between tests.")