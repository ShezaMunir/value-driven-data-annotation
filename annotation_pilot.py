import streamlit as st
import google.generativeai as genai
import json
import os
import uuid
from datetime import datetime

# 1. API Configuration
# Directly integrated Gemini 2.5 Pro API Key
genai.configure(api_key="AIzaSyAVdxZ8CT0aY7pCihsUcyaLuuYxgSGqZkQ")

# Utilizing the Gemini 2.5 Pro model for advanced reasoning in elicitation
# Model ID: gemini-2.5-pro (GA as of June 17, 2025)
model = genai.GenerativeModel('gemini-2.5-pro')

STORAGE_DIR = "pilot_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# Initialize Session State for Participant Tracking
if "prolific_id" not in st.session_state:
    # Captures Prolific ID from URL or generates a unique session ID
    st.session_state.prolific_id = st.query_params.get("prolific_id", str(uuid.uuid4()))
if "stage" not in st.session_state:
    st.session_state.stage = "elicitation_pass_1"
if "responses" not in st.session_state:
    st.session_state.responses = {"elicitation": {}, "annotations": []}

def save_to_json():
    """Securely exports participant data to the research storage directory."""
    file_path = os.path.join(STORAGE_DIR, f"resp_{st.session_state.prolific_id}.json")
    with open(file_path, "w") as f:
        json.dump(st.session_state.responses, f, indent=4)

st.title("Value-Driven Annotation Pilot")
st.caption(f"Project: The Consensus Trap | Participant: {st.session_state.prolific_id}")

# --- Phase 3: AI-Guided Lived Experience Elicitation ---

if st.session_state.stage == "elicitation_pass_1":
    st.header("Phase 3, Pass 1: Scenario & Values")
    st.markdown("""
    *Vignette: A platform is updating its community guidelines regarding 'immigration status' in profile bios. 
    Some argue visibility promotes belonging; others fear it invites systemic bias.*
    """)
    
    with st.form("pass_1"):
        q1 = st.text_area("What aspects of this situation stand out most to you, and why?")
        q2 = st.text_area("What values or principles guide how you evaluate harm or support here?")
        if st.form_submit_button("Next"):
            st.session_state.responses["elicitation"]["pass_1"] = {"aspects": q1, "values": q2}
            st.session_state.stage = "elicitation_pass_2"
            st.rerun()

elif st.session_state.stage == "elicitation_pass_2":
    st.header("Phase 3, Pass 2: Authority & Standing")
    with st.form("pass_2"):
        st.write("Reflecting on the previous scenario:")
        q3 = st.text_area("What gives you standing to interpret content like this? What makes your perspective specific?")
        
        if st.form_submit_button("Generate Micronarrative"):
            # Leveraging Gemini 2.5 Pro's reasoning to synthesize a research-ready micronarrative
            prompt = f"""
            System: You are a research assistant for a project on data annotation positionality.
            Task: Synthesize the following participant responses into a 200-300 word micronarrative profile.
            Focus: Extract their interpretive framework and relationship to the topic (Immigration/Belonging).
            
            Participant Inputs: {st.session_state.responses['elicitation']['pass_1']}
            Authority Claim: {q3}
            """
            response = model.generate_content(prompt)
            st.session_state.responses["elicitation"]["micronarrative"] = response.text
            st.session_state.stage = "annotation"
            st.rerun()

# --- Phase 5: Annotation + Rationale Collection ---

elif st.session_state.stage == "annotation":
    # Dataset 1: Immigration and Belonging
    items = [
        "Comment 1: 'The visa process is intentionally designed to be a labyrinth.'",
        "Comment 2: 'They should learn the language before coming here.'",
        "Comment 3: 'My neighbor is a first-gen immigrant and the hardest worker I know.'",
        "Comment 4: 'Is it really 'othering' to prefer cultural homogeneity?'",
        "Comment 5: 'Welcome home. We are glad you are here.'"
    ]
    
    current_idx = len(st.session_state.responses["annotations"])
    
    if current_idx < 5:
        st.subheader(f"Item {current_idx + 1} of 5")
        st.info(f"**Text:** {items[current_idx]}")
        
        with st.form(f"item_{current_idx}"):
            label = st.selectbox("Label", ["Inclusion", "Othering", "Ambiguous", "Neutral"])
            rationale = st.text_area("Rationale (Explain your decision in 2-3 sentences):")
            salience = st.select_slider("How much did your background influence this judgment? (1-5)", options=[1, 2, 3, 4, 5])
            
            if st.form_submit_button("Submit"):
                # Basic validation for rationale depth as per Phase 5 requirements
                if len(rationale.split()) < 15:
                    st.warning("Please provide a more detailed rationale for qualitative analysis.")
                else:
                    st.session_state.responses["annotations"].append({
                        "item_id": current_idx,
                        "text": items[current_idx],
                        "label": label,
                        "rationale": rationale,
                        "salience": salience,
                        "timestamp": str(datetime.now())
                    })
                    save_to_json() # Incremental save for data security
                    st.rerun()
    else:
        st.success("Annotation sequence complete. Data stored securely.")
        st.markdown("### Your Positionality Micronarrative (Generated by Gemini 2.5 Pro):")
        st.write(st.session_state.responses["elicitation"]["micronarrative"])