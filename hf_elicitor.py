import streamlit as st
import json
import os
import uuid
from datetime import datetime
from huggingface_hub import InferenceClient

# 1. Configuration & Research Context
HF_TOKEN = "INSERT_HF_TOKEN_HERE" 
client = InferenceClient("meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)

STORAGE_DIR = "pilot_data"
if not os.path.exists(STORAGE_DIR):
    os.makedirs(STORAGE_DIR)

# Phase 3, Pass 1: Scenario Injection [cite: 44]
IMMIGRANT_VIGNETTE = (
    "A local community board is debating a proposal to mandate that all signage in "
    "public parks be written in English or French only, arguing it promotes 'national "
    "unity.' Some residents feel this excludes those whose first language is neither, "
    "while others believe it is a necessary step for integration."
)

# 2. Session State Initialization (Persistent across reruns)
if "prolific_id" not in st.session_state:
    st.session_state.prolific_id = st.query_params.get("prolific_id", str(uuid.uuid4()))
if "workflow_stage" not in st.session_state:
    st.session_state.workflow_stage = "disclosure"  # Starts at Phase 2, Stage 2
if "responses" not in st.session_state:
    st.session_state.responses = {"elicitation": [], "annotations": []}

def save_data():
    """Securely saves all entries as JSON to preserve rationale provenance.""" #[cite: 1, 11]
    filepath = os.path.join(STORAGE_DIR, f"immig_{st.session_state.prolific_id}.json")
    with open(filepath, "w") as f:
        json.dump(st.session_state.responses, f, indent=4)

st.title("Immigration & Belonging: Annotation Pilot")
st.caption(f"Researcher: Sheza Munir | Participant ID: {st.session_state.prolific_id}")

# --- Phase 2, Stage 2: Self-Disclosure Survey [cite: 32-40] ---
if st.session_state.workflow_stage == "disclosure":
    st.header("Step 1: Contextual Background")
    with st.form("disclosure_form"):
        st.write("Before starting, please define your connection to this topic.") # [cite: 39]
        conn = st.selectbox("How are you connected to this topic?", 
                            ["Direct personal experience", "Caregiver or advocate", "Professional expertise", "No direct connection"])
        duration = st.text_input("How long has this topic intersected with your life?")
        disclosure = st.text_area("Briefly describe how this topic relates to your life (optional):")
        
        if st.form_submit_button("Proceed to Elicitation"):
            st.session_state.responses["disclosure"] = {"type": conn, "duration": duration, "text": disclosure}
            st.session_state.workflow_stage = "elicitation_chat" # Fixed attribution
            st.rerun()

# --- Phase 3: AI-Guided Lived Experience Elicitation [cite: 41-57] ---
elif st.session_state.workflow_stage == "elicitation_chat":
    st.header("Step 2: Lived Experience Elicitation")
    st.markdown(f"> **Scenario for Reflection:** {IMMIGRANT_VIGNETTE}")

    # Display History for the AI-guided interface [cite: 42]
    for msg in st.session_state.responses["elicitation"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Share your thoughts on the scenario..."):
        st.session_state.responses["elicitation"].append({"role": "user", "content": prompt})
        
        # Two-pass instruction: Values (Pass 1) and Standing (Pass 2) [cite: 43-53]
        instruction = (
            "You are an AI assistant helping a researcher elicit micro-narratives about immigration. "
            "Ensure you ask about: (1) what values guide their evaluation of harm/support, " # [cite: 49]
            "and (2) what gives them 'standing' to interpret this content. " # [cite: 52]
            "If they have shared enough detail for a 200-word profile, reply: 'READY_TO_BUILD'."
            # which theories are connected to this how to elicit lived experiences - get theories then get the questions 
        )
        
        # Model call via Hugging Face Inference API
        response = client.chat_completion(
            messages=[{"role": "system", "content": instruction}] + st.session_state.responses["elicitation"],
            max_tokens=250
        ).choices[0].message.content

        if "READY_TO_BUILD" in response:
            st.session_state.workflow_stage = "synthesis"
            st.rerun()
        else:
            st.session_state.responses["elicitation"].append({"role": "assistant", "content": response})
            st.rerun()

# --- Stage 3: Synthesis and Narrative Review [cite: 54, 386-392] ---
# --- Step 3: Synthesis and Narrative Review [cite: 386, 390] ---
elif st.session_state.workflow_stage == "synthesis":
    st.header("Step 3: Narrative Review")
    st.info("The AI is synthesizing your story fragments into three 'persona' voices. Please select the one that resonates most.")

    # 1. AI-only Narrative Building (Stage 2 of workflow) [cite: 386]
    if "options" not in st.session_state:
        # Construct the synthesis prompt for the 3 personas [cite: 389]
        synth_instruction = (
            "You are a research assistant. Take the following immigrant experience fragments "
            "and create 3 distinct 200-word narratives. \n\n"
            "Personas to generate:\n"
            "1. Friend: Casual, peer-like tone using colloquial language.\n" # [cite: 579]
            "2. Psychologist: Formal, empathetic, focused on emotional depth.\n" # [cite: 697]
            "3. Sibling: Casual, direct, simple prose.\n\n" # [cite: 704]
            f"User Fragments: {st.session_state.responses['elicitation']}\n\n"
            "Output EXACTLY as a JSON object with keys: 'Friend', 'Psychologist', 'Sibling'."
        )

        try:
            # Using chat_completion to avoid the Task Support error
            raw_response = client.chat_completion(
                messages=[{"role": "user", "content": synth_instruction}],
                max_tokens=1200
            ).choices[0].message.content

            # Clean and parse the JSON output
            json_str = raw_response[raw_response.find("{"):raw_response.rfind("}")+1]
            st.session_state.options = json.loads(json_str)
        except Exception as e:
            st.error(f"Synthesis error: {e}. Please ensure your Hugging Face token is active.")
            st.stop()

    # 2. Human-only Choice (Stage 3 of workflow) [cite: 390]
    choice = st.radio(
        "Which version captures your 'voice' and the objective circumstances best?",
        options=list(st.session_state.options.keys())
    )

    # 3. Human-AI Collaboration: Optional Adaptation [cite: 392]
    final_draft = st.text_area(
        "Final Edit: You may adapt the story to ensure it accurately reflects your experience.",
        value=st.session_state.options[choice],
        height=300
    )

    if st.button("Finalize Narrative & Start Annotations"):
        st.session_state.responses["final_micronarrative"] = final_draft
        st.session_state.workflow_stage = "annotation" # Move to Phase 5 [cite: 65]
        save_data() # Securely save the provenance 
        st.rerun()

# --- Phase 5: Annotation Collection [cite: 65-71, 92] ---
elif st.session_state.workflow_stage == "annotation":
    items = [
        "Text: 'We need stricter border controls to preserve our social fabric.'",
        "Text: 'Why do people always ask where I am *really* from?'",
        "Text: 'Access to healthcare shouldn't depend on a passport status.'",
        "Text: 'Is it 'othering' to prefer cultural homogeneity?'",
        "Text: 'The hyphenated identity is the most Canadian thing there is.'"
    ]
    
    idx = len(st.session_state.responses["annotations"])
    if idx < 5:
        st.subheader(f"Annotation Item {idx+1} of 5")
        st.info(items[idx])
        with st.form(f"anno_{idx}"):
            label = st.radio("Label ", ["Inclusion", "Othering", "Ambiguous"], horizontal=True) #[cite: 67]
            rationale = st.text_area("Rationale (2-3 sentences justifying your label ):") # [cite: 68]
            salience = st.slider("Positionality Salience: Did your lived experience influence this judgment? ", 1, 5, 3) #[cite: 71]
            
            if st.form_submit_button("Submit Annotation"):
                # Rationale validation [cite: 68]
                if len(rationale.split()) < 15:
                    st.warning("Please provide a more detailed rationale for our qualitative analysis.") #[cite: 73]
                else:
                    st.session_state.responses["annotations"].append({
                        "item": items[idx], "label": label, "rationale": rationale, "salience": salience
                    })
                    save_data()
                    st.rerun()
    else:
        st.success("Task Complete. Your positionality-aware annotations are saved.")
        st.balloons()