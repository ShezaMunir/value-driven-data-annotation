"""
pilot_study_v2.py — HatEval Positionality Pilot Study (Version B: Annotation → Elicitation)
Researcher: Sheza Munir

COUNTERBALANCE VERSION:
  This version runs annotation BEFORE elicitation to enable comparison of annotation quality
  between the two orderings. ~7 of 15 participants should receive this version.
  Comparing label consistency and rationale depth between Version A and Version B is a
  methodological contribution for ACL: it tests whether reflective priming (elicitation first)
  improves annotation quality or introduces perspective bias.

  Predicted result (based on Smythe 2008 + Rocchio 2022):
    Version A (elicitation-first) → richer rationale, more nuanced labels
    Version B (annotation-first) → more "cold" labels, shorter rationale, but potentially
    less primed by the narrative framing of the vignette

  This design directly addresses a gap in the positionality-aware NLP annotation literature:
  no prior work has experimentally compared annotation quality across elicitation-order conditions.

See pilot_study.py for full theoretical grounding and shared components.
"""

import streamlit as st
import json
import os
import re
import requests
from datetime import datetime

# ─── Shared configuration (identical to Version A) ───────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "INSERT_HF_TOKEN_HERE")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
STORAGE_DIR = "pilot_data_v2"
TURNS_BEFORE_READY = 4

HATEVAL_DATAPOINTS = [
    {"id": 1,  "text": "These immigrants are destroying our culture and taking what's ours.", "hs": 1, "ag": 1, "tr": "G", "score": 0.81},
    {"id": 2,  "text": "Why do people feel the need to bring their customs here instead of adapting?", "hs": 0, "ag": 0, "tr": "G", "score": 0.74},
    {"id": 3,  "text": "Women who dress like that are just asking for attention.", "hs": 1, "ag": 0, "tr": "I", "score": 0.79},
    {"id": 4,  "text": "I'm so tired of foreigners complaining about the country that gave them shelter.", "hs": 1, "ag": 1, "tr": "G", "score": 0.76},
    {"id": 5,  "text": "She should stick to cooking, not politics.", "hs": 1, "ag": 0, "tr": "I", "score": 0.72},
    {"id": 6,  "text": "Maybe if they learned the language they'd feel less excluded.", "hs": 0, "ag": 0, "tr": "G", "score": 0.69},
    {"id": 7,  "text": "No one forced them to come here. They can leave if they don't like our values.", "hs": 1, "ag": 1, "tr": "G", "score": 0.83},
    {"id": 8,  "text": "She's pretty articulate for someone from that background.", "hs": 1, "ag": 0, "tr": "I", "score": 0.77},
    {"id": 9,  "text": "Immigration isn't the problem, it's the lack of integration policy.", "hs": 0, "ag": 0, "tr": "G", "score": 0.65},
    {"id": 10, "text": "Why are women always so emotional in debates? It's hard to take seriously.", "hs": 1, "ag": 0, "tr": "G", "score": 0.80},
    {"id": 11, "text": "Not all immigrants are criminals but the statistics don't lie.", "hs": 1, "ag": 1, "tr": "G", "score": 0.85},
    {"id": 12, "text": "Diversity is fine but not when it changes the fundamental character of a place.", "hs": 0, "ag": 0, "tr": "G", "score": 0.71},
    {"id": 13, "text": "She got the job because of quotas, not merit. Classic.", "hs": 1, "ag": 1, "tr": "I", "score": 0.78},
    {"id": 14, "text": "Why do immigrants always form their own communities instead of mixing?", "hs": 0, "ag": 0, "tr": "G", "score": 0.67},
    {"id": 15, "text": "These women in politics are ruining everything with their identity agenda.", "hs": 1, "ag": 1, "tr": "G", "score": 0.82},
]

SCENARIOS = [
    {
        "id": "A", "theme": "Immigration & Belonging", "target_group": "immigrants",
        "hateval_ids": [1, 2, 4, 6, 7],
        "vignette": (
            "A city councillor proposes removing multilingual signage from public transit, "
            "arguing it will 'encourage newcomers to integrate faster.' A coalition of "
            "immigrant-led community organizations responds that the signs are a lifeline for "
            "elderly and newly arrived residents navigating the city. The debate spills onto "
            "social media, where reactions range from solidarity to hostility. "
            "You are about to read some of those online comments."
        ),
    },
    {
        "id": "B", "theme": "Gender & Public Voice", "target_group": "women",
        "hateval_ids": [3, 5, 8, 10, 13],
        "vignette": (
            "A woman running for city council gives a speech on housing affordability. "
            "A clip goes viral — not for the policy content, but because commenters focus "
            "on her appearance, her tone, and whether she 'belongs' in politics. Her "
            "supporters push back; others double down. The comment section becomes a "
            "case study in how gender shapes who gets to be heard. "
            "You are about to read some of those online comments."
        ),
    },
    {
        "id": "C", "theme": "Identity, Integration & Othering", "target_group": "immigrants and women",
        "hateval_ids": [9, 11, 12, 14, 15],
        "vignette": (
            "A think-piece about 'parallel societies' and cultural integration goes viral. "
            "It argues that certain immigrant communities 'refuse to integrate' and that "
            "activist women in those communities are 'importing foreign values.' The article "
            "draws fierce responses — some calling it a necessary conversation, others calling "
            "it coded hate speech. "
            "You are about to read some of those online comments."
        ),
    },
]


def get_participant_dir(name): 
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip().lower())
    path = os.path.join(STORAGE_DIR, safe)
    os.makedirs(path, exist_ok=True)
    return path


def load_participant(name):
    d = get_participant_dir(name)
    fp = os.path.join(d, "session.json")
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    existing = [x for x in os.listdir(STORAGE_DIR) if os.path.isdir(os.path.join(STORAGE_DIR, x))]
    scenario_idx = (len(existing) - 1) % len(SCENARIOS)
    return {
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        "scenario_id": SCENARIOS[scenario_idx]["id"],
        "workflow_stage": "disclosure",
        "disclosure": {},
        "annotations": [],     # collected FIRST in Version B
        "elicitation": [],
        "micronarrative": "",
        "version": "B",        # annotation-first
    }


def save_participant(data):
    d = get_participant_dir(data["name"])
    fp = os.path.join(d, "session.json")
    with open(fp, "w") as f:
        json.dump(data, f, indent=2)


def get_scenario(sid): return next(s for s in SCENARIOS if s["id"] == sid)
def get_datapoints(sid):
    s = get_scenario(sid)
    ids = set(s["hateval_ids"])
    return [d for d in HATEVAL_DATAPOINTS if d["id"] in ids]


def call_mistral(system_prompt, messages, max_tokens=300):
    if HF_TOKEN == "INSERT_HF_TOKEN_HERE":
        return "[LLM unavailable — set HF_TOKEN env variable]"
    formatted = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    for m in messages:
        if m["role"] == "user":
            formatted += f"{m['content']} [/INST] "
        else:
            formatted += f"{m['content']} [INST] "
    formatted = formatted.rstrip(" [INST] ")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": formatted, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.7, "return_full_text": False}}
    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
        r.raise_for_status()
        result = r.json()
        return result[0].get("generated_text", "").strip() if isinstance(result, list) else str(result)
    except Exception as e:
        return f"[Model error: {e}]"


def elicitation_system_prompt(scenario, turn_idx, prev_answer=""):
    base = (
        f"You are a research interviewer helping elicit a person's lived-experience perspective "
        f"on online hate speech, in the context of: '{scenario['theme']}'. "
        f"They have ALREADY annotated 5 related tweets, so they have some context. "
        f"Ask ONE focused follow-up question (2–3 sentences max). "
        f"Anchor your question to one of these lived-experience axes: "
        f"personal identity, fairness, belonging, or relationships. "
        f"Do NOT ask multiple questions. Do NOT summarize what they said. "
        f"If they've had 4+ substantive turns, end with exactly: READY_TO_BUILD"
    )
    if turn_idx >= 3 and prev_answer:
        base += (
            f"\n\nThey previously said: \"{prev_answer[:200]}\". "
            f"Reference this and ask what made it personally significant, not just what happened."
        )
    return base


def synthesis_system_prompt():
    return (
        "Write a single 4–5 sentence first-person micronarrative from these fragments. "
        "Friendly tone, preserve the participant's own words and framing. "
        "Do NOT add interpretations they didn't express. No academic language. "
        "Output ONLY the narrative."
    )


def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,500;1,300&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #F5F3EF; color: #1C1C1C; }
    h1, h2, h3 { font-family: 'Fraunces', serif; font-weight: 300; }
    .stApp { background-color: #F5F3EF; }
    .scenario-card { background:#EDE9E0; border-left:3px solid #5B7FA6; padding:1.2rem 1.5rem; border-radius:4px; margin-bottom:1.5rem; font-size:0.95rem; line-height:1.7; }
    .tweet-card { background:#FFFFFF; border:1px solid #DDD8CE; border-radius:6px; padding:1rem 1.2rem; margin-bottom:1rem; font-size:1rem; line-height:1.6; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
    .progress-bar-bg { background:#E0DCD3; border-radius:20px; height:6px; margin:0.8rem 0 1.5rem 0; }
    .progress-bar-fill { background:#5B7FA6; height:6px; border-radius:20px; }
    .stage-label { font-size:0.75rem; color:#888; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:0.3rem; }
    .version-badge { background:#5B7FA6; color:white; font-size:0.7rem; padding:2px 8px; border-radius:10px; display:inline-block; margin-bottom:0.5rem; }
    </style>
    """, unsafe_allow_html=True)


def progress_bar(step, total=5):
    pct = int((step / total) * 100)
    st.markdown(f"<div class='progress-bar-bg'><div class='progress-bar-fill' style='width:{pct}%'></div></div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Pilot Study — Version B", layout="centered")
    inject_styles()
    os.makedirs(STORAGE_DIR, exist_ok=True)

    # ── Sign-in ───────────────────────────────────────────────────────────────
    if "participant_name" not in st.session_state:
        st.markdown("<div class='stage-label'>Pilot Study · Version B</div>", unsafe_allow_html=True)
        st.title("Language, Identity & Online Harm")
        st.markdown("*A positionality-aware annotation study*")
        st.markdown("---")
        name = st.text_input("Enter your first name or pseudonym:")
        if st.button("Continue →", type="primary") and name.strip():
            st.session_state.participant_name = name.strip()
            st.session_state.data = load_participant(name.strip())
            st.rerun()
        return

    if "data" not in st.session_state:
        st.session_state.data = load_participant(st.session_state.participant_name)

    data = st.session_state.data
    scenario = get_scenario(data["scenario_id"])
    stage = data["workflow_stage"]

    with st.sidebar:
        st.markdown(f"**Participant:** {data['name']}")
        st.markdown(f"**Scenario:** {scenario['theme']}")
        st.markdown(f"<span class='version-badge'>Version B: Annotate → Reflect</span>", unsafe_allow_html=True)
        if st.button("Save & Exit"):
            save_participant(data)
            st.success("Saved.")

    # ── Stage 1: Disclosure ──────────────────────────────────────────────────
    if stage == "disclosure":
        st.markdown("<div class='stage-label'>Step 1 of 4</div>", unsafe_allow_html=True)
        st.title("A bit about you")
        progress_bar(1)
        conn = st.selectbox("How would you describe your connection to topics of immigration, belonging, or gender?",
                            ["Direct personal experience", "Caregiver or community advocate",
                             "Professional or academic background", "Interested observer — no direct connection"])
        duration = st.text_input("How long has this been part of your life or work?")
        disclosure = st.text_area("Optionally: briefly describe your connection (won't affect your participation):", height=100)
        if st.button("Begin →", type="primary"):
            data["disclosure"] = {"connection_type": conn, "duration": duration, "text": disclosure}
            data["workflow_stage"] = "annotation"
            save_participant(data)
            st.rerun()

    # ── Stage 2: Annotation (FIRST in Version B) ─────────────────────────────
    elif stage == "annotation":
        datapoints = get_datapoints(data["scenario_id"])
        idx = len(data["annotations"])

        if idx == 0:
            # Show scenario vignette before first annotation
            st.markdown("<div class='stage-label'>Step 2 of 4 — Annotations</div>", unsafe_allow_html=True)
            st.title("Some context")
            progress_bar(2)
            st.markdown(f"<div class='scenario-card'><strong>Context:</strong> {scenario['vignette']}</div>",
                        unsafe_allow_html=True)
            st.write("You'll now annotate five online comments related to this context.")
            if st.button("Start annotating →", type="primary"):
                # Add a sentinel to skip this screen on rerun
                data["annotations_intro_done"] = True
                save_participant(data)
                st.rerun()
            return

        if idx < len(datapoints):
            dp = datapoints[idx]
            st.markdown(f"<div class='stage-label'>Step 2 of 4 — Comment {idx+1} of {len(datapoints)}</div>",
                        unsafe_allow_html=True)
            st.title("Annotating online speech")
            progress_bar(idx + 1, total=len(datapoints))

            st.markdown(f"<div class='tweet-card'>💬 &nbsp; {dp['text']}</div>", unsafe_allow_html=True)

            with st.form(f"anno_{idx}"):
                label = st.radio("**How would you label this comment?**",
                                 ["Hate speech — targets a person or group harmfully",
                                  "Borderline — could be harmful depending on context",
                                  "Not hate speech — critical or blunt but not hateful"])
                target = st.radio("**Directed at a specific individual or a broader group?**",
                                  ["Specific individual", "A group of people", "Unclear / both"], horizontal=True)
                rationale = st.text_area(
                    "**Your rationale** (3–4 sentences: what in the text led to your label, and why)",
                    height=130, placeholder="E.g. The phrase '…' signals… because…")
                # Positionality salience — reframed as reflexivity, not bias admission
                # Theoretical basis: reflexivity measure from Smythe (2008) interpretive phenomenology
                st.markdown("**How relevant did your own background feel as you read this?**")
                st.caption("1 = not at all relevant · 5 = very relevant to my own experience")
                salience = st.slider("", 1, 5, 3, label_visibility="collapsed")

                if st.form_submit_button("Submit & next →", type="primary"):
                    if len(rationale.split()) < 30:
                        st.warning("Please expand your rationale — aim for 3–4 full sentences.")
                    else:
                        data["annotations"].append({
                            "tweet_id": dp["id"], "tweet_text": dp["text"],
                            "hateval_label": dp["hs"],
                            "participant_label": label, "participant_target": target,
                            "rationale": rationale, "positionality_salience": salience,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                        save_participant(data)
                        st.rerun()
        else:
            data["workflow_stage"] = "elicitation_chat"
            save_participant(data)
            st.rerun()

    # ── Stage 3: Elicitation (SECOND in Version B) ────────────────────────────
    elif stage == "elicitation_chat":
        st.markdown("<div class='stage-label'>Step 3 of 4 — Lived Experience</div>", unsafe_allow_html=True)
        st.title("Now, your perspective")
        progress_bar(3)

        st.write(
            "Now that you've annotated those comments, we'd like to understand "
            "the perspective and experiences you brought to that task."
        )
        st.markdown(f"<div class='scenario-card'><strong>Scenario:</strong> {scenario['vignette']}</div>",
                    unsafe_allow_html=True)

        for msg in data["elicitation"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if not data["elicitation"]:
            q = "Reflecting on those comments and the scenario: what stood out most to you, and was there a moment where something felt personally resonant — connected to your own identity, your sense of fairness, or a community you belong to?"
            data["elicitation"].append({"role": "assistant", "content": q})
            save_participant(data)
            st.rerun()

        user_input = st.chat_input("Your response…")
        if user_input:
            data["elicitation"].append({"role": "user", "content": user_input})
            save_participant(data)
            turn_idx = len([m for m in data["elicitation"] if m["role"] == "user"])
            sys_p = elicitation_system_prompt(scenario, turn_idx, user_input[:200])
            response = call_mistral(sys_p, data["elicitation"], max_tokens=180)
            if "READY_TO_BUILD" in response:
                response = response.replace("READY_TO_BUILD", "").strip()
                if response:
                    data["elicitation"].append({"role": "assistant", "content": response})
                data["workflow_stage"] = "synthesis"
            else:
                data["elicitation"].append({"role": "assistant", "content": response})
            save_participant(data)
            st.rerun()

    # ── Stage 4: Synthesis ───────────────────────────────────────────────────
    elif stage == "synthesis":
        st.markdown("<div class='stage-label'>Step 4 of 4 — Your Narrative</div>", unsafe_allow_html=True)
        st.title("Your story, in your words")
        progress_bar(4)

        if not data.get("micronarrative"):
            with st.spinner("Drafting your narrative…"):
                fragments = "\n".join(
                    f"USER: {m['content']}" for m in data["elicitation"] if m["role"] == "user"
                )
                draft = call_mistral(synthesis_system_prompt(),
                                     [{"role": "user", "content": fragments}], max_tokens=250)
                data["micronarrative"] = draft
                save_participant(data)

        edited = st.text_area("Your narrative (edit freely):", value=data["micronarrative"], height=200)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Regenerate"):
                data["micronarrative"] = ""
                save_participant(data)
                st.rerun()
        with col2:
            if st.button("Accept & finish →", type="primary"):
                data["micronarrative"] = edited
                data["workflow_stage"] = "complete"
                save_participant(data)
                st.rerun()

    elif stage == "complete":
        st.balloons()
        st.title("Thank you.")
        st.markdown(f"All done, **{data['name']}**. Your annotations and reflections have been saved.")
        st.caption("Study by Sheza Munir · You may close this window.")


if __name__ == "__main__":
    main()
