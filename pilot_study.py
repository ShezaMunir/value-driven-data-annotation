"""
pilot_study.py — HatEval Positionality Pilot Study (Version A: Elicitation → Annotation)
Researcher: Sheza Munir

═══════════════════════════════════════════════════════════════════════════════
THEORETICAL GROUNDING (not shown to participants)
═══════════════════════════════════════════════════════════════════════════════

ORDERING — Elicitation FIRST, annotation SECOND.
  Smythe et al. (2008) "Methodological Considerations for Interpretive Phenomenological
  Inquiry" — narrative/reflective mindset before evaluation tasks enriches rationale depth.
  Annotation-first activates cold "judge mode" that suppresses lived-experience disclosure.
  Version B (pilot_study_v2.py) counterbalances for ~7 of 15 participants.

ELICITATION DESIGN
  • Lived-experience axes: Rocchio et al. (2022) "Eliciting Values of Patients with MCC"
    Domain-driven prompts surface values that free-response alone misses. Each turn is
    anchored to one of: personal identity, fairness, belonging, relationships.
  • Vignette construction: Barter & Renold (1999) "Using Vignettes in Educational Research"
    80–150 words, grounded in HatEval thematic clusters, max 3 vignettes to avoid fatigue.
    Vignettes elicit lived experience ONLY — no tweets shown during elicitation.
  • Iterative probing: Willig (2013) "Elicitation Interview Technique"
    initial reaction → harm/target cue → lived experience axis → revisit & deepen.
  • Hermeneutic circle: Smythe (2008) after Heidegger — later prompts reference earlier
    answers to preserve part-whole coherence across turns.

ANNOTATION DESIGN
  • 15 tweets from HatEval English training split (Basile et al. 2019, SemEval Task 5).
    Sampled via multi-classifier disagreement + entropy scoring:
      cardiffnlp/twitter-roberta-base-hate, unitary/toxic-bert,
      cardiffnlp/twitter-roberta-base-offensive
    combined_score = 0.5 × norm(variance) + 0.5 × norm(mean binary entropy)
  • 5 tweets per scenario, thematically grouped:
      Scenario A — Border security & national identity rhetoric
      Scenario B — Refugees, humanitarian crisis & family separation
      Scenario C — Cultural/religious belonging & counter-speech
  • Positionality salience reframed as reflexivity (Smythe 2008):
    "How relevant did your own background feel?" — not a bias admission.
  • Rationale minimum: ~35 words / 3–4 sentences.

MICRONARRATIVE: Boyle & Butcher (2024) "Drawn from Life" — fidelity to participant's
  own words, no reinterpretation, first person, friendly tone.

MODEL: mistralai/Mistral-7B-Instruct-v0.3 via HF Inference API (Apache 2.0).

COUNTERBALANCE: Version B runs annotation → elicitation for ~7/15 participants.
  Comparing rationale depth + label distribution = ACL methodological contribution.
"""

# import streamlit as st
# import json
# import os
# import re
# import requests
# from datetime import datetime
# from huggingface_hub import InferenceClient
# from streamlit_gsheets import GSheetsConnection
# import pandas as pd

import streamlit as st
import json
import re
import random
from datetime import datetime
from huggingface_hub import InferenceClient
from streamlit_gsheets import GSheetsConnection
import pandas as pd

# ─── Config ───────────────────────────────────────────────────────────────────

# HF_TOKEN = os.environ.get("HF_TOKEN", "INSERT_HF_TOKEN_HERE")
# Safely fetch the token
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    # Fallback to local environment variable if secrets aren't set
    HF_TOKEN = os.environ.get("HF_TOKEN", "INSERT_HF_TOKEN_HERE")

# Initialize the Hugging Face client
client = InferenceClient(api_key=HF_TOKEN)
# HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
HF_API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
# STORAGE_DIR = "pilot_data"

# ─── HatEval Datapoints ───────────────────────────────────────────────────────
# Source: Basile et al. (2019), SemEval-2019 Task 5, English training split.
# Sampled via multi-classifier disagreement + entropy (see docstring above).
# HS=hate speech, TR=target individual (1) or group (0), AG=aggressive.
# URLs stripped; text lightly cleaned for annotation display clarity.

HATEVAL_DATAPOINTS = [
    # ── Scenario A: Border security & national identity rhetoric ──────────────
    # High classifier disagreement: explicitly nativist language, wall-building.
    # cardiffnlp_hate_prob > 0.91 on all; dynabench_hate_prob < 0.10 on all.
    {"id": 4375, "scenario": "A", "hs": 1, "tr": 0, "ag": 1, "combined_score": 0.5457,
     "text": "Shooter opens fire on Border Patrol agent in San Ysidro — reason why we need this Wall. #buildThatWall"},
    {"id": 4088, "scenario": "A", "hs": 1, "tr": 0, "ag": 1, "combined_score": 0.5447,
     "text": "Protecting the COUNTRY is the sole purpose of the Military. Not for unnecessary foreign endeavours. The COUNTRY starts at the Border. #BuildThatWall"},
    {"id": 1185, "scenario": "A", "hs": 1, "tr": 0, "ag": 0, "combined_score": 0.5437,
     "text": "Please visit today to honor the thousands of victims killed by illegal aliens! We are building a memorial wall. #BuildThatWall"},
    {"id": 3104, "scenario": "A", "hs": 1, "tr": 0, "ag": 1, "combined_score": 0.5435,
     "text": "#BuildThatWall — 100% of your donation goes to secure America's southern border! Together we can save taxpayer money, make our communities safer, and build American Pride!"},
    {"id": 4149, "scenario": "A", "hs": 1, "tr": 0, "ag": 0, "combined_score": 0.5446,
     "text": "In the UK, we're full. Nearly 70 million people on a tiny island that's already the most densely populated country in Europe. We don't need more immigration."},

    # ── Scenario B: Refugees, humanitarian crisis & family separation ─────────
    # HS=0 throughout yet cardiffnlp_hate_prob 0.84–0.96 — high false-positive ambiguity.
    # These posts are factual/advocacy; classifier disagreement is about framing, not content.
    {"id": 4368, "scenario": "B", "hs": 0, "tr": 0, "ag": 0, "combined_score": 0.5480,
     "text": "Did this woman and child drown because the EU-backed Libyan coastguard scuttled their boat after they refused to return to Libya — where intercepted refugees face inhumane detention?"},
    {"id": 3069, "scenario": "B", "hs": 0, "tr": 0, "ag": 0, "combined_score": 0.5471,
     "text": "New Poll: Majority of Republican voters support separating families to deter illegal immigration."},
    {"id": 3324, "scenario": "B", "hs": 0, "tr": 0, "ag": 0, "combined_score": 0.5426,
     "text": "The U.S. has inflicted lasting trauma and pain upon immigrant families forced to live in horrifying uncertainty. It has to end now. #FamiliesBelongTogether #EndFamilySeparation"},
    {"id": 1245, "scenario": "B", "hs": 0, "tr": 0, "ag": 0, "combined_score": 0.5430,
     "text": "Imagine injured migrant workers who have no family to take care of them and can't afford medical scans."},
    {"id": 872, "scenario": "B", "hs": 0, "tr": 0, "ag": 0, "combined_score": 0.5426,
     "text": "Developing countries host 85% of the world's refugees."},

    # ── Scenario C: Cultural/religious belonging & counter-speech ─────────────
    # Mix of counter-speech (1872), irony (926), Islamophobia framing (493),
    # policy reporting (4287, 3824). Highest combined_score cluster in the sample.
    {"id": 926, "scenario": "C", "hs": 0, "tr": 0, "ag": 0, "combined_score": 0.5507,
     "text": "Just back from Bilbao where flags say 'Refugees Welcome' — a society that embraces diversity. Returned to RW smears and Tories doing what they want. Grim."},
    {"id": 493, "scenario": "C", "hs": 1, "tr": 0, "ag": 0, "combined_score": 0.5504,
     "text": "Why is it OK to make offensive comments about Christianity, but similar comments about Islam are called Islamophobic by the mainstream media? #Burqa #Brexit #immigration"},
    {"id": 4287, "scenario": "C", "hs": 0, "tr": 0, "ag": 0, "combined_score": 0.5461,
     "text": "We are officially no longer a nation of immigrants. U.S. Citizenship and Immigration Services will remove 'nation of immigrants' from their mission statement."},
    {"id": 3824, "scenario": "C", "hs": 0, "tr": 0, "ag": 0, "combined_score": 0.5419,
     "text": "UK rejects Christian refugees recommended by the UN, admits only Muslims among 1,112 Syrians admitted Jan–March 2018."},
    {"id": 1872, "scenario": "C", "hs": 0, "tr": 0, "ag": 0, "combined_score": 0.5419,
     "text": "The best tool against anti-immigrant sentiment is no safe space for white supremacists — be they legislators or poor folks. Shame them, smash them."},
]

# ─── Scenarios ────────────────────────────────────────────────────────────────
# Each scenario is a standalone vignette to elicit lived experience.
# Vignettes are thematically grounded in the HatEval tweet clusters but contain NO tweets.
# Purpose: activate relevant autobiographical memory before annotation (Smythe 2008).
# Design: Barter & Renold (1999) — 80–150 words, theory-grounded, max 3, avoid fatigue.
# Assignment: round-robin by sign-up order → 5 participants per scenario.

SCENARIOS = [
    {
        "id": "A",
        "theme": "Borders, Safety & Who Belongs",
        "vignette": (
            "A local community board is holding a heated town hall meeting about a proposed temporary "
            "housing facility for undocumented migrants. Outside the building, two groups have gathered. "
            "One group holds signs demanding strict border enforcement, shouting that the town must "
            "prioritize local safety and resources first. The other group is distributing hot tea and "
            "holding banners that read 'No one is illegal.' As everyday residents walk past the crowd "
            "to get to the grocery store next door, the tension in the air is palpable, forcing everyone "
            "to silently choose a side, engage, or just keep their heads down."
        ),
        "elicitation_focus": "personal safety, national identity, belonging, or community ties",
        "hateval_ids": [4375, 4088, 1185, 3104, 4149],
        "opening_q": (
            "Reading through that passage — what stands out to you first? "
            "Is there something you connect with on a personal level?"
        ),
    },
    {
        "id": "B",
        "theme": "Crisis, Suffering & Who We Choose to See",
        "vignette": (
            "A viral news video appears on your social media timeline showing a family being separated "
            "at a border crossing. The comments section beneath the video is a warzone. Some users are "
            "posting detailed legal arguments, justifying the policy as a necessary deterrent for illegal "
            "crossings. Others are sharing links to humanitarian funds, expressing profound outrage and "
            "heartbreak. Buried deep in the middle of the feed, a former refugee shares a quiet, "
            "ten-word comment about remembering the day they lost their own mother in transit. "
            "It gets completely ignored under the wave of political arguments."
        ),
        "elicitation_focus": "solidarity, responsibility, fairness, or witnessing suffering",
        "hateval_ids": [4368, 3069, 3324, 1245, 872],
        "opening_q": (
            "Reading through that passage — what's the first thing that strikes you? "
            "Is there a word, an image, or a feeling that surfaces before anything else?"
        ),
    },
    {
        "id": "C",
        "theme": "Culture, Religion & the Politics of Belonging",
        "vignette": (
            "A popular local bakery posts a photo celebrating a religious holiday that is not widely "
            "observed by the majority of the town. Within hours, a prominent local figure screenshots "
            "the post, claiming the bakery is 'erasing our traditional culture' and promoting dangerous "
            "ideologies. Followers flood the bakery’s page with hostile reviews and calls for a boycott. "
            "In response, a coalition of neighborhood residents organizes a 'buy-out' day, lining up "
            "around the block to purchase pastries, hoping to drown out the hostility with a highly "
            "visible show of support and solidarity."
        ),
        "elicitation_focus": "cultural identity, religious belonging, dignity, or what 'home' means",
        "hateval_ids": [926, 493, 4287, 3824, 1872],
        "opening_q": (
            "Reading through that passage — what feels most alive to you in it? "
            "Is there something that resonates personally, or something that feels distant or foreign?"
        ),
    },
]

# SCENARIOS = [
#     {
#         "id": "A",
#         "theme": "Borders, Safety & Who Belongs",
#         "vignette": (
#             "In many countries, debates about immigration have become inseparable from debates "
#             "about national identity — who counts as a 'real' citizen, whether borders should be "
#             "fortified or opened, and how to talk about crime and safety without stoking fear. "
#             "These conversations happen in parliaments, on social media, and at kitchen tables. "
#             "For some people, they are abstract political questions. "
#             "For others, they touch directly on their own sense of place, safety, and belonging — "
#             "or on the experiences of people they love."
#         ),
#         "elicitation_focus": "personal safety, national identity, belonging, or community ties",
#         "hateval_ids": [4375, 4088, 1185, 3104, 4149],
#         "opening_q": (
#             "Reading through that passage — what stands out to you first? "
#             "Is there something you connect with on a personal level?"
#         ),
#     },
#     {
#         "id": "B",
#         "theme": "Crisis, Suffering & Who We Choose to See",
#         "vignette": (
#             "Every year, thousands of people make dangerous journeys across borders — "
#             "fleeing war, persecution, or poverty. Their stories reach the public through "
#             "news headlines, policy debates, and social media. Some people respond with calls "
#             "for solidarity and systemic change. Others argue for stricter controls or question "
#             "the scale of responsibility a country can or should take on. "
#             "In between, there are millions of individual human stories that rarely make it "
#             "into those debates — of families separated, of workers with no safety net, "
#             "of children born into uncertainty."
#         ),
#         "elicitation_focus": "solidarity, responsibility, fairness, or witnessing suffering",
#         "hateval_ids": [4368, 3069, 3324, 1245, 872],
#         "opening_q": (
#             "Reading through that passage — what resonates with you? "
#             "Is there a word, an image, or a feeling that surfaces before anything else?"
#         ),
#     },
#     {
#         "id": "C",
#         "theme": "Culture, Religion & the Politics of Belonging",
#         "vignette": (
#             "When people from different cultural or religious backgrounds share the same public space, "
#             "questions arise about whose customs, whose language, and whose identity gets centred — "
#             "and whose gets treated as foreign or suspect. "
#             "These debates often overlap with questions about free speech: what can be said, "
#             "what counts as criticism versus hostility, and who gets to decide. "
#             "For some, these are questions of integration and national cohesion. "
#             "For others, they are questions of dignity — of feeling welcome, seen, or safe "
#             "in the place they call home."
#         ),
#         "elicitation_focus": "cultural identity, religious belonging, dignity, or what 'home' means",
#         "hateval_ids": [926, 493, 4287, 3824, 1872],
#         "opening_q": (
#             "Reading through that passage — what feels most alive to you in it? "
#             "Is there something that resonates personally, or something that feels distant or foreign?"
#         ),
#     },
# ]

# ─── Storage ──────────────────────────────────────────────────────────────────

def init_participant(name: str) -> dict:
    """Initializes a new session state for a participant purely in memory."""
    return {
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
        # Randomly assign a scenario to balance the distribution
        "scenario_id": random.choice(SCENARIOS)["id"],
        "workflow_stage": "disclosure",
        "disclosure": {},
        "elicitation": [],
        "micronarrative": "",
        "annotations": [],
        "version": "A",
    }

def get_scenario(sid: str) -> dict:
    return next(s for s in SCENARIOS if s["id"] == sid)

def get_datapoints(sid: str) -> list:
    ids = set(get_scenario(sid)["hateval_ids"])
    return [d for d in HATEVAL_DATAPOINTS if d["id"] in ids]

# def get_participant_dir(name: str) -> str:
#     safe = re.sub(r"[^a-zA-Z0-9_-]", "_", name.strip().lower())
#     path = os.path.join(STORAGE_DIR, safe)
#     os.makedirs(path, exist_ok=True)
#     return path


# def load_participant(name: str) -> dict:
#     d = get_participant_dir(name)
#     fp = os.path.join(d, "session.json")
#     if os.path.exists(fp):
#         with open(fp) as f:
#             return json.load(f)
#     existing = [x for x in os.listdir(STORAGE_DIR) if os.path.isdir(os.path.join(STORAGE_DIR, x))]
#     scenario_idx = (len(existing) - 1) % len(SCENARIOS)
#     return {
#         "name": name,
#         "created_at": datetime.utcnow().isoformat(),
#         "scenario_id": SCENARIOS[scenario_idx]["id"],
#         "workflow_stage": "disclosure",
#         "disclosure": {},
#         "elicitation": [],
#         "micronarrative": "",
#         "annotations": [],
#         "version": "A",
#     }


# def save_participant(data: dict):
#     d = get_participant_dir(data["name"])
#     with open(os.path.join(d, "session.json"), "w") as f:
#         json.dump(data, f, indent=2)


# def get_scenario(sid: str) -> dict:
#     return next(s for s in SCENARIOS if s["id"] == sid)


# def get_datapoints(sid: str) -> list:
#     ids = set(get_scenario(sid)["hateval_ids"])
#     return [d for d in HATEVAL_DATAPOINTS if d["id"] in ids]

# ─── LLM ──────────────────────────────────────────────────────────────────────

def call_mistral(system_prompt: str, messages: list, max_tokens: int = 200) -> str:
    if HF_TOKEN == "INSERT_HF_TOKEN_HERE":
        return "[Set HF_TOKEN to enable the AI interviewer — see README.]"
    
    # Build the message list natively for the client
    formatted_messages = [{"role": "system", "content": system_prompt}] + messages
    
    try:
        # The client automatically handles the correct endpoint routing!
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct", 
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Model temporarily unavailable: {e}. Please try again.]"
# def call_mistral(system_prompt: str, messages: list, max_tokens: int = 200) -> str:
#     if HF_TOKEN == "INSERT_HF_TOKEN_HERE":
#         return "[Set HF_TOKEN to enable the AI interviewer — see README.]"
#     fmt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
#     for m in messages:
#         fmt += f"{m['content']} [/INST] " if m["role"] == "user" else f"{m['content']} [INST] "
#     fmt = fmt.rstrip(" [INST] ")
#     try:
#         r = requests.post(
#             HF_API_URL,
#             headers={"Authorization": f"Bearer {HF_TOKEN}"},
#             json={"inputs": fmt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.7, "return_full_text": False}},
#             timeout=50,
#         )
#         r.raise_for_status()
#         res = r.json()
#         return res[0].get("generated_text", "").strip() if isinstance(res, list) else str(res)
#     except Exception as e:
#         return f"[Model temporarily unavailable: {e}. Please try again.]"

# ─── Elicitation prompt builder ───────────────────────────────────────────────
# Implements Willig (2013) turn structure + Rocchio (2022) domain-anchoring
# + Smythe (2008) hermeneutic circle from turn 3 onward.

# def elicitation_sys(scenario: dict, user_turns: int, last_user: str = "") -> str:
#     focus = scenario["elicitation_focus"]
#     p = (
#         "You are a research interviewer. Your only job is to ask ONE good follow-up question "
#         "that helps the participant explore their lived experience — not their political opinions. "
#         "Strict rules: 2–3 sentences maximum; do not summarize what they said; do not ask multiple questions; "
#         "do not use vague prompts like 'tell me more' or 'can you elaborate'; "
#         f"anchor your question to one specific axis from: {focus}. "
#         "A good question names something concrete the person just said and asks what made it personally significant."
#     )
#     if user_turns == 1:
#         # Turn 1: Harm/target cue (Willig 2013)
#         p += (
#             " This turn specifically: ask who, if anyone, they see as most affected by the tensions "
#             "described in the scenario — and whether that connects to anyone in their own life."
#         )
#     elif user_turns == 2:
#         # Turn 2: Domain-driver (Rocchio 2022) — give axes explicitly so they can pick
#         p += (
#             f" This turn specifically: offer them the axes explicitly — {focus} — and ask which one "
#             "feels most personally alive right now, and why. Giving them the options helps people "
#             "surface values they wouldn't otherwise name."
#         )
#     elif user_turns >= 3:
#         # Turn 3+: Hermeneutic circle — reference prior answer (Smythe 2008)
#         p += (
#             f' This turn specifically: reference something from what they just said — '
#             f'"{last_user[:140]}" — and ask what makes that personally significant. '
#             "Not just what happened, but why it matters or what it meant to them."
#         )
#     if user_turns == 4 or user_turns == 5:
#         p += (
#             "\n\nIf the participant has touched on at least one lived-experience axis with "
#             "personal detail (enough for a 4–5 sentence narrative), end your message with: READY_TO_BUILD"
#         )
#     elif user_turns >= 6:
#         p += (
#             "\n\nCRITICAL INSTRUCTION: This is the absolute final turn. You must warmly thank the participant "
#             "for sharing in 1 short sentence, and you MUST end your message with the exact phrase: READY_TO_BUILD"
#         )

    # return p

# ─── Elicitation prompt builder ───────────────────────────────────────────────

def elicitation_sys(scenario: dict, user_turns: int, last_user: str = "") -> str:
    axes_context = (
        "1. Sense of Self (introspective reflection, personal identity, memory)\n"
        "2. Social & Cultural (community ties, cultural background, moral alignment)\n"
        "3. Wellbeing (emotional impact, mental health, feelings of safety)"
    )

    p = (
        "You are an empathetic qualitative research interviewer. "
        "The participant just read the following scenario:\n"
        f"SCENARIO: \"{scenario['vignette']}\"\n\n"
        "Your goal is to guide them to reflect on this scenario through specific dimensions of their own lived experience.\n"
        f"LIVED EXPERIENCE AXES:\n{axes_context}\n\n"
        "RULES:\n"
        "- 1 to 2 sentences maximum.\n"
        "- Never summarize what they just said.\n"
        "- Never ask more than one question at a time.\n\n"
    )

    # Turn-specific instructions
    if user_turns == 1:
        p += (
            "CURRENT STAGE: TURN 1 (Initial Reaction).\n"
            "INSTRUCTION: Identify the core emotion they shared. Ask a single question connecting that reaction "
            "to ONE of the Lived Experience Axes (e.g., how it ties into their community, or their sense of safety)."
        )
    elif user_turns == 2:
        p += (
            "CURRENT STAGE: TURN 2 (Pivot).\n"
            "INSTRUCTION: Gently pivot to a DIFFERENT axis from the list that they haven't explored yet. "
        )
    elif user_turns >= 3:
        p += (
            "CURRENT STAGE: TURN 3+ (Deepening).\n"
            f"INSTRUCTION: Reference a specific detail they mentioned earlier \"{last_user}...\" "
            "Ask them to provide a specific memory or concrete example from their own life that explains WHY they feel that way."
        )

    # Stopping logic
    if user_turns == 4 or user_turns == 5:
        p += (
            "\n\nSTOPPING CONDITION: If the participant has provided enough personal detail to form a coherent "
            "4-5 sentence story, you MUST end your response with the exact text: READY_TO_BUILD"
        )
    elif user_turns >= 6:
        p += (
            "\n\nCRITICAL OVERRIDE: This is the absolute final turn (Turn 6). You are strictly forbidden from asking "
            "another question. You must warmly thank the participant in 1 sentence, and append the exact text: READY_TO_BUILD"
        )

    return p

SYNTHESIS_SYS = (
    "Write a single first-person micronarrative (4–5 sentences, ~80–100 words) based on the participant's responses. "
    "Rules: use 'I' throughout; preserve the participant's own specific words and framing; "
    "do NOT add feelings or interpretations they didn't express; "
    "friendly, natural tone — not clinical or academic; "
    "output ONLY the narrative, no preamble or explanation."
)

# ─── Styles ───────────────────────────────────────────────────────────────────

def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background: #000000; color: #1A1A1A; }
    h1, h2 { font-family: 'Lora', serif; font-weight: 400; letter-spacing: -0.01em; }
    .stApp { background: #000000; }
    .vignette-card {
        background: #EEEAE0; border-left: 4px solid #8B6F47;
        padding: 1.2rem 1.5rem; border-radius: 3px; margin: 1rem 0 1.4rem 0;
        font-size: 0.96rem; line-height: 1.85; color: #2A2A2A; font-style: italic;
    }
                .tweet-card {
    background: #111111;              /* dark card instead of white */
    border: 1px solid #2A2A2A;        /* subtle border */
    border-radius: 10px;              /* slightly softer */
    padding: 1.2rem 1.4rem;
    font-size: 1rem;
    line-height: 1.7;
    color: #EAEAEA;                  /* readable text */
    box-shadow: 0 2px 8px rgba(0,0,0,0.6);  /* deeper shadow for depth */
    margin-bottom: 1.2rem;
}

.meta-pill {
    display: inline-block;
    background: #222;                /* darker pill */
    border-radius: 12px;
    font-size: 0.7rem;
    padding: 3px 10px;
    color: #B5B5B5;                  /* softer gray text */
    margin-right: 6px;
}
    # .tweet-card {
    #     background: #fff; border: 1px solid #E0DBD0; border-radius: 8px;
    #     padding: 1.1rem 1.3rem; font-size: 1rem; line-height: 1.65;
    #     box-shadow: 0 1px 4px rgba(0,0,0,0.06); margin-bottom: 1.2rem;
    # }
    # .meta-pill {
    #     display: inline-block; background: #E8E4DC; border-radius: 12px;
    #     font-size: 0.7rem; padding: 2px 9px; color: #777; margin-right: 5px;
    # }
    .prog-bg { background: #DDD9D0; border-radius: 20px; height: 5px; margin: 0.4rem 0 1.4rem 0; }
    .prog-fill { background: #8B6F47; height: 5px; border-radius: 20px; }
    .step-label { font-size: 0.71rem; color: #999; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.2rem; }
    div[data-testid="stChatMessage"] { background: transparent !important; }
    </style>
    """, unsafe_allow_html=True)
    


def prog(step, total):
    pct = int(step / total * 100)
    st.markdown(f"<div class='prog-bg'><div class='prog-fill' style='width:{pct}%'></div></div>", unsafe_allow_html=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Immigration & Belonging — Pilot Study", layout="centered", page_icon="🗣")
    inject_styles()
    # os.makedirs(STORAGE_DIR, exist_ok=True)

    # # Sign-in
    # if "participant_name" not in st.session_state:
    #     st.markdown("<div class='step-label'>Pilot Study · Version A</div>", unsafe_allow_html=True)
    #     st.title("Immigration & Belonging")
    #     st.markdown("*A positionality-aware annotation study*")
    #     st.markdown("---")
    #     st.markdown(
    #         "This study takes about **15–20 minutes**. You'll first share a bit about your own "
    #         "perspective on immigration and belonging, then annotate a small set of social media posts."
    #     )
    #     name = st.text_input("Enter your first name or a pseudonym — or the same name as before (if you have done this pilot study before) to continue:")
    #     if st.button("Begin →", type="primary") and name.strip():
    #         st.session_state.participant_name = name.strip()
    #         st.session_state.pdata = load_participant(name.strip())
    #         st.rerun()
    #     return

    # if "pdata" not in st.session_state:
    #     st.session_state.pdata = load_participant(st.session_state.participant_name)

    # data = st.session_state.pdata
    # scenario = get_scenario(data["scenario_id"])
    # stage = data["workflow_stage"]

    # with st.sidebar:
    #     st.markdown(f"**{data['name']}**")
    #     st.markdown(f"*{scenario['theme']}*")
    #     labels = {"disclosure": "1 — Background", "elicitation_chat": "2 — Your experience",
    #               "synthesis": "3 — Your narrative", "annotation": "4 — Annotations", "complete": "✓ Done"}
    #     st.caption(labels.get(stage, stage))
    #     st.markdown("---")
    #     if st.button("Save & pause"):
    #         save_participant(data)
    #         st.success("Saved. Return any time with the same name.")

    # Sign-in
    if "participant_name" not in st.session_state:
        st.markdown("<div class='step-label'>Pilot Study · Version A</div>", unsafe_allow_html=True)
        st.title("Immigration & Belonging")
        st.markdown("*A positionality-aware annotation study*")
        st.markdown("---")
        st.markdown(
            "This study takes about **15–20 minutes**. You'll first share a bit about your own "
            "perspective on immigration and belonging, then annotate a small set of social media posts."
        )
        name = st.text_input("Enter your first name or a pseudonym:")
        if st.button("Begin →", type="primary") and name.strip():
            st.session_state.participant_name = name.strip()
            st.session_state.pdata = init_participant(name.strip())
            st.rerun()
        return

    if "pdata" not in st.session_state:
        st.session_state.pdata = init_participant(st.session_state.participant_name)

    data = st.session_state.pdata
    scenario = get_scenario(data["scenario_id"])
    stage = data["workflow_stage"]

    with st.sidebar:
        st.markdown(f"**{data['name']}**")
        st.markdown(f"*{scenario['theme']}*")
        labels = {"disclosure": "1 — Background", "elicitation_chat": "2 — Your experience",
                  "synthesis": "3 — Your narrative", "annotation": "4 — Annotations", "complete": "✓ Done"}
        st.caption(labels.get(stage, stage))
        st.markdown("---")
        st.caption("Data is temporarily held in memory and saved securely at the end.")

    # ── STAGE 1: DISCLOSURE ───────────────────────────────────────────────────
    if stage == "disclosure":
        st.markdown("<div class='step-label'>Step 1 of 4</div>", unsafe_allow_html=True)
        st.title("A bit about you")
        prog(1, 4)
        st.write("Before we begin, we'd like to understand your connection to this topic. This context shapes how we interpret your annotations — it won't affect your participation.")

        conn = st.selectbox(
            "How would you describe your connection to topics of immigration or belonging?",
            ["I have direct personal experience (as an immigrant, refugee, or child of one)",
             "I'm a caregiver, partner, or close community member of someone with this experience",
             "I work or study in this area professionally or academically",
             "I'm an interested observer — no direct personal connection"],
        )
        duration = st.text_input("How long has this been part of your life or work? (e.g., 'my whole life', '3 years')")
        disclosure = st.text_area(
            "Briefly describe how this topic relates to your life. "
            "This stays confidential and helps us contextualise your responses.",
            height=100, placeholder="e.g. My parents immigrated to Canada in the 1990s...",
        )
        if st.button("Continue →", type="primary"):
            data["disclosure"] = {"connection_type": conn, "duration": duration, "text": disclosure}
            data["workflow_stage"] = "elicitation_chat"
            # save_participant(data)
            st.rerun()

    # ── STAGE 2: ELICITATION ──────────────────────────────────────────────────
    elif stage == "elicitation_chat":
        st.markdown("<div class='step-label'>Step 2 of 4 — Your Experience</div>", unsafe_allow_html=True)
        st.title("Your perspective")
        prog(2, 4)
        st.markdown(f"<div class='vignette-card'>{scenario['vignette']}</div>", unsafe_allow_html=True)
        st.caption("Read the passage above, then respond below. Aim for 2-3 sentences per reply.")

        for msg in data["elicitation"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if not data["elicitation"]:
            data["elicitation"].append({"role": "assistant", "content": scenario["opening_q"]})
            # save_participant(data)
            st.rerun()
        
        if user_input := st.chat_input("Your response…"):
            data["elicitation"].append({"role": "user", "content": user_input})
            # save_participant(data)
            user_turns = sum(1 for m in data["elicitation"] if m["role"] == "user")
            st.caption(f"*(Debug: AI is currently on Turn {user_turns})*")
            sys_p = elicitation_sys(scenario, user_turns, user_input)
            response = call_mistral(sys_p, data["elicitation"], max_tokens=160)
            if "READY_TO_BUILD" in response:
                clean = response.replace("READY_TO_BUILD", "").strip()
                if clean:
                    data["elicitation"].append({"role": "assistant", "content": clean})
                data["workflow_stage"] = "synthesis"
            else:
                data["elicitation"].append({"role": "assistant", "content": response})
            # save_participant(data)
            st.rerun()

    # ── STAGE 3: SYNTHESIS ────────────────────────────────────────────────────
    elif stage == "synthesis":
        st.markdown("<div class='step-label'>Step 3 of 4 — Your Narrative</div>", unsafe_allow_html=True)
        st.title("Your story, in your words")
        prog(3, 4)
        st.write("Based on what you shared, we've drafted a short narrative in your voice. Edit anything that doesn't feel accurate, then accept it to continue.")

        if not data.get("micronarrative"):
            with st.spinner("Drafting your narrative…"):
                fragments = "\n".join(m["content"] for m in data["elicitation"] if m["role"] == "user")
                data["micronarrative"] = call_mistral(SYNTHESIS_SYS, [{"role": "user", "content": fragments}], max_tokens=200)
                # save_participant(data)

        edited = st.text_area("Your narrative (edit freely):", value=data["micronarrative"], height=190)
        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("↺ Regenerate"):
                data["micronarrative"] = ""
                # save_participant(data)
                st.rerun()
        with c2:
            if st.button("Accept & start annotations →", type="primary"):
                data["micronarrative"] = edited
                data["workflow_stage"] = "annotation"
                # save_participant(data)
                st.rerun()

    # ── STAGE 4: ANNOTATION ───────────────────────────────────────────────────
    elif stage == "annotation":
        datapoints = get_datapoints(data["scenario_id"])
        idx = len(data["annotations"])

        if idx < len(datapoints):
            dp = datapoints[idx]
            st.markdown(f"<div class='step-label'>Step 4 of 4 — Post {idx+1} of {len(datapoints)}</div>", unsafe_allow_html=True)
            st.title("Annotating social media posts")
            prog(idx + 1, len(datapoints))
            st.write("You'll now read and annotate five social media posts related to immigration. There are no right or wrong answers — your interpretation is the data.")

            st.markdown(
                f"<div class='tweet-card'>"
                f"<span class='meta-pill'>Post {idx+1} of {len(datapoints)}</span>"
                # f"<span class='meta-pill'>ambiguity score: {dp['combined_score']:.2f}</span>"
                f"<br><br>{dp['text']}</div>",
                unsafe_allow_html=True,
            )

            with st.form(f"anno_{idx}"):
                st.markdown("**1. How would you label this post?**")
                label = st.radio("", [
                    "Hate speech — dehumanises or incites harm toward a person or group",
                    "Borderline — potentially harmful, but depends on context or intent",
                    "Not hate speech — critical, blunt, or politically charged but not hateful",
                ], label_visibility="collapsed")

                st.markdown("**2. If harm is present, who seems most targeted?**")
                target = st.radio("", [
                    "A specific individual",
                    "Immigrants or refugees as a group",
                    "A religious or ethnic community",
                    "No specific target — it's about a policy or event",
                    "Unclear",
                ], label_visibility="collapsed")

                st.markdown(
                    "**3. Your rationale** — 3–4 sentences. "
                    "What specific words or framing led to your label? "
                    "What makes this feel harmful, ambiguous, or acceptable?"
                )
                rationale = st.text_area("", height=140, label_visibility="collapsed",
                    placeholder="e.g. The phrase '...' implies... because... What makes this different from policy criticism is...")

                # Positionality salience — reflexivity framing (Smythe 2008), not bias admission
                st.markdown("**4. How relevant did your own background feel as you read this post?**")
                st.caption("1 = not at all relevant to my experience · 5 = very relevant")
                salience = st.slider("", 1, 5, 3, label_visibility="collapsed")

                if st.form_submit_button("Submit & next →", type="primary"):
                    words = len(rationale.strip().split())
                    if words < 35:
                        st.warning(f"Your rationale is {words} words — please expand to 3–4 full sentences (aim for 35+ words). The detail you provide is the most valuable part of the study.")
                    else:
                        data["annotations"].append({
                            "tweet_id": dp["id"], "tweet_text": dp["text"],
                            "hateval_hs": dp["hs"], "hateval_ag": dp["ag"], "hateval_tr": dp["tr"],
                            "combined_score": dp["combined_score"],
                            "participant_label": label, "participant_target": target,
                            "rationale": rationale, "positionality_salience": salience,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                        # save_participant(data)
                        st.rerun()
        # else:
        #     data["workflow_stage"] = "complete"
        #     # save_participant(data)
        #     st.rerun()
        else:
            # --- ONE-TIME GOOGLE SHEETS SAVE ---
            with st.spinner("Saving your responses securely..."):
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    
                    new_row = {
                        "Name": data["name"],
                        "Timestamp": datetime.now().isoformat(),
                        "Scenario": data["scenario_id"],
                        "Connection_Type": data["disclosure"].get("connection_type", ""),
                        "Duration": data["disclosure"].get("duration", ""),
                        "Disclosure_Text": data["disclosure"].get("text", ""),
                        "Micronarrative": data["micronarrative"],
                        "Chat_Log": json.dumps(data["elicitation"]),
                        "Annotations": json.dumps(data["annotations"])
                    }
                    
                    # REPLACE THIS URL WITH YOUR ACTUAL SHEET URL
                    SHEET_URL = "https://docs.google.com/spreadsheets/d/1xAvNGAvny-1uCS2s2Iw4ij5OG1gF1LjKAdbLlcDnAkM/edit"
                    
                    # existing_data = conn.read(spreadsheet=SHEET_URL, usecols=list(new_row.keys()))
                    existing_data = conn.read(spreadsheet=SHEET_URL, usecols=list(new_row.keys()), ttl=0)
                    updated_data = pd.concat([existing_data, pd.DataFrame([new_row])], ignore_index=True)
                    
                    conn.update(spreadsheet=SHEET_URL, data=updated_data)
                    
                    data["workflow_stage"] = "complete"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save to database. Please leave this window open and contact the researcher. Error: {e}")

    # ── COMPLETE ──────────────────────────────────────────────────────────────
    elif stage == "complete":
        st.balloons()
        st.title("Thank you.")
        st.markdown(
            f"Your annotations and narrative are saved, **{data['name']}**. "
            "The perspectives you bring — including your background and lived experience — "
            "are what makes this kind of research meaningful."
        )
        st.caption("Pilot study by Sheza Munir · Data stored locally · You may close this window.")


if __name__ == "__main__":
    main()
