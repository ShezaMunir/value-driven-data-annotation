"""
pilot_study.py — Ambiguous Hate Speech Positionality Pilot Study (Version A: Elicitation → Annotation)
Researcher: Sheza Munir

═══════════════════════════════════════════════════════════════════════════════
THEORETICAL GROUNDING (not shown to participants)
═══════════════════════════════════════════════════════════════════════════════

ORDERING — Elicitation FIRST, annotation SECOND.
  Smythe et al. (2008) "Methodological Considerations for Interpretive Phenomenological
  Inquiry" — narrative/reflective mindset before evaluation tasks enriches rationale depth.
  Annotation-first activates cold "judge mode" that suppresses lived-experience disclosure.

ELICITATION DESIGN
  • Lived-experience axes (critical computing / NLP / HCI framework):
      1. Sociocultural & Geographic Context
      2. Linguistic Background & Dialect
      3. Socioeconomic Status & Labor Dynamics
      4. Race & Ethnicity
      5. Gender Identity & Sexual Orientation
      6. Disability & Neurodivergence
      7. Domain Expertise vs. Epistemic Proximity
  • 5-turn hard cap with graceful wind-down at turn 4.
  • Vignette construction: Barter & Renold (1999) — 80–150 words, grounded in dataset
    thematic clusters, max 3 vignettes to avoid fatigue.
  • Iterative probing: Willig (2013) — initial reaction → harm/target cue → axis deepening.
  • Hermeneutic circle: Smythe (2008) — later prompts reference earlier answers.

ANNOTATION DESIGN
  • 10 posts per scenario from the ambiguous hate speech dataset (Munir 2024).
    Sampled across 4 domains: Immigration/Nativism, Religion, Gender/Sexuality, Intersectional.
  • All items satisfy C1–C4 of the ambiguity codebook (surface plausible deniability,
    experiential divergence, discourse embeddedness, no explicit derogation).
  • Positionality salience reframed as identity resonance (reflexivity, not bias admission).
  • Rationale minimum: 50 words.

MICRONARRATIVE: Boyle & Butcher (2024) — fidelity to participant's own words,
  no reinterpretation, first person, warm tone, minimum 120 words.

MODEL: Qwen/Qwen2.5-7B-Instruct via HF Inference API.
"""

import streamlit as st
import json
import random
from datetime import datetime
from huggingface_hub import InferenceClient
from streamlit_gsheets import GSheetsConnection
import time
import pandas as pd
import os

# ─── Config ───────────────────────────────────────────────────────────────────

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except KeyError:
    HF_TOKEN = os.environ.get("HF_TOKEN", "INSERT_HF_TOKEN_HERE")

client = InferenceClient(api_key=HF_TOKEN)

SHEET_URL = "https://docs.google.com/spreadsheets/d/1xAvNGAvny-1uCS2s2Iw4ij5OG1gF1LjKAdbLlcDnAkM/edit"

MIN_RATIONALE_WORDS = 50

# ─── Dataset ──────────────────────────────────────────────────────────────────
# Source: Munir (2024) Ambiguous Hate Speech Dataset.
# Domain: IMM = Immigration/Nativism, REL = Religion, GEN = Gender/Sexuality, INT = Intersectional.
# All items satisfy C1–C4 of the ambiguity codebook (surface plausible deniability,
# experiential divergence, discourse embeddedness, no explicit derogation).

DATAPOINTS = [
    # ── Scenario A: Borders, Safety & Who Belongs ──────────────────────────────
    # Theme: nativist plausible-deniability framing, demographic anxiety, rule-of-law rhetoric.
    # High C2 divergence: in-group (immigrant/minority) vs. out-group readings diverge substantively.
    {"id": "A01", "scenario": "A", "domain": "IMM",
     "text": "Our hospitals can't keep up. At some point we have to ask who this country is actually for."},
    {"id": "A02", "scenario": "A", "domain": "IMM",
     "text": "My grandparents came here legally and waited years. Why should anyone get to skip that line?"},
    {"id": "A03", "scenario": "A", "domain": "IMM",
     "text": "When you flood a labor market, wages go down. Economics doesn't care about feelings."},
    {"id": "A04", "scenario": "A", "domain": "IMM",
     "text": "There's nothing wrong with wanting your hometown to stay recognizable."},
    {"id": "A05", "scenario": "A", "domain": "IMM",
     "text": "Demographic change at this pace isn't natural. Someone is choosing this."},
    {"id": "A06", "scenario": "A", "domain": "IMM",
     "text": "Chain migration turns one visa into fifty. That's not the deal Americans signed up for."},
    {"id": "A07", "scenario": "A", "domain": "IMM",
     "text": "Every nation has the right to exist as a distinct people. Even European ones."},
    {"id": "A08", "scenario": "A", "domain": "IMM",
     "text": "We keep importing poverty and then wonder why poverty doesn't go away."},
    {"id": "A09", "scenario": "A", "domain": "IMM",
     "text": "Western countries are the only ones expected to dissolve their own majorities."},
    {"id": "A10", "scenario": "A", "domain": "IMM",
     "text": "The people most affected by immigration policy are the working class, not politicians."},

    # ── Scenario B: Religion, Culture & the Politics of Belonging ──────────────
    # Theme: religious double-standards, secular liberal framing, epistemic proximity.
    # High C2 divergence: Muslim/religious minority annotators read threat-coding invisible to others.
    {"id": "B01", "scenario": "B", "domain": "REL",
     "text": "I support religious freedom, but some belief systems just aren't compatible with liberal democracy. That's not bigotry, it's observation."},
    {"id": "B02", "scenario": "B", "domain": "REL",
     "text": "We accommodate every faith's dietary rules in schools now. At some point you have to ask whose schools these are."},
    {"id": "B03", "scenario": "B", "domain": "REL",
     "text": "I have no problem with people praying five times a day. I do have a problem when it starts affecting workplace schedules for everyone else."},
    {"id": "B04", "scenario": "B", "domain": "REL",
     "text": "There's a reason the Enlightenment happened in one part of the world and not others. That's not racism, it's history."},
    {"id": "B05", "scenario": "B", "domain": "REL",
     "text": "If a Christian baker refusing a gay wedding is front-page news, why isn't a halal butcher refusing to serve a non-Muslim customer?"},
    {"id": "B06", "scenario": "B", "domain": "REL",
     "text": "Secularism means no religion gets special status. That includes the ones progressives have decided are beyond criticism."},
    {"id": "B07", "scenario": "B", "domain": "INT",
     "text": "Immigrant communities often hold the most conservative views on gender. The left just doesn't want to say that out loud."},
    {"id": "B08", "scenario": "B", "domain": "INT",
     "text": "Gay rights were won by fighting religious conservatism. Now we're importing more of it and calling it diversity."},
    {"id": "B09", "scenario": "B", "domain": "INT",
     "text": "Female genital mutilation is illegal here. Calling attention to where it's practiced is not Islamophobia."},
    {"id": "B10", "scenario": "B", "domain": "INT",
     "text": "The same activists who say believe all women go very quiet when the accused is from a protected group."},

    # ── Scenario C: Gender, Identity & Whose Experience Counts ────────────────
    # Theme: gender essentialism, weaponized empiricism, concern-trolling, intersectional erasure.
    # C2 divergence fractured: trans/NB, cisgender feminist, and non-feminist readers diverge.
    {"id": "C01", "scenario": "C", "domain": "GEN",
     "text": "I believe women. I also believe due process. These shouldn't be mutually exclusive."},
    {"id": "C02", "scenario": "C", "domain": "GEN",
     "text": "Calling everything a 'microaggression' doesn't empower women. It treats them like they can't handle everyday life."},
    {"id": "C03", "scenario": "C", "domain": "GEN",
     "text": "There are two genders. I'm sorry if that's hurtful, but being hurtful isn't the same as being wrong."},
    {"id": "C04", "scenario": "C", "domain": "GEN",
     "text": "If gender is a social construct, then so is the wage gap. You can't have it both ways."},
    {"id": "C05", "scenario": "C", "domain": "GEN",
     "text": "Women's sports existed to give women a fair playing field. That's it. That's the whole point."},
    {"id": "C06", "scenario": "C", "domain": "GEN",
     "text": "Men aren't broken women. Treating masculinity as a pathology isn't progressivism, it's prejudice with better vocabulary."},
    {"id": "C07", "scenario": "C", "domain": "INT",
     "text": "When a working-class white man loses a job to a diversity hire, he's not allowed to be angry. That anger has to go somewhere."},
    {"id": "C08", "scenario": "C", "domain": "INT",
     "text": "Intersectionality is just a hierarchy that puts some victims above others. It's oppression with extra steps."},
    {"id": "C09", "scenario": "C", "domain": "INT",
     "text": "First-generation immigrant women are often the most opposed to feminism. Nobody ever asks them why."},
    {"id": "C10", "scenario": "C", "domain": "INT",
     "text": "Jewish students are told their discomfort with certain political spaces doesn't count as discrimination. Wonder why."},
]

# ─── Scenarios ────────────────────────────────────────────────────────────────

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
        "datapoint_ids": ["A01","A02","A03","A04","A05","A06","A07","A08","A09","A10"],
        "opening_q": (
            "Reading through that passage — what stands out to you first? "
            "Is there something you connect with on a personal level, or something that feels distant?"
        ),
    },
    {
        "id": "B",
        "theme": "Religion, Culture & the Politics of Belonging",
        "vignette": (
            "A popular local bakery posts a photo celebrating a religious holiday that is not widely "
            "observed by the majority of the town. Within hours, a prominent local figure screenshots "
            "the post, claiming the bakery is 'erasing our traditional culture' and promoting dangerous "
            "ideologies. Followers flood the bakery's page with hostile reviews and calls for a boycott. "
            "In response, a coalition of neighbourhood residents organises a 'buy-out' day, lining up "
            "around the block to purchase pastries, hoping to drown out the hostility with a highly "
            "visible show of support and solidarity."
        ),
        "elicitation_focus": "cultural identity, religious belonging, dignity, or what 'home' means",
        "datapoint_ids": ["B01","B02","B03","B04","B05","B06","B07","B08","B09","B10"],
        "opening_q": (
            "Reading through that passage — what feels most alive to you in it? "
            "Is there something that resonates personally, or something that feels uncomfortable or foreign?"
        ),
    },
    {
        "id": "C",
        "theme": "Gender, Identity & Whose Experience Counts",
        "vignette": (
            "A university student forum erupts after a well-known professor publishes an op-ed arguing "
            "that 'identity politics has gone too far' and that certain protected categories now receive "
            "more institutional sympathy than others. The comments split immediately: some students share "
            "personal accounts of discrimination they say went unacknowledged, while others post "
            "screenshots of what they describe as reverse bias in hiring and disciplinary processes. "
            "A graduate student quietly posts a one-sentence reply — 'I've never once felt my experience "
            "was counted the same as others here' — and it disappears under the wave of debate."
        ),
        "elicitation_focus": "identity, fairness, visibility, belonging, or feeling counted",
        "datapoint_ids": ["C01","C02","C03","C04","C05","C06","C07","C08","C09","C10"],
        "opening_q": (
            "Reading through that passage — what's the first thing that strikes you? "
            "Is there a moment, a feeling, or a person in it you find yourself drawn to?"
        ),
    },
]

# ─── Storage ──────────────────────────────────────────────────────────────────

def init_participant(name: str) -> dict:
    return {
        "name": name,
        "created_at": datetime.utcnow().isoformat(),
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
    ids = set(get_scenario(sid)["datapoint_ids"])
    return [d for d in DATAPOINTS if d["id"] in ids]

# ─── LLM ──────────────────────────────────────────────────────────────────────

def call_qwen(system_prompt: str, messages: list, max_tokens: int = 200) -> str:
    if HF_TOKEN == "INSERT_HF_TOKEN_HERE":
        return "[Set HF_TOKEN to enable the AI interviewer — see README.]"
    formatted_messages = [{"role": "system", "content": system_prompt}] + messages
    try:
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=formatted_messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Model temporarily unavailable: {e}. Please try again.]"

# ─── Elicitation prompt builder ───────────────────────────────────────────────
# Implements Willig (2013) turn structure + Rocchio (2022) domain-anchoring
# + Smythe (2008) hermeneutic circle from turn 3 onward.
# Hard 5-turn cap: wind-down at turn 4, mandatory READY_TO_BUILD at turn 5.

# AXES_CONTEXT = """\
# Use the following seven lived-experience axes as lenses to guide each question. \
# Do not name the axes explicitly — let your question naturally open up that dimension of experience:

# 1. Sociocultural & Geographic Context — where they grew up, cultural norms, value systems, urban/rural
# 2. Linguistic Background & Dialect — native language, multilingualism, how they navigate registers
# 3. Socioeconomic Status & Labor Dynamics — class, economic precarity, workplace power dynamics
# 4. Race & Ethnicity — racial identity, historical marginalization, how they are perceived by others
# 5. Gender Identity & Sexual Orientation — lived gender/sexuality, how systems categorize them
# 6. Disability & Neurodivergence — physical, cognitive, or sensory experience; what 'normal' excludes
# 7. Epistemic Proximity — how close or distant they personally are from the people most affected\
# """

# def elicitation_sys(scenario: dict, user_turns: int, last_user: str = "") -> str:
#     p = (
#         "You are an empathetic qualitative research interviewer conducting a lived-experience elicitation. "
#         "The participant just read this scenario:\n"
#         f"SCENARIO: \"{scenario['vignette']}\"\n\n"
#         f"{AXES_CONTEXT}\n\n"
#         "STRICT RULES:\n"
#         "- Ask exactly ONE question per turn. Never two questions in one response.\n"
#         "- 1–2 sentences maximum. No preamble, no summaries.\n"
#         "- Never paraphrase or reflect back what the participant just said.\n"
#         "- Never use vague prompts like 'tell me more' or 'can you elaborate'.\n"
#         "- Ground every question in something concrete the participant just said or implied.\n\n"
#     )

#     if user_turns == 1:
#         p += (
#             "TURN 1 — GROUNDING:\n"
#             "Identify the most emotionally charged or specific thing they said. "
#             "Ask a single question connecting that reaction to their relationship "
#             "to the people, place, or tension in the scenario "
#             "(axes 1, 4, or 7 are usually best here)."
#         )
#     elif user_turns == 2:
#         p += (
#             "TURN 2 — PIVOT:\n"
#             "Move to a DIFFERENT axis from the list — one they haven't touched yet. "
#             "A good pivot opens up a new dimension of their experience: for example, "
#             "how their language background, class position, or gender shapes how they read this."
#         )
#     elif user_turns == 3:
#         p += (
#             "TURN 3 — DEEPENING:\n"
#             "Ask for a concrete personal memory or lived example that explains "
#             f"WHY they feel the way they described. Reference something specific they said: "
#             f"\"{last_user[:180]}\". Push past opinion into experience."
#         )
#     elif user_turns >= 4:
#         p += (
#             "TURN 4 — CLOSING:\n"
#             "Ask one final question inviting them to reflect on how their identity, background, "
#             "values, or beliefs — broadly understood — shaped the way they read this scenario. "
#             "This should feel like a natural, gentle closing of the conversation."
#         )

#     if user_turns == 4:
#         p += (
#             "\n\nSTOPPING CONDITION: If the participant has shared enough personal detail across "
#             "the conversation to support a coherent 4–5 sentence narrative, "
#             "end your response with the exact text: READY_TO_BUILD"
#         )
#     elif user_turns >= 5:
#         p += (
#             "\n\nCRITICAL OVERRIDE — TURN 5 IS THE ABSOLUTE FINAL TURN. "
#             "You are forbidden from asking another question. "
#             "Thank the participant warmly in one sentence. "
#             "You MUST append the exact text: READY_TO_BUILD"
#         )

#     return p

AXES_CONTEXT = """\
The seven lived-experience axes below are lenses for your questions. \
Do not name them or list them to the participant — use them invisibly to guide what you ask about:

1. Sociocultural & Geographic Context — where they grew up, cultural norms, value systems
2. Linguistic Background & Dialect — native language, multilingualism, code-switching
3. Socioeconomic Status & Labor Dynamics — class, economic precarity, workplace power
4. Race & Ethnicity — racial identity, marginalization, how others perceive them
5. Gender Identity & Sexual Orientation — lived gender/sexuality, how systems categorize them
6. Disability & Neurodivergence — physical, cognitive, or sensory experience
7. Epistemic Proximity — how personally close they are to the people most affected\
"""

def elicitation_sys(scenario: dict, user_turns: int, last_user: str = "") -> str:
    p = (
        "You are a warm, curious qualitative research interviewer doing a lived-experience elicitation. "
        "The participant just read this scenario:\n"
        f"SCENARIO: \"{scenario['vignette']}\"\n\n"
        f"{AXES_CONTEXT}\n\n"
        "CORE RULES — follow these every turn:\n"
        "- Ask exactly ONE question. Never two.\n"
        "- 1–2 sentences maximum. No preamble, no summaries, no affirmations like 'great' or 'I see'.\n"
        "- Never paraphrase what they just said back to them.\n"
        "- If their answer was thin, vague, or a single word, do NOT pivot to a new axis. "
        "  Stay with what they just raised and ask for a concrete moment or example from their life.\n"
        "- Only move to a new axis when the participant has given a substantive answer on the current one.\n"
        "- Ground every question in something specific they actually said — not generic.\n\n"
    )

    if user_turns == 1:
        p += (
            "TURN 1 — OPENING:\n"
            "Pick the most specific or emotionally loaded thing they said. "
            "Ask one question that connects it to their personal experience — "
            "who they are, where they're from, or how close they feel to the situation described. "
            "Axes 1, 4, and 7 are natural starting points."
        )
    elif user_turns == 2:
        # Check if last answer was thin — if so, stay and probe deeper
        last_clean = last_user.strip()
        word_count_hint = len(last_clean.split())
        if word_count_hint <= 6:
            p += (
                "TURN 2 — REPAIR:\n"
                f"The participant's last answer was very short: \"{last_clean}\". "
                "Do not pivot to a new topic. Instead, gently open up what they just said "
                "by asking for a specific memory, moment, or example behind it. "
                "Stay on the same axis — just go deeper."
            )
        else:
            p += (
                "TURN 2 — BROADEN:\n"
                "They've shared one dimension. Now open up a related but different angle "
                "— not an abrupt topic switch, but a natural extension. "
                "For example: if they talked about community, you might ask about "
                "a time they personally felt inside or outside that community. "
                "Choose the next axis based on what feels most alive in what they said."
            )
    elif user_turns == 3:
        last_clean = last_user.strip()
        word_count_hint = len(last_clean.split())
        if word_count_hint <= 6:
            p += (
                "TURN 3 — REPAIR:\n"
                f"The participant's last answer was very short: \"{last_clean}\". "
                "Do not move on. Ask them to ground what they said in a specific moment "
                "or lived example — something that actually happened to them or someone they know."
            )
        else:
            p += (
                "TURN 3 — DEEPEN:\n"
                f"They said: \"{last_clean[:200]}\". "
                "Ask what made that personally significant — not just what happened, "
                "but why it stayed with them, or what it revealed about how they see themselves "
                "or their place in situations like the one in the scenario."
            )
    elif user_turns >= 4:
        p += (
            "TURN 4 — CLOSE:\n"
            "Ask one final, open question that invites them to reflect on how their identity, "
            "values, background, or beliefs — in whatever way feels right to them — "
            "shaped the way they read this scenario. Keep it gentle and open-ended."
        )

    if user_turns == 4:
        p += (
            "\n\nSTOPPING CONDITION: If across the conversation the participant has shared "
            "enough personal detail to support a coherent 4–5 sentence narrative, "
            "end your response with the exact text: READY_TO_BUILD"
        )
    elif user_turns >= 6:
        p += (
            "\n\nFINAL TURN — CRITICAL OVERRIDE: You must not ask another question. "
            "Thank the participant in one warm sentence. "
            "You MUST end your message with the exact text: READY_TO_BUILD"
        )

    return p


SYNTHESIS_SYS = """\
Write a first-person narrative (minimum 120 words) that faithfully captures what this participant shared.

Rules:
- Use 'I' throughout.
- Preserve the participant's own specific words and phrases wherever possible — do not sanitise their voice.
- Do NOT add feelings, interpretations, or experiences they did not express.
- Cover at least three distinct dimensions of lived experience (e.g. cultural background, personal memory, \
emotional response, sense of identity or belonging, relationship to the community depicted).
- The narrative should read as a coherent, flowing piece of personal reflection — not a bullet list or summary.
- Friendly, natural, warm tone — not clinical or academic.
- Do not include a title or preamble. Output ONLY the narrative text.\
"""

# ─── Styles ───────────────────────────────────────────────────────────────────

def inject_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,500;1,400&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background: #000000 !important; color: #F0F0F0 !important; }
    h1, h2 { font-family: 'Lora', serif; font-weight: 400; letter-spacing: -0.01em; }
    .stApp { background: #000000; }
    .stTextInput input, .stTextArea textarea, [data-testid="stChatInput"] {
        background-color: #222222 !important;
        border: 1px solid #444444 !important;
        color: #F0F0F0 !important;
        border-radius: 8px !important;
    }
    .vignette-card {
        background: #EEEAE0; border-left: 4px solid #8B6F47;
        padding: 1.2rem 1.5rem; border-radius: 3px; margin: 1rem 0 1.4rem 0;
        font-size: 0.96rem; line-height: 1.85; color: #2A2A2A; font-style: italic;
    }
    .tweet-card {
        background: #111111;
        border: 1px solid #2A2A2A;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        font-size: 1rem;
        line-height: 1.7;
        color: #EAEAEA;
        box-shadow: 0 2px 8px rgba(0,0,0,0.6);
        margin-bottom: 1.2rem;
    }
    .meta-pill {
        display: inline-block;
        background: #222;
        border-radius: 12px;
        font-size: 0.7rem;
        padding: 3px 10px;
        color: #B5B5B5;
        margin-right: 6px;
    }
    .prog-bg { background: #2A2A2A; border-radius: 20px; height: 5px; margin: 0.4rem 0 1.4rem 0; }
    .prog-fill { background: #8B6F47; height: 5px; border-radius: 20px; }
    .step-label { font-size: 0.71rem; color: #999; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.2rem; }
    div[data-testid="stChatMessage"] { background: transparent !important; }
    .word-counter-ok { color: #6FCF97; font-size: 0.8rem; margin-top: 4px; }
    .word-counter-low { color: #EB5757; font-size: 0.8rem; margin-top: 4px; }
    .chat-replay {
        background: #0D0D0D;
        border: 1px solid #2A2A2A;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.2rem;
        font-size: 0.88rem;
        line-height: 1.75;
        color: #C8C8C8;
    }
    .chat-replay .label {
        color: #8B6F47; font-weight: 600;
        font-size: 0.75rem; letter-spacing: 0.07em;
        text-transform: uppercase; display: block;
        margin-bottom: 0.6rem;
    }
    .chat-replay .turn { margin-bottom: 0.9rem; }
    .chat-replay .q { color: #9E9E9E; margin-bottom: 0.2rem; font-style: italic; }
    .chat-replay .a { color: #E0E0E0; padding-left: 0.8rem; border-left: 2px solid #333; }
    </style>
    """, unsafe_allow_html=True)

def prog(step, total):
    pct = int(step / total * 100)
    st.markdown(
        f"<div class='prog-bg'><div class='prog-fill' style='width:{pct}%'></div></div>",
        unsafe_allow_html=True
    )

def render_word_counter(text: str, minimum: int):
    words = len(text.strip().split()) if text.strip() else 0
    if words >= minimum:
        st.markdown(f"<div class='word-counter-ok'>✓ {words} words</div>", unsafe_allow_html=True)
    else:
        remaining = minimum - words
        st.markdown(
            f"<div class='word-counter-low'>{words} / {minimum} words — {remaining} more to go</div>",
            unsafe_allow_html=True
        )

def render_chat_replay(elicitation: list):
    """Renders a compact Q&A transcript of the elicitation exchange."""
    pairs = []
    for i, m in enumerate(elicitation):
        if m["role"] == "assistant":
            answer = elicitation[i + 1]["content"] if i + 1 < len(elicitation) and elicitation[i + 1]["role"] == "user" else None
            if answer:
                pairs.append((m["content"], answer))

    html = "<div class='chat-replay'><span class='label'>Your conversation</span>"
    for q, a in pairs:
        html += (
            f"<div class='turn'>"
            f"<div class='q'>{q}</div>"
            f"<div class='a'>{a}</div>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def scroll_to_bottom():
    """JS to scroll the last chat message into view."""
    st.components.v1.html(
        """
        <script>
        (function() {
            var msgs = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            if (msgs.length > 0) {
                msgs[msgs.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
            }
        })();
        </script>
        """,
        height=0,
    )

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Hate Speech & Belonging — Pilot Study",
        layout="centered",
        page_icon="🗣"
    )
    inject_styles()

    # ── SIGN-IN ───────────────────────────────────────────────────────────────
    if "participant_name" not in st.session_state:
        st.markdown("<div class='step-label'>Pilot Study · Version A</div>", unsafe_allow_html=True)
        st.title("Hate Speech & Belonging")
        st.markdown("*A positionality-aware annotation study*")
        st.markdown("---")
        st.markdown(
            "This study takes about **20–25 minutes**. You'll first share a bit about your own "
            "perspective and experiences, then read and annotate a small set of social media posts. "
            "There are no right or wrong answers."
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
        labels = {
            "disclosure": "1 — Background",
            "elicitation_chat": "2 — Your experience",
            "synthesis": "3 — Your narrative",
            "annotation": "4 — Annotations",
            "complete": "✓ Done"
        }
        st.caption(labels.get(stage, stage))
        st.markdown("---")
        st.caption("Data is held in memory and saved securely at the end.")

    # ── STAGE 1: DISCLOSURE ───────────────────────────────────────────────────
    if stage == "disclosure":
        st.markdown("<div class='step-label'>Step 1 of 4</div>", unsafe_allow_html=True)
        st.title("A bit about you")
        prog(1, 4)
        st.write(
            "Before we begin, we'd like to understand your connection to this topic. "
            "This context shapes how we interpret your annotations — it won't affect your participation."
        )

        conn = st.selectbox(
            "How would you describe your connection to topics of immigration, religion, or identity and belonging?",
            [
                "— please select —",
                "I have direct personal experience (as an immigrant, refugee, religious minority, or member of a marginalised group)",
                "I'm a caregiver, partner, or close community member of someone with this experience",
                "I work or study in this area professionally or academically",
                "I'm an interested observer — no direct personal connection",
            ],
        )
        duration = st.text_input(
            "How long has this been part of your life or work? (e.g., 'my whole life', '3 years')"
        )
        disclosure = st.text_area(
            "Briefly describe how this topic relates to your life. "
            "This stays confidential and helps us contextualise your responses.",
            height=100,
            placeholder="e.g. My parents immigrated to Canada in the 1990s...",
        )

        if st.button("Continue →", type="primary"):
            if conn == "— please select —":
                st.warning("Please select an option before continuing.")
            elif not duration.strip():
                st.warning("Please fill in how long this topic has been part of your life.")
            else:
                data["disclosure"] = {"connection_type": conn, "duration": duration, "text": disclosure}
                data["workflow_stage"] = "elicitation_chat"
                st.rerun()

    # ── STAGE 2: ELICITATION ──────────────────────────────────────────────────
    elif stage == "elicitation_chat":
        st.markdown("<div class='step-label'>Step 2 of 4 — Your Experience</div>", unsafe_allow_html=True)
        st.title("Your perspective")
        prog(2, 4)

        # Instruction BEFORE the passage
        st.write(
            "Read the passage below, then respond to the interviewer's questions in the chat. "
            "There are no right answers — we're interested in your genuine reactions and "
            "personal experiences. Aim for 2–3 sentences per reply."
        )
        st.markdown(f"<div class='vignette-card'>{scenario['vignette']}</div>", unsafe_allow_html=True)

        # Render chat history
        for msg in data["elicitation"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Seed opening question on first load
        if not data["elicitation"]:
            data["elicitation"].append({"role": "assistant", "content": scenario["opening_q"]})
            st.rerun()

        user_turns = sum(1 for m in data["elicitation"] if m["role"] == "user")
        max_turns = 5

        if user_turns < max_turns:
            if user_input := st.chat_input("Your response…"):
                data["elicitation"].append({"role": "user", "content": user_input})
                sys_p = elicitation_sys(scenario, user_turns + 1, user_input)
                response = call_qwen(sys_p, data["elicitation"], max_tokens=180)

                if "READY_TO_BUILD" in response:
                    clean = response.replace("READY_TO_BUILD", "").strip()
                    if clean:
                        data["elicitation"].append({"role": "assistant", "content": clean})
                    data["workflow_stage"] = "synthesis"
                else:
                    data["elicitation"].append({"role": "assistant", "content": response})
                st.rerun()
        else:
            # Auto-advance after hard cap
            if data["workflow_stage"] == "elicitation_chat":
                data["workflow_stage"] = "synthesis"
                st.rerun()

        scroll_to_bottom()

    # ── STAGE 3: SYNTHESIS ────────────────────────────────────────────────────
    elif stage == "synthesis":
        st.markdown("<div class='step-label'>Step 3 of 4 — Your Narrative</div>", unsafe_allow_html=True)
        st.title("Your story, in your words")
        prog(3, 4)
        st.write(
            "Based on what you shared, we've drafted a short narrative in your voice. "
            "Edit anything that doesn't feel accurate or complete, then accept it to continue."
        )

        # Q&A replay shown above the narrative
        render_chat_replay(data["elicitation"])

        if not data.get("micronarrative"):
            with st.spinner("Drafting your narrative…"):
                fragments = "\n".join(
                    m["content"] for m in data["elicitation"] if m["role"] == "user"
                )
                data["micronarrative"] = call_qwen(
                    SYNTHESIS_SYS,
                    [{"role": "user", "content": fragments}],
                    max_tokens=400
                )

        edited = st.text_area(
            "Your narrative (edit freely):",
            value=data["micronarrative"],
            height=230
        )
        nar_words = len(edited.strip().split()) if edited.strip() else 0
        render_word_counter(edited, 80)

        c1, c2 = st.columns([1, 2])
        with c1:
            if st.button("↺ Regenerate"):
                data["micronarrative"] = ""
                st.rerun()
        with c2:
            if st.button("Accept & start annotations →", type="primary"):
                if nar_words < 80:
                    st.warning(
                        f"Your narrative is {nar_words} words. "
                        "Please expand it to at least 80 words before continuing."
                    )
                else:
                    data["micronarrative"] = edited
                    data["workflow_stage"] = "annotation"
                    st.rerun()

    # ── STAGE 4: ANNOTATION ───────────────────────────────────────────────────
    elif stage == "annotation":
        datapoints = get_datapoints(data["scenario_id"])
        idx = len(data["annotations"])

        if idx < len(datapoints):
            dp = datapoints[idx]
            st.markdown(
                f"<div class='step-label'>Step 4 of 4 — Post {idx+1} of {len(datapoints)}</div>",
                unsafe_allow_html=True
            )
            st.title("Annotating social media posts")
            prog(idx + 1, len(datapoints))
            st.write(
                "You'll now read and annotate ten social media posts. "
                "There are no right or wrong answers — your interpretation is the data."
            )

            # Q&A + narrative accessible as collapsible above the post
            with st.expander("Your lived experience narrative (tap to view)", expanded=False):
                # render_chat_replay(data["elicitation"])
                # st.markdown("---")
                st.markdown(f"*{data['micronarrative']}*")

            # Instruction BEFORE the passage (improves flow)
            st.markdown("**Read the post below, then answer the questions.**")
            st.markdown(
                f"<div class='tweet-card'>"
                f"<span class='meta-pill'>Post {idx+1} of {len(datapoints)}</span>"
                f"<span class='meta-pill'>{dp['domain']}</span>"
                f"<br><br>{dp['text']}</div>",
                unsafe_allow_html=True,
            )

            with st.form(f"anno_{idx}"):

                # # Q1: Label — no default (index=None)
                # st.markdown("**1. How would you label this post?**")
                # label = st.radio(
                #     "",
                #     options=[
                #         "Hate speech — dehumanises or incites harm toward a person or group",
                #         "Borderline — potentially harmful, but depends on context or intent",
                #         "Not hate speech — critical, blunt, or politically charged but not hateful",
                #     ],
                #     index=None,
                #     label_visibility="collapsed"
                # )
                # Q1: Label — 5-point spectrum, no default
                st.markdown("**1. How would you label this post?**")
                label = st.radio(
                    "",
                    options=[
                        "Clearly harmful — targets a group in a way that dehumanises or incites",
                        "Likely harmful — framing that would harm most readers from the targeted group, even if deniable",
                        "Ambiguous — could be read as harmful or legitimate depending on who is reading and why",
                        "Likely not harmful — critical or provocative, but the harm reading requires a stretch",
                        "Not harmful — policy concern, cultural commentary, or frustration with no hateful subtext",
                    ],
                    index=None,
                    label_visibility="collapsed"
                )

                # st.markdown("---")

                # # Q2: Target — no default
                # st.markdown("**2. If harm is present, who seems most targeted?**")
                # target = st.radio(
                #     "",
                #     options=[
                #         "A specific individual",
                #         "Immigrants, refugees, or a racialised group",
                #         "A religious community",
                #         "A gender or LGBTQ+ community",
                #         "No specific target — it's about a policy or idea",
                #         "Unclear",
                #     ],
                #     index=None,
                #     label_visibility="collapsed"
                # )

                st.markdown("---")

                # Q2: Target — selectbox, no default
                st.markdown("**2. Who, if anyone, does this post seem directed at?**")
                target = st.selectbox(
                    "",
                    options=[
                        "— select —",
                        "No one — I don't read this as targeting anyone",
                        "A specific individual",
                        "Immigrants, refugees, or a racialised group",
                        "A religious community",
                        "A gender or LGBTQ+ community",
                        "Multiple overlapping groups",
                        "Unclear",
                    ],
                    index=0,
                    label_visibility="collapsed"
                )

                st.markdown("---")

                # Q3: Rationale with live word counter
                st.markdown(
                    f"**3. Your rationale** *(minimum {MIN_RATIONALE_WORDS} words)*"
                )
                st.caption(
                    "What specific language, framing, or context shaped your judgement? "
                    "Did any aspect of your identity, beliefs, values, or lived experience "
                    "affect how you read this post?"
                )
                # rationale = st.text_area(
                #     "",
                #     height=170,
                #     label_visibility="collapsed",
                #     placeholder=(
                #         "e.g. The phrase '...' stood out to me because... "
                #         "My background made me read this differently in that... "
                #         "What makes this feel harmful / ambiguous / acceptable is..."
                #     )
                # )

                rationale = st.text_area(
                    "",
                    height=170,
                    label_visibility="collapsed",
                )
                rationale_words = len(rationale.strip().split()) if rationale.strip() else 0
                render_word_counter(rationale, MIN_RATIONALE_WORDS)

                st.markdown("---")

                # Q4: Positionality salience — no default, improved label
                st.markdown(
                    "**4. How relevant did your identity or personal experience feel "
                    "to how much this post resonated with you?**"
                )
                st.caption("1 = not at all relevant to my identity or experience · 5 = very much so")
                salience = st.slider("", 1, 5, value=None, label_visibility="collapsed")

                submitted = st.form_submit_button("Submit & next →", type="primary")

                if submitted:
                    errors = []
                    if label is None:
                        errors.append("Please select a label for question 1.")
                    if target == "— select —":
                        errors.append("Please select a target for question 2.")
                    if rationale_words < MIN_RATIONALE_WORDS:
                        errors.append(
                            f"Your rationale is {rationale_words} words — please expand to at least "
                            f"{MIN_RATIONALE_WORDS} words. The detail you provide is the most valuable "
                            "part of the study."
                        )
                    if salience is None:
                        errors.append("Please move the slider in question 4.")

                    if errors:
                        for e in errors:
                            st.warning(e)
                    else:
                        data["annotations"].append({
                            "datapoint_id": dp["id"],
                            "domain": dp["domain"],
                            "tweet_text": dp["text"],
                            "participant_label": label,
                            "participant_target": target,
                            "rationale": rationale,
                            "positionality_salience": salience,
                            "timestamp": datetime.utcnow().isoformat(),
                        })
                        st.rerun()

        else:
            # All annotations done — save to Google Sheets
            with st.spinner("Saving your responses securely…"):
                max_retries = 3
                for attempt in range(max_retries):
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
                        existing_data = conn.read(
                            spreadsheet=SHEET_URL,
                            usecols=list(new_row.keys()),
                            ttl=0
                        )
                        updated_data = pd.concat(
                            [existing_data, pd.DataFrame([new_row])],
                            ignore_index=True
                        )
                        conn.update(spreadsheet=SHEET_URL, data=updated_data)
                        data["workflow_stage"] = "complete"
                        st.rerun()
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            time.sleep(2)
                        else:
                            st.error(
                                f"Failed to save after {max_retries} attempts. "
                                "Please leave this window open and contact the researcher. "
                                f"Error: {e}"
                            )

    # ── COMPLETE ──────────────────────────────────────────────────────────────
    elif stage == "complete":
        st.balloons()
        st.title("Thank you.")
        st.markdown(
            f"Your annotations and narrative are saved, **{data['name']}**. "
            "The perspectives you bring — including your background and lived experience — "
            "are what makes this kind of research meaningful."
        )
        st.caption("Pilot study by Sheza Munir · Data stored securely · You may close this window.")


if __name__ == "__main__":
    main()
