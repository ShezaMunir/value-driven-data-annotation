# HatEval Positionality Pilot Study

**Researcher:** Sheza Munir · **Target venue:** ACL  
**Dataset:** HatEval (Basile et al. 2019, SemEval-2019 Task 5, English)

---

## Files

| File | Description |
|------|-------------|
| `pilot_study.py` | **Version A**: Elicitation → Annotation (~8 participants) |
| `pilot_study_v2.py` | **Version B**: Annotation → Elicitation (~7 participants) |
| `requirements.txt` | Python dependencies |
| `vercel.json` | Vercel deployment config |

---

## Datapoints

All 15 tweets come from the HatEval English training split, sampled via multi-classifier
disagreement + entropy scoring (combined_score = 0.5×norm_variance + 0.5×norm_entropy).
Grouped into 3 thematic scenarios, 5 tweets each:

| Scenario | Theme | HS=1 count | HS=0 count |
|----------|-------|-----------|-----------|
| A | Borders, Safety & Who Belongs | 5 | 0 |
| B | Crisis, Suffering & Who We Choose to See | 0 | 5 |
| C | Culture, Religion & the Politics of Belonging | 1 | 4 |

Scenario B is the most interesting annotation task: all 5 tweets are HS=0 (factual/advocacy)
yet cardiffnlp_hate_prob ranges 0.80–0.96. This classifier-human disagreement is the core
ambiguity the study is designed to unpack.

---

## Scenarios & Vignettes

Vignettes are standalone (no tweets shown during elicitation). Purpose: activate relevant
autobiographical memory and lived-experience framing before annotation. Design follows
Barter & Renold (1999): 80–150 words, theory-grounded, max 3 to avoid fatigue.

---

## Running the study

### Local / development
```bash
pip install -r requirements.txt
HF_TOKEN=hf_xxxx streamlit run pilot_study.py      # Version A
HF_TOKEN=hf_xxxx streamlit run pilot_study_v2.py   # Version B
```

### Streamlit Community Cloud (recommended for sharing)
1. Push repo to GitHub
2. share.streamlit.io → New app → select `pilot_study.py`
3. Settings → Secrets → add: `HF_TOKEN = "hf_xxxx"`
4. Repeat for `pilot_study_v2.py` as a second app

### Vercel
```bash
npm i -g vercel
vercel env add HF_TOKEN   # paste token when prompted
vercel --prod
```
> Note: Streamlit Community Cloud is more reliable for Streamlit apps on Vercel
> due to serverless cold-start and streaming constraints.

---

## Participant assignment

- 15 participants total, round-robin scenario assignment (5 per scenario)
- ~8 → Version A (pilot_study.py); ~7 → Version B (pilot_study_v2.py)
- Sign-in by name: returning participants automatically resume their session
- Data stored in `./pilot_data/<name>/session.json` (Version A) and `./pilot_data_v2/` (Version B)

---

## Data collected per participant

```json
{
  "name": "...",
  "version": "A" | "B",
  "scenario_id": "A" | "B" | "C",
  "disclosure": { "connection_type": "...", "duration": "...", "text": "..." },
  "elicitation": [ {"role": "user"|"assistant", "content": "..."} ],
  "micronarrative": "...",
  "annotations": [
    {
      "tweet_id": 4375,
      "tweet_text": "...",
      "hateval_hs": 1,
      "hateval_ag": 1,
      "hateval_tr": 0,
      "combined_score": 0.5457,
      "participant_label": "Hate speech | Borderline | Not hate speech",
      "participant_target": "...",
      "rationale": "...",
      "positionality_salience": 1-5,
      "timestamp": "..."
    }
  ]
}
```

---

## ACL design notes

### Core contribution
Positionality-aware annotation with an experimentally counterbalanced elicitation-order
condition (Versions A vs B). No prior work in NLP annotation has tested whether
lived-experience elicitation *before* vs *after* annotation affects label quality.

### Predicted findings
- **Version A** → richer rationale, more borderline labels, higher salience scores
- **Version B** → colder baseline labels (less narrative priming), useful as control

### Analysis plan
- Rationale depth: mean word count + qualitative coding by lived-experience axis
- Label distribution: borderline rate, agreement with HatEval majority label
- Salience × label: do high-salience participants disagree more with HS=0/hate_prob>0.8 cases?
- Micro-narrative content: what axes (identity, fairness, belonging, relationships) appear?

### Theoretical grounding
| Reference | Role in study |
|-----------|--------------|
| Basile et al. (2019) SemEval Task 5 | Dataset source |
| Rocchio et al. (2022) Eliciting Values of Patients with MCC | Domain-anchored elicitation prompts |
| Barter & Renold (1999) Using Vignettes in Educational Research | Vignette design |
| Smythe et al. (2008) Interpretive Phenomenological Inquiry | Elicitation order rationale; hermeneutic circle; positionality salience framing |
| Willig (2013) Elicitation Interview Technique | Turn structure (initial → harm cue → lived exp → deepen) |
| Boyle & Butcher (2024) Drawn from Life | Micronarrative synthesis (fidelity-first) |
