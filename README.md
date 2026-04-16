# HatEval Positionality Pilot Study


Dataset: HatEval (Basile et al. 2019, SemEval-2019 Task 5)

---

## Files

| File | Description |
|------|-------------|
| `pilot_study.py` | **Version A**: Elicitation → Annotation (recommended, ~8 participants) |
| `pilot_study_v2.py` | **Version B**: Annotation → Elicitation (counterbalance, ~7 participants) |
| `requirements.txt` | Python dependencies |
| `vercel.json` | Vercel deployment config |

---

## Deployment

### Local (recommended for testing)
```bash
pip install -r requirements.txt
HF_TOKEN=your_token_here streamlit run pilot_study.py
```

### Vercel
> Note: Vercel's serverless functions have a 50MB limit and 10s timeout. Streamlit apps
> work best on Vercel via the community `@streamlit/vercel-adapter` or by wrapping
> in a FastAPI app. The simpler production path is **Streamlit Community Cloud** (free):

**Streamlit Community Cloud (easier):**
1. Push this repo to GitHub
2. Go to share.streamlit.io → New app → select `pilot_study.py`
3. Add secret: `HF_TOKEN = "your_token_here"` in the Secrets panel
4. Deploy — you'll get a shareable URL

**Vercel (if required):**
1. `npm i -g vercel`
2. `vercel env add HF_TOKEN` → paste your HF token
3. `vercel --prod`

---

## Model

**mistralai/Mistral-7B-Instruct-v0.3** via HF Inference API  
- Instruction-tuned, stable on HF free tier
- Research-permissive license (Apache 2.0)
- Endpoint: `https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3`

Get a free HF token at: https://huggingface.co/settings/tokens

---

## Participant Assignment

- 15 participants total
- 3 scenarios × 5 participants each (round-robin by sign-up order)
- ~8 participants → Version A (pilot_study.py)
- ~7 participants → Version B (pilot_study_v2.py)

Scenarios:
- **A**: Immigration & Belonging (HatEval IDs 1, 2, 4, 6, 7)
- **B**: Gender & Public Voice (IDs 3, 5, 8, 10, 13)
- **C**: Identity, Integration & Othering (IDs 9, 11, 12, 14, 15)

---

## Data Collection

Each participant's data is saved as `pilot_data/<name>/session.json`.  
Key fields collected:
- `disclosure`: background + connection type
- `elicitation`: full AI-guided chat turns
- `micronarrative`: final accepted first-person narrative
- `annotations`: per-tweet labels, rationale, target, positionality salience

---

## ACL Design Notes

### Key methodological contribution
No prior positionality-aware NLP annotation work has experimentally compared annotation 
quality across elicitation-order conditions. Versions A and B enable this comparison.

### Predicted findings
- Version A → richer rationale depth, more nuanced borderline labels
- Version B → "colder" labels, potentially less vignette-primed

### Theoretical grounding (encoded in code comments)
1. Rocchio et al. (2022) — dimension-anchored elicitation prompts
2. Barter & Renold (1999) — vignette construction (50–200 words, max 3, theory-grounded)
3. Smythe et al. (2008) — interpretive phenomenology, hermeneutic circle
4. Willig (2013) — elicitation interview structure (initial → harm cue → lived exp → deepen)
5. Boyle & Butcher (2024) — fidelity-first micronarrative synthesis
6. Basile et al. (2019) — HatEval dataset (SemEval-2019 Task 5)

### On positionality salience
Reframed from "did your experience bias you?" to "how relevant did your background feel?" —
this preserves the reflexivity measure without activating objectivity anxiety (Smythe 2008).
