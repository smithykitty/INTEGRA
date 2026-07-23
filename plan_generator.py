"""
plan_generator.py — RAG-based care plan generation via the Anthropic API.

Note: you must have a key, and may need to pay for usage.
Usage:
    python src/plan_generator.py  (runs a quick smoke test with a sample patient)
"""

import json
import re
import uuid
import anthropic
from config import ANTHROPIC_API_KEY, LLM_MODEL, TRADITION_COLS
from knowledge_base import retrieve


def build_rag_prompt(patient: dict, active_modalities: list) -> str:
    query = " ".join(filter(None, [
        patient.get("chief_complaint", ""),
        patient.get("tcm_complaints", ""),
        patient.get("wellness_goals", ""),
        patient.get("vikruti", ""),
        patient.get("tongue", ""),
    ]))

    context_sections = []
    for mod in active_modalities:
        ctx = retrieve(query, mod, n_results=3)
        if ctx:
            context_sections.append(f"=== {mod} knowledge ===\n{ctx}")
    knowledge_block = "\n\n".join(context_sections)

    return f"""You are the AI reasoning engine of a five-tradition integrative medicine \
HITL-RL system. Use the retrieved knowledge below to generate a specific, grounded care plan.

RETRIEVED KNOWLEDGE:
{knowledge_block}

PATIENT DATA:
{json.dumps(patient, indent=2)}

Generate a JSON array of 6-9 care plan items. Each item must have:
- system: one of {active_modalities + ["Integrated"]}
- category: one of: Herbal formula, Acupuncture/bodywork, Diet and nutrition, \
Targeted supplementation, Lifestyle medicine, Mind-body practice, \
Detox and elimination, Functional testing, Adaptogenic protocol, Gut restoration
- title: max 8 words, no special punctuation
- detail: 2-3 sentences. Name specific herbs with Latin names, nutrients with doses, \
acupoints, foods. Use only plain ASCII punctuation (hyphens not dashes).
- cross_system_synergy: plain text string if 2 or more traditions align, else null
- priority: High, Medium, or Low
- source_tradition: classical text or principle cited
- herb_drug_flag: interaction warning if pharmaceuticals listed, else null
- rl_confidence: integer 0-100
- evidence_grade: one of: A - RCT evidence, B - observational/traditional clinical, \
C - traditional use only

IMPORTANT: Return ONLY a valid JSON array. Use plain ASCII punctuation only. \
No em dashes, no curly quotes, no special characters. No markdown, no preamble."""


def generate_plan(patient: dict, active_modalities: list) -> tuple[str, list]:
    """Call the Claude API and return (episode_id, plan_items)."""
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-..."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = build_rag_prompt(patient, active_modalities)

    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = message.content[0].text

    # Strip markdown fences
    text = text.strip()
    text = re.sub(r'^```[a-z]*\n?', '', text)
    text = re.sub(r'```\s*$', '', text)
    text = text.strip()

    # Replace special unicode punctuation with ASCII equivalents
    replacements = {
        '\u2014': '-',   # em dash
        '\u2013': '-',   # en dash
        '\u2012': '-',   # figure dash
        '\u2011': '-',   # non-breaking hyphen
        '\u2018': "'",   # left single quote
        '\u2019': "'",   # right single quote
        '\u201a': "'",   # single low quote
        '\u201c': '"',   # left double quote
        '\u201d': '"',   # right double quote
        '\u201e': '"',   # double low quote
        '\u2026': '...', # ellipsis
        '\u00b7': '-',   # middle dot
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Extract just the JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        text = match.group()

    # Remove any remaining non-printable control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    plan_items = json.loads(text)
    episode_id = str(uuid.uuid4())[:8]
    return episode_id, plan_items


if __name__ == "__main__":
    sample_patient = {
        "age": 45, "sex": "Female", "chief_complaint": "chronic fatigue and poor sleep",
        "wellness_goals": "improve energy and reduce anxiety",
        "tongue": "Peeled / no coat", "pulse": "Thin / thready",
        "dosha": "Vata-Pitta", "vikruti": "Vata excess", "agni": "Vishama (irregular)",
        "stress_level": 8, "diet_pattern": "Vegetarian",
        "suspected_deficiencies": "Vitamin D, Magnesium", "pharmaceuticals": "",
        "active_modalities": ["TCM", "Ayurveda", "Functional Nutrition"],
    }
    print("Generating care plan for sample patient...")
    ep_id, plan = generate_plan(sample_patient, sample_patient["active_modalities"])
    print(f"\nEpisode ID: {ep_id}")
    for i, item in enumerate(plan):
        print(f"\n[{i}] [{item['system']}] {item['title']} ({item['priority']})")
        print(f"    {item['detail'][:100]}...")
