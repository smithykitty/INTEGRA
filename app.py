"""
app.py — Unified Gradio application for the Integrative Medicine HITL-RL system.

Tuned for UVA Rivanna / Open OnDemand JupyterLab (Python 3.13, share=True).

Run from a JupyterLab terminal:
    cd ~/integrative_rl
    python src/app.py

A public gradio.live link will be printed — click it to open the app.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import EPISODES_DIR
from episode_store import (
    init_db, compute_item_reward, save_draft_episode,
    load_draft_episode, save_reviewed_episode, load_all_episodes, episode_count,
)
from plan_generator import generate_plan

init_db()

MODALITY_CHOICES = [
    "TCM", "Ayurveda", "Naturopathy", "Clinical Herbalism", "Functional Nutrition"
]

_state = {"episode_id": None, "plan": None, "patient": None, "reviews": {}}


# ── Tab 1: Patient intake ────────────────────────────────────────────────────

def run_intake(
    age, sex, weight, height,
    chief_complaint, wellness_goals,
    tongue, pulse, emotion, sleep_pattern, tcm_complaints,
    dosha, agni, vikruti, dhatu, ayur_lifestyle,
    bowel, stress, water, exercise, sensitivities,
    herb_trad, current_herbs, herb_evidence, pharmaceuticals, organ_focus,
    diet_pattern, meal_timing, deficiencies, supps,
    gut_symptoms, inflam_markers, metgoal, sleep_quality,
    active_mods,
):
    if not active_mods:
        return "⚠ Select at least one modality.", "", gr.update(visible=False)

    patient = dict(
        age=age, sex=sex, weight_kg=weight, height_cm=height,
        bmi=round(weight / ((height / 100) ** 2), 1) if height and weight else None,
        chief_complaint=chief_complaint, wellness_goals=wellness_goals,
        tongue=tongue, pulse=pulse, emotion=emotion,
        sleep_pattern=sleep_pattern, tcm_complaints=tcm_complaints,
        dosha=dosha, agni=agni, vikruti=vikruti, dhatu=dhatu, ayur_lifestyle=ayur_lifestyle,
        bowel=bowel, stress_level=stress, water_l=water, exercise=exercise,
        sensitivities=sensitivities,
        herb_tradition=herb_trad, current_herbs=current_herbs,
        herb_evidence=herb_evidence, pharmaceuticals=pharmaceuticals,
        organ_focus=organ_focus,
        diet_pattern=diet_pattern, meal_timing=meal_timing,
        suspected_deficiencies=deficiencies, current_supps=supps,
        gut_symptoms=gut_symptoms, inflam_markers=inflam_markers,
        metgoal=metgoal, sleep_quality=sleep_quality,
        active_modalities=active_mods,
    )

    try:
        ep_id, plan = generate_plan(patient, active_mods)
    except Exception as e:
        return f"❌ Error: {e}", "", gr.update(visible=False)

    _state.update({"episode_id": ep_id, "plan": plan, "patient": patient, "reviews": {}})
    save_draft_episode(ep_id, patient, plan, EPISODES_DIR)

    preview = f"✅ Episode: {ep_id}   ({len(plan)} items)\n\n"
    for i, item in enumerate(plan):
        flag    = "  ⚠ HERB-DRUG" if item.get("herb_drug_flag") else ""
        synergy = "  ✦ cross-system" if item.get("cross_system_synergy") else ""
        preview += f"[{i}] [{item['system']}] {item['title']} ({item['priority']}){flag}{synergy}\n"
        preview += f"     {item['detail'][:120]}...\n\n"

    return preview, ep_id, gr.update(visible=True)


# ── Tab 2: Clinician review ──────────────────────────────────────────────────

def load_episode_for_review(episode_id):
    ep_id = episode_id.strip()
    try:
        ep = load_draft_episode(ep_id, EPISODES_DIR)
    except FileNotFoundError:
        if _state["episode_id"] == ep_id and _state["plan"]:
            ep = {"episode_id": ep_id, "patient": _state["patient"], "plan": _state["plan"]}
        else:
            return f"Episode {ep_id} not found.", gr.update(visible=False)

    _state.update({"episode_id": ep_id, "plan": ep["plan"],
                   "patient": ep["patient"], "reviews": {}})
    plan = ep["plan"]
    p    = ep["patient"]

    summary  = f"Episode {ep_id}  —  {len(plan)} items\n"
    summary += f"Patient: {p.get('age','?')}yo {p.get('sex','?')}, Chief: {p.get('chief_complaint','N/A')}\n\n"
    for i, item in enumerate(plan):
        flag    = " ⚠" if item.get("herb_drug_flag") else ""
        synergy = " ✦" if item.get("cross_system_synergy") else ""
        summary += f"[{i}] [{item['system']}] {item['title']} ({item['priority']}){flag}{synergy}\n"

    return summary, gr.update(visible=True)


def review_item(item_index, action, note, reviewer_id):
    plan = _state.get("plan")
    if not plan:
        return "No episode loaded."
    idx    = int(item_index)
    item   = plan[idx]
    reward = compute_item_reward(action, item, note)
    _state["reviews"][idx] = {"action": action, "note": note, "reward": reward}
    return (f"Item {idx} → {action.upper()}  (reward: {reward:+.3f})\n"
            f"{len(_state['reviews'])}/{len(plan)} items reviewed.")


def submit_full_review(reviewer_id):
    plan    = _state.get("plan")
    reviews = _state.get("reviews", {})
    if not plan:
        return "No episode loaded."
    pending = [i for i in range(len(plan)) if i not in reviews]
    if pending:
        return f"Please review all items. Pending: {pending}"
    total = save_reviewed_episode(
        _state["episode_id"], _state["patient"], plan, reviews, reviewer_id
    )
    n = episode_count()
    return (f"✅ Episode {_state['episode_id']} saved.\n"
            f"Episode reward: {total:+.3f}\n"
            f"Total episodes: {n}\n"
            f"{'🚀 Run: python src/train.py' if n >= 10 else f'Need {10-n} more episodes before RL training.'}")


# ── Tab 3: Analytics ─────────────────────────────────────────────────────────

def refresh_dashboard():
    items, eps = load_all_episodes()
    n_eps  = len(eps)
    n_items = len(items)
    n_app  = int((items["action"] == "approve").sum()) if not items.empty else 0
    mean_r = eps["reward_total"].mean() if not eps.empty else 0.0

    stats = (f"Episodes: {n_eps}  |  Item reviews: {n_items}  |  "
             f"Approved: {n_app} ({100*n_app/max(n_items,1):.0f}%)  |  "
             f"Mean reward: {mean_r:.3f}  |  "
             f"RL ready: {'✅' if n_eps >= 10 else f'❌ need {10-n_eps} more'}")

    def reward_curve(eps):
        if eps.empty: return go.Figure().update_layout(title="No data yet")
        e = eps.copy(); e["n"] = range(1, len(e)+1)
        e["roll"] = e["reward_total"].rolling(5, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=e["n"], y=e["reward_total"], mode="markers",
                                 name="Episode", marker_color="#9FE1CB"))
        fig.add_trace(go.Scatter(x=e["n"], y=e["roll"], mode="lines",
                                 name="5-ep avg", line_color="#0F6E56"))
        fig.update_layout(title="Episode reward over time",
                          xaxis_title="Episode", yaxis_title="Reward")
        return fig

    def approval_chart(items):
        if items.empty: return go.Figure().update_layout(title="No data yet")
        df = items.groupby(["system","action"]).size().reset_index(name="count")
        return px.bar(df, x="system", y="count", color="action", barmode="group",
                      color_discrete_map={"approve":"#1D9E75","modify":"#BA7517","reject":"#E24B4A"},
                      title="Actions by tradition")

    def heatmap(items):
        if items.empty: return go.Figure().update_layout(title="No data yet")
        approved = items[items["action"]=="approve"]
        if approved.empty: return go.Figure().update_layout(title="No approvals yet")
        pivot = approved.groupby(["system","category"]).size().unstack(fill_value=0)
        return px.imshow(pivot, color_continuous_scale="Teal",
                         title="Approval heatmap: tradition × category", aspect="auto")

    def evidence_chart(items):
        if items.empty: return go.Figure().update_layout(title="No data yet")
        df = items.groupby(["evidence_grade","action"]).size().reset_index(name="count")
        return px.bar(df, x="evidence_grade", y="count", color="action", barmode="stack",
                      color_discrete_map={"approve":"#1D9E75","modify":"#BA7517","reject":"#E24B4A"},
                      title="Actions by evidence grade")

    return stats, reward_curve(eps), approval_chart(items), heatmap(items), evidence_chart(items)


# ── Build app ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Integrative Medicine HITL-RL") as app:
    gr.Markdown("# 🌿 Integrative Medicine — Clinician-in-the-Loop RL")
    gr.Markdown("TCM · Ayurveda · Naturopathy · Clinical Herbalism · Functional Nutrition")

    with gr.Tabs():

        with gr.Tab("Patient Intake"):
            with gr.Row():
                age    = gr.Number(label="Age", minimum=1, maximum=120)
                sex    = gr.Dropdown(["Male","Female","Other/non-binary"], label="Sex")
                weight = gr.Number(label="Weight (kg)")
                height = gr.Number(label="Height (cm)")
            chief_complaint = gr.Textbox(label="Chief complaint", lines=2)
            wellness_goals  = gr.Textbox(label="Wellness goals", lines=2)

            with gr.Accordion("TCM intake", open=False):
                with gr.Row():
                    tongue  = gr.Dropdown(["None/thin white","Thick white","Yellow","Greasy yellow","Peeled/no coat","Purple/dusky"], label="Tongue")
                    pulse   = gr.Dropdown(["Wiry","Slippery","Thin/thready","Deep & weak","Rapid","Slow & deep","Choppy"], label="Pulse")
                    emotion = gr.Dropdown(["Anger/frustration","Worry/overthinking","Grief/sadness","Fear/anxiety","Excessive joy"], label="Emotion")
                    sleep_p = gr.Dropdown(["Difficulty falling asleep","Waking 1–3AM","Waking 3–5AM","Oversleeping","Restless/vivid dreams"], label="Sleep")
                tcm_complaints = gr.Textbox(label="TCM complaints", lines=2)

            with gr.Accordion("Ayurvedic intake", open=False):
                with gr.Row():
                    dosha   = gr.Dropdown(["Vata","Pitta","Kapha","Vata-Pitta","Pitta-Kapha","Vata-Kapha","Tridoshic"], label="Prakriti")
                    agni    = gr.Dropdown(["Sama (balanced)","Vishama (irregular)","Tikshna (sharp)","Manda (slow)"], label="Agni")
                    vikruti = gr.Dropdown(["Vata excess","Pitta excess","Kapha excess","Ama accumulation"], label="Vikruti")
                    dhatu   = gr.Dropdown(["Rasa","Rakta","Mamsa","Meda","Asthi","Majja","Shukra"], label="Dhatu")
                ayur_lifestyle = gr.Textbox(label="Lifestyle & diet", lines=2)

            with gr.Accordion("Naturopathic intake", open=False):
                with gr.Row():
                    bowel    = gr.Dropdown(["Daily/regular","Every 2–3 days","Loose/frequent","Alternating"], label="Bowel")
                    stress   = gr.Slider(1, 10, step=1, label="Stress", value=5)
                    water    = gr.Number(label="Water (L/day)", value=1.5)
                    exercise = gr.Dropdown(["Daily","3–5×/week","1–2×/week","Rarely/none"], label="Exercise")
                sensitivities = gr.Textbox(label="Sensitivities / allergies", lines=2)

            with gr.Accordion("Clinical Herbalism", open=False):
                with gr.Row():
                    herb_trad     = gr.Dropdown(["Western/Eclectic","Traditional Chinese","Ayurvedic","Mediterranean","Practitioner choice"], label="Tradition")
                    herb_evidence = gr.Dropdown(["Traditional use only","Traditional + clinical trials","Clinical evidence prioritised"], label="Evidence pref")
                    current_herbs = gr.Textbox(label="Current herbal use")
                with gr.Row():
                    pharmaceuticals = gr.Textbox(label="Pharmaceuticals (herb-drug check)", lines=2)
                    organ_focus     = gr.Textbox(label="Organ system focus", lines=2)

            with gr.Accordion("Functional Nutrition & Wellness", open=False):
                with gr.Row():
                    diet_pattern  = gr.Dropdown(["Omnivore","Vegetarian","Vegan","Paleo","Mediterranean","Ketogenic","AIP/Elimination","Whole food plant-based"], label="Diet")
                    meal_timing   = gr.Dropdown(["3 regular meals","Irregular/skipping","Intermittent fasting","Grazing/snacks"], label="Meals")
                    metgoal       = gr.Dropdown(["Energy & fatigue","Weight optimisation","Blood sugar regulation","Cardiovascular support","Hormonal balance","Cognitive performance","Athletic recovery"], label="Goal")
                    sleep_quality = gr.Slider(1, 10, step=1, label="Sleep quality", value=5)
                deficiencies = gr.Textbox(label="Suspected deficiencies (e.g. Vit D, B12, Mg)")
                supps        = gr.Textbox(label="Current supplements")
                with gr.Row():
                    gut_symptoms   = gr.Dropdown(["None","Bloating/gas","Reflux/heartburn","IBS pattern","SIBO suspected","Dysbiosis"], label="Gut")
                    inflam_markers = gr.Textbox(label="Inflammatory markers")

            active_mods = gr.CheckboxGroup(MODALITY_CHOICES, value=MODALITY_CHOICES, label="Active modalities")
            gen_btn     = gr.Button("Generate care plan", variant="primary")
            plan_output = gr.Textbox(label="Generated plan", lines=22)
            ep_id_out   = gr.Textbox(label="Episode ID — copy to Review tab", visible=False)
            go_review   = gr.Button("→ Switch to Review tab", visible=False)

            gen_btn.click(
                run_intake,
                inputs=[age, sex, weight, height, chief_complaint, wellness_goals,
                        tongue, pulse, emotion, sleep_p, tcm_complaints,
                        dosha, agni, vikruti, dhatu, ayur_lifestyle,
                        bowel, stress, water, exercise, sensitivities,
                        herb_trad, current_herbs, herb_evidence, pharmaceuticals, organ_focus,
                        diet_pattern, meal_timing, deficiencies, supps,
                        gut_symptoms, inflam_markers, metgoal, sleep_quality, active_mods],
                outputs=[plan_output, ep_id_out, go_review],
            )

        with gr.Tab("Clinician Review"):
            gr.Markdown("Paste an Episode ID from the Intake tab, review each item, then submit.")
            with gr.Row():
                ep_id_input = gr.Textbox(label="Episode ID")
                reviewer_id = gr.Textbox(label="Reviewer ID", value="DR01")
                load_btn    = gr.Button("Load episode")
            ep_summary = gr.Textbox(label="Episode summary", lines=12)

            with gr.Group(visible=False) as review_panel:
                gr.Markdown("### Review items one at a time (index 0, 1, 2 …)")
                with gr.Row():
                    item_idx = gr.Number(label="Item index", value=0, precision=0)
                    action   = gr.Radio(["approve","modify","reject"], label="Action", value="approve")
                note         = gr.Textbox(label="Clinician note (required when modifying)", lines=3)
                review_btn   = gr.Button("Save item review")
                review_status = gr.Textbox(label="Status")
                submit_btn   = gr.Button("Submit full review", variant="primary")
                submit_status = gr.Textbox(label="Result", lines=5)

            load_btn.click(load_episode_for_review, [ep_id_input], [ep_summary, review_panel])
            review_btn.click(review_item, [item_idx, action, note, reviewer_id], [review_status])
            submit_btn.click(submit_full_review, [reviewer_id], [submit_status])

        with gr.Tab("Analytics Dashboard"):
            refresh_btn   = gr.Button("Refresh")
            stats_box     = gr.Textbox(label="Summary")
            with gr.Row():
                reward_plot   = gr.Plot(label="Reward curve")
                approval_plot = gr.Plot(label="Actions by tradition")
            heatmap_plot  = gr.Plot(label="Approval heatmap")
            evidence_plot = gr.Plot(label="Actions by evidence grade")

            refresh_btn.click(refresh_dashboard,
                              outputs=[stats_box, reward_plot, approval_plot,
                                       heatmap_plot, evidence_plot])
            app.load(refresh_dashboard,
                     outputs=[stats_box, reward_plot, approval_plot,
                               heatmap_plot, evidence_plot])


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Integrative Medicine HITL-RL")
    print("="*60)
    print(f"  Episodes in store: {episode_count()}")
    print(f"  Data directory:    {EPISODES_DIR.parent}")
    print("="*60 + "\n")
    print("Starting app — a public gradio.live link will appear below.")
    print("Click it to open the app in your browser.\n")

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,        # generates a public gradio.live link
        inbrowser=False,
        show_error=True,
    )
