"""
knowledge_base.py — Builds and queries the five-tradition ChromaDB vector store.

Usage:
    python src/knowledge_base.py          # build / refresh the vector store
    python src/knowledge_base.py --query "liver qi stagnation fatigue"
"""

import argparse
import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_DIR, EMBED_MODEL, TRADITION_COLS

# ── Seed knowledge (one dict per tradition)
TRADITIONS = {
    "TCM": {
        "description": "Traditional Chinese Medicine — pattern differentiation, acupoints, herbal formulae",
        "seed_texts": [
            "Liver Qi stagnation presents with wiry pulse, hypochondriac distension, emotional irritability, and purple tongue edges. Treatment: soothe liver, move Qi. Formula: Chai Hu Shu Gan San. Key points: LV3, LV14, PC6, GB34.",
            "Kidney Yang deficiency: deep weak pulse especially chi position, cold limbs, frequent pale urination, lower back soreness, fatigue worsening in cold. Formula: Jin Gui Shen Qi Wan. Points: KD3, BL23, CV4, moxa.",
            "Spleen Qi deficiency: soggy pulse, pale swollen tongue with teeth marks, poor appetite, loose stools, fatigue after eating. Formula: Si Jun Zi Tang. Points: SP6, ST36, CV12, BL20.",
            "Phlegm-damp obstruction: slippery pulse, greasy white tongue coat, heaviness, brain fog, nausea, obesity. Formula: Er Chen Tang. Points: ST40, SP9, CV12, PC6.",
            "Heart Blood deficiency: thin pulse, pale tongue, palpitations, insomnia with vivid dreams, poor memory. Formula: Gui Pi Tang. Points: HT7, PC6, BL15, SP6.",
            "Yin deficiency with empty heat: rapid thin pulse, red tongue with no coat, night sweats, afternoon fever, dry throat. Formula: Liu Wei Di Huang Wan. Points: KD3, KD6, SP6, HT6.",
        ],
    },
    "Ayurveda": {
        "description": "Ayurvedic medicine — dosha theory, rasa, virya, vipaka, panchakarma",
        "seed_texts": [
            "Vata excess (Vata Vikruti): dry skin, constipation, variable appetite, anxiety, insomnia, joint cracking, cold sensitivity. Pacify with warm unctuous foods, Ashwagandha (Withania somnifera), Bala, sesame oil Abhyanga, Basti panchakarma. Avoid cold raw foods and irregular routine.",
            "Pitta excess: yellow greasy tongue coat, sharp rapid pulse, inflammation, skin rashes, acid reflux, irritability, night sweats. Pacify with cooling foods, Shatavari, Amalaki, Brahmi, Neem, Tikta Ghrita. Avoid spicy, fermented foods. Virechana panchakarma.",
            "Kapha excess: thick white tongue coat, slow deep pulse, weight gain, mucus congestion, sluggish digestion, depression. Use Trikatu, Guggulu, Triphala, dry vigorous exercise, Kapha-reducing fasting. Vamana panchakarma when indicated.",
            "Ama accumulation: coated tongue, foul breath, heavy joints, dull appetite, low-grade fatigue. Priority: Deepana-Pachana with Chitrakadi Vati, ginger, long pepper before any tonification.",
            "Meda dhatu imbalance (adipose): Guggulu (Commiphora mukul), Triphala Guggulu, Medohar Guggulu. Reduce Kapha diet. Udvartana dry powder massage.",
            "Majja dhatu (nerve/marrow) depletion: anxiety, memory loss, tremors, insomnia. Brahmi (Bacopa monnieri), Shankhpushpi, Ashwagandha, warm milk with turmeric and ghee. Shirodhara panchakarma.",
        ],
    },
    "Naturopathy": {
        "description": "Naturopathic medicine — vis medicatrix naturae, tolle causam, therapeutic order",
        "seed_texts": [
            "Naturopathic therapeutic order: (1) Remove obstacles to cure. (2) Stimulate vital force. (3) Support weakened systems. (4) Correct structural integrity. (5) Prescribe specific natural substances. (6) Pharmacological agents. (7) Surgery.",
            "Constitutional hydrotherapy protocol for immune stimulation: alternating hot and cold towel applications to trunk, 5 min hot / 1 min cold, 3 cycles. Stimulates lymphatic circulation and white blood cell activity.",
            "Liver detoxification support: Elimination diet (remove gluten, dairy, soy, corn, eggs, sugar, alcohol 21 days). Support phase I/II with cruciferous vegetables, NAC 600mg BD, milk thistle 140mg TID (70% silymarin), dandelion root tea.",
            "Stress and adrenal support: Adaptogen rotation — Rhodiola rosea (morning), Ashwagandha (evening). Phosphatidylserine 300mg. B-complex, Magnesium glycinate 400mg nocte. Sleep hygiene protocol.",
            "Gut restoration 4R protocol: Remove (pathogens, irritants), Replace (digestive enzymes, HCl), Reinoculate (Lactobacillus rhamnosus GG, Saccharomyces boulardii), Repair (L-glutamine 5g BD, zinc carnosine, slippery elm).",
            "Vis medicatrix naturae (the healing power of nature): the body has an inherent self-healing process. The physician's role is to support and facilitate this process, not to suppress symptoms. Identify and treat the cause (Tolle Causam) rather than symptom suppression.",
        ],
    },
    "Clinical Herbalism": {
        "description": "Evidence-informed clinical herbalism — Western Eclectic, actions, constituents, interactions",
        "seed_texts": [
            "Withania somnifera (Ashwagandha) — adaptogen, HPA axis modulation, cortisol reduction. Grade A: multiple RCTs. Dose: 300–600mg KSM-66 extract daily. Caution: thyroid conditions, nightshade sensitivity, pregnancy. Interaction: sedatives (additive), thyroid medications.",
            "Valeriana officinalis (Valerian) — nervine, GABAergic, sleep latency reduction. Grade B. Dose: 300–600mg standardised extract at bedtime or 5ml 1:3 tincture. Caution: hepatotoxicity at high long-term doses. Interaction: benzodiazepines, alcohol (potentiation).",
            "Silybum marianum (Milk thistle) — hepatoprotective, silymarin antioxidant, hepatocyte regeneration. Grade A. Dose: 140mg silymarin TID. Safe in pregnancy. Interaction: CYP3A4 substrates (mild inhibition), statins.",
            "Curcuma longa (Turmeric) — anti-inflammatory, NF-kB and COX-2 inhibition. Grade A. Dose: 500mg curcumin with piperine 5mg TID. Poor bioavailability without lipid or piperine. Interaction: anticoagulants including Warfarin (potentiates), chemotherapy.",
            "Echinacea purpurea — immune modulator, innate immune stimulation, reduces URI duration. Grade A. Dose: 300mg TID standardised to 4% alkylamides. Max 8 weeks continuous. Contraindicated: autoimmune conditions, immunosuppressants.",
            "Crataegus monogyna (Hawthorn) — cardiovascular, positive inotrope, vasodilator, antioxidant. Grade B. Dose: 160–900mg extract daily. Interaction: digoxin (additive), antihypertensives (additive hypotension). Safe long-term.",
        ],
    },
    "Functional Nutrition": {
        "description": "Functional medicine nutritional protocols — nutrient deficiencies, gut-brain axis, metabolic health",
        "seed_texts": [
            "Magnesium deficiency (estimated 50% of Western population): symptoms — muscle cramps, poor sleep, anxiety, constipation, fatigue, headaches. Testing: RBC magnesium. Repletion: Magnesium glycinate or malate 200–400mg daily. Food sources: dark leafy greens, pumpkin seeds, dark chocolate.",
            "Vitamin D optimisation: target 100–150 nmol/L (40–60 ng/mL). Dose: 2000–4000 IU D3 daily with K2 MK-7 100mcg. Co-factors: magnesium (activates D), zinc. Deficiency linked to autoimmunity, depression, infections, insulin resistance.",
            "Omega-3 fatty acids (EPA/DHA): anti-inflammatory via resolvin and protectin synthesis. Dose: 2–4g combined EPA+DHA daily. Fish oil or algae-based DHA for vegans. Monitor if on anticoagulants. Omega-3 index target: >8%.",
            "Gut microbiome and metabolic health: target 30+ different plant foods per week. Fermented foods (kefir, kimchi, sauerkraut) increase microbiome diversity. Butyrate from fermentable fibre supports colonocyte health and reduces intestinal permeability.",
            "Mitochondrial support protocol: CoQ10 100–300mg (ubiquinol preferred over 40yo), B-complex, Alpha lipoic acid 300mg, NAC 600mg, magnesium malate. Indicated: fatigue, fibromyalgia, post-viral syndromes.",
            "Blood sugar regulation: Berberine 500mg TID (Grade A: AMPK activation, comparable to metformin in T2D), Chromium picolinate 200–400mcg, Inositol 2–4g (PCOS, insulin resistance), time-restricted eating 16:8, low glycaemic load diet.",
        ],
    },
}


def get_client():
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_embed_fn():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)


def build_vector_store(verbose: bool = True):
    """Embed all seed texts and persist to ChromaDB. Safe to re-run (skips existing docs)."""
    client = get_client()
    emb_fn = get_embed_fn()

    for tradition, meta in TRADITIONS.items():
        col_name = TRADITION_COLS[tradition]
        try:
            col = client.get_collection(col_name, embedding_function=emb_fn)
        except Exception:
            col = client.create_collection(
                col_name,
                embedding_function=emb_fn,
                metadata={"tradition": tradition, "description": meta["description"]},
            )

        existing_ids = set(col.get()["ids"])
        to_add = [
            (f"seed_{i}", text)
            for i, text in enumerate(meta["seed_texts"])
            if f"seed_{i}" not in existing_ids
        ]

        if to_add:
            col.add(
                documents=[t for _, t in to_add],
                ids=[i for i, _ in to_add],
                metadatas=[{"tradition": tradition, "source": "seed"} for _ in to_add],
            )
            if verbose:
                print(f"[{tradition}] Added {len(to_add)} docs. Total: {col.count()}")
        else:
            if verbose:
                print(f"[{tradition}] Already up to date ({col.count()} docs).")

    if verbose:
        print("\nVector store ready at:", CHROMA_DIR)


def retrieve(query: str, tradition: str, n_results: int = 3) -> str:
    """Return the top-n relevant chunks for a query from a tradition's collection."""
    col_name = TRADITION_COLS.get(tradition)
    if not col_name:
        return ""
    try:
        client = get_client()
        emb_fn = get_embed_fn()
        col = client.get_collection(col_name, embedding_function=emb_fn)
        results = col.query(query_texts=[query], n_results=min(n_results, col.count()))
        return "\n".join(results["documents"][0])
    except Exception as e:
        return f"[Retrieval error: {e}]"


def corpus_stats() -> dict:
    """Return document counts per tradition (for EDA)."""
    client = get_client()
    emb_fn = get_embed_fn()
    stats = {}
    for tradition, col_name in TRADITION_COLS.items():
        try:
            col = client.get_collection(col_name, embedding_function=emb_fn)
            stats[tradition] = col.count()
        except Exception:
            stats[tradition] = 0
    return stats


def get_all_docs() -> dict:
    """Return all documents per tradition (for EDA embedding visualisation)."""
    client = get_client()
    emb_fn = get_embed_fn()
    docs = {}
    for tradition, col_name in TRADITION_COLS.items():
        try:
            col = client.get_collection(col_name, embedding_function=emb_fn)
            docs[tradition] = col.get()["documents"]
        except Exception:
            docs[tradition] = []
    return docs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--tradition", type=str, default="TCM")
    args = parser.parse_args()

    if args.query:
        print(f"\nQuery: {args.query}\nTradition: {args.tradition}\n")
        print(retrieve(args.query, args.tradition))
    else:
        build_vector_store()
