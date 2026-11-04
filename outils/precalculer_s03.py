"""Précalcule les sorties de Qwen2.5-0.5B affichées dans la page de la semaine 3.

Les pages tournent dans le navigateur (Pyodide), où PyTorch n'est pas
disponible : on calcule ici, une fois, ce que la page montre du vrai modèle.

    uv run python outils/precalculer_s03.py
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NOM = "Qwen/Qwen2.5-0.5B"
SORTIE = Path(__file__).parent.parent / "_includes" / "s03-donnees.qmd"

DEBUTS = [
    "Le chat dort sur le",
    "Le chat dort.",
    "La souris mange le",
    "Le chat noir qui dormait sur le tapis s'est",
    "La capitale du Canada est",
    "Quelle est la capitale du Canada?",
    "def moyenne(valeurs):\n    return",
]
PHRASES = [
    "Le chat noir qui dormait sur le tapis s'est réveillé.",
    "Le chat noir qui dormait sur le tapis s'est réveillée.",
    "Les enfants que nous avons vus hier jouent dehors.",
    "Les enfants que nous avons vus hier joue dehors.",
    "le chien dort sur le fromage .",
    "le fromage dort .",
]
PHRASE_ATTENTION = "Le chat noir qui dormait sur le tapis s'est réveillé."
N_MEILLEURS = 40
N_SUITE = 40
N_ECHANTILLONS = 3

tok = AutoTokenizer.from_pretrained(NOM)
# En float32, comme dans le laboratoire, pour que les nombres coïncident.
modele = AutoModelForCausalLM.from_pretrained(NOM, dtype=torch.float32)
modele.eval()
torch.manual_seed(0)


def jetons(ids):
    return [tok.decode(i) for i in ids]


def debut(texte):
    ids = tok(texte, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = modele(ids).logits[0, -1]
    meilleurs = logits.topk(N_MEILLEURS)
    glouton = modele.generate(ids, max_new_tokens=N_SUITE, do_sample=False)
    echantillons = modele.generate(
        ids, max_new_tokens=N_SUITE, do_sample=True, temperature=1.0, top_k=0,
        num_return_sequences=N_ECHANTILLONS,
    )
    return {
        "texte": texte,
        "jetons": jetons(ids[0]),
        "suivants": [[tok.decode(i), round(v, 3)] for v, i in
                     zip(meilleurs.values.tolist(), meilleurs.indices.tolist())],
        "glouton": tok.decode(glouton[0, ids.shape[1]:], skip_special_tokens=True),
        "echantillons": [tok.decode(e[ids.shape[1]:], skip_special_tokens=True) for e in echantillons],
    }


def surprise(texte):
    ids = tok(texte, return_tensors="pt").input_ids
    with torch.no_grad():
        logprobs = modele(ids).logits[0, :-1].log_softmax(-1)
    bits = -logprobs.gather(1, ids[0, 1:, None])[:, 0] / torch.log(torch.tensor(2.0))
    return {"texte": texte, "jetons": jetons(ids[0]), "bits": [abs(round(b, 2)) for b in bits.tolist()]}


def attention(texte):
    # Seule l'implémentation « eager » retourne les poids d'attention.
    modele.set_attn_implementation("eager")
    ids = tok(texte, return_tensors="pt").input_ids
    with torch.no_grad():
        poids = torch.stack(modele(ids, output_attentions=True).attentions)[:, 0]
    # En pourcentages entiers, ligne i limitée aux colonnes 0..i (masque causal).
    pourcents = (poids * 100).round().int().tolist()
    return {
        "jetons": jetons(ids[0]),
        "poids": [[[ligne[: i + 1] for i, ligne in enumerate(tete)] for tete in couche]
                  for couche in pourcents],
    }


config = modele.config
donnees = {
    "modele": {
        "nom": NOM,
        "parametres": sum(p.numel() for p in modele.parameters()),
        "couches": config.num_hidden_layers,
        "tetes": config.num_attention_heads,
        "dimension": config.hidden_size,
        "vocabulaire": len(tok),
    },
    "debuts": [debut(t) for t in DEBUTS],
    "phrases": [surprise(t) for t in PHRASES],
    "attention": attention(PHRASE_ATTENTION),
}

SORTIE.write_text(
    "```python {.marimo}\n"
    "# Généré par outils/precalculer_s03.py — ne pas modifier à la main.\n"
    f"QWEN = __import__('json').loads(r'''{json.dumps(donnees, ensure_ascii=False, separators=(',', ':'))}''')\n"
    "```\n"
)
print(f"{SORTIE} : {SORTIE.stat().st_size / 1000:.0f} ko")
