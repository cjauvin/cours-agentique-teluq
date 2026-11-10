"""Entraîne le mini-transformeur montré dans la page de la semaine 3.

Les pages tournent dans le navigateur (Pyodide), où PyTorch n'est pas
disponible : on entraîne ici, une fois, le petit modèle dont la page montre la
courbe de perte et quelques prédictions.

    uv run python outils/precalculer_mini_transformeur.py
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SORTIE = Path(__file__).parent.parent / "_includes" / "s03-mini-transformeur.qmd"

# Le corpus des semaines 1 et 2.
CORPUS = """
le chat dort sur le tapis . le chien dort sur le canapé .
le chat mange la souris . le chien mange le repas du chat .
la souris dort dans le trou . le chat regarde le chien .
le chien regarde la souris . la souris regarde le fromage .
la souris mange le fromage . le chat dort dans le panier .
le chien court dans le jardin . le chat court après la souris .
la souris court dans le trou . le chien dort dans le jardin .
le chat mange le fromage du chien . la souris dort sur le tapis .
"""
CONTEXTES = ["le chat mange le", "le chien mange le", "la souris dort dans le",
             "le chien court dans le"]
T = 8          # longueur du contexte
D = 16         # dimension des vecteurs
TETES = 2
PAS = 300
TAUX = 0.01

jetons = ["."] + CORPUS.split()
vocabulaire = sorted(set(jetons))
index = {m: i for i, m in enumerate(vocabulaire)}
V = len(vocabulaire)

# Le plancher : l'entropie croisée du bigramme par comptage (semaine 2).
comptes = defaultdict(Counter)
for a, b in zip(jetons, jetons[1:]):
    comptes[a][b] += 1
plancher = -sum(math.log2(comptes[a][b] / sum(comptes[a].values()))
                for a, b in zip(jetons, jetons[1:])) / (len(jetons) - 1)


class Bloc(nn.Module):
    """Un bloc : attention causale, puis un petit réseau par jeton."""

    def __init__(self):
        super().__init__()
        self.norme1, self.norme2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.attention = nn.MultiheadAttention(D, TETES, batch_first=True)
        self.reseau = nn.Sequential(nn.Linear(D, 4 * D), nn.ReLU(), nn.Linear(4 * D, D))

    def forward(self, x):
        n = x.shape[1]
        masque = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        h = self.norme1(x)
        x = x + self.attention(h, h, h, attn_mask=masque, need_weights=False)[0]
        return x + self.reseau(self.norme2(x))


class MiniTransformeur(nn.Module):
    def __init__(self):
        super().__init__()
        self.plongements = nn.Embedding(V, D)
        self.positions = nn.Embedding(T, D)
        self.bloc = Bloc()
        self.norme = nn.LayerNorm(D)
        self.sortie = nn.Linear(D, V)

    def forward(self, x):
        h = self.plongements(x) + self.positions(torch.arange(x.shape[1]))
        return self.sortie(self.norme(self.bloc(h)))


torch.manual_seed(0)
modele = MiniTransformeur()
ids = torch.tensor([index[m] for m in jetons])
# Toutes les fenêtres de T + 1 jetons : chaque position prédit le jeton suivant.
fenetres = torch.stack([ids[i:i + T + 1] for i in range(len(ids) - T)])
optimiseur = torch.optim.Adam(modele.parameters(), lr=TAUX)
pertes = []
for _ in range(PAS):
    logits = modele(fenetres[:, :-1])
    perte = F.cross_entropy(logits.reshape(-1, V), fenetres[:, 1:].reshape(-1)) / math.log(2)
    optimiseur.zero_grad()
    perte.backward()
    optimiseur.step()
    pertes.append(round(perte.item(), 3))


def meilleurs(probas, n=3):
    return [[m, round(p, 3)] for m, p in sorted(probas.items(), key=lambda c: -c[1])[:n]]


bigramme = {m: c / sum(comptes["le"].values()) for m, c in comptes["le"].items()}
predictions = []
with torch.no_grad():
    for contexte in CONTEXTES:
        x = torch.tensor([[index[m] for m in [".", *contexte.split()]]])
        p = F.softmax(modele(x)[0, -1], dim=-1)
        predictions.append({
            "contexte": contexte,
            "transformeur": meilleurs({m: p[i].item() for m, i in index.items()}),
            "bigramme": meilleurs(bigramme),
        })

donnees = {
    "plancher": round(plancher, 3),
    "pertes": pertes,
    "parametres": sum(p.numel() for p in modele.parameters()),
    "contexte": T, "dimension": D, "tetes": TETES, "taux": TAUX,
    "predictions": predictions,
}
SORTIE.write_text(
    "```python {.marimo}\n"
    "# Généré par outils/precalculer_mini_transformeur.py — ne pas modifier à la main.\n"
    f"MINI = __import__('json').loads(r'''{json.dumps(donnees, ensure_ascii=False, separators=(',', ':'))}''')\n"
    "```\n"
)
print(f"Plancher {plancher:.3f}, perte finale {pertes[-1]:.3f}, "
      f"{donnees['parametres']} paramètres → {SORTIE}")
for p in predictions:
    print(p)
