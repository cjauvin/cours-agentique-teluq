import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Semaine 3 — un vrai petit modèle")


@app.cell
def _():
    import math

    import marimo as mo
    import torch
    import torch.nn.functional as F

    from agentique.verif import rapport, verifier
    return F, math, mo, rapport, torch, verifier


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Laboratoire 3 — un vrai petit modèle

    Ce laboratoire utilise **Qwen2.5-0.5B**, le modèle de base ouvert dans la page
    de la semaine. La cellule suivante le télécharge à la première exécution
    (environ 1 Go), puis le garde en cache.
    """)
    return


@app.cell
def _(torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    NOM = "Qwen/Qwen2.5-0.5B"
    tok = AutoTokenizer.from_pretrained(NOM)
    # En float32 plutôt qu'en bfloat16, le format du fichier, pour des calculs
    # précis; « eager » est la seule implémentation qui retourne les poids
    # d'attention.
    modele = AutoModelForCausalLM.from_pretrained(
        NOM, dtype=torch.float32, attn_implementation="eager"
    )
    modele.eval()
    return modele, tok


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Exercice — `entropie_croisee`

    `entropie_croisee(logits, cibles)` reçoit :

    - `logits` : un tenseur $n \times V$, une ligne de logits par position;
    - `cibles` : un tenseur de $n$ entiers, le bon jeton à chaque position.

    Elle retourne l'entropie croisée moyenne **en bits** :
    $-\frac{1}{n}\sum_t \log_2 \mathrm{softmax}(z_t)[y_t]$.

    Calculez le logarithme du softmax de façon stable : soustrayez le maximum de
    chaque ligne avant l'exponentielle (comme pour le softmax de la semaine 2),
    ou utilisez `torch.logsumexp`. N'utilisez pas `F.cross_entropy` : c'est
    elle qui sert à vous vérifier.
    """)
    return


@app.cell
def _(math, torch):
    def entropie_croisee(logits, cibles):
        # À compléter
        ...
    return (entropie_croisee,)


@app.cell(hide_code=True)
def _(F, entropie_croisee, math, rapport, torch, verifier):
    _g = torch.Generator().manual_seed(0)
    _z = torch.randn(6, 10, generator=_g)
    _y = torch.randint(0, 10, (6,), generator=_g)

    def _valeur(z, y):
        return float(entropie_croisee(z, y))

    rapport([
        verifier("retourne un seul nombre",
                 lambda: torch.as_tensor(entropie_croisee(_z, _y)).numel() == 1),
        verifier("égale à F.cross_entropy, convertie en bits",
                 lambda: math.isclose(_valeur(_z, _y),
                                      float(F.cross_entropy(_z, _y)) / math.log(2), rel_tol=1e-4)),
        verifier("des logits égaux donnent log2(V) bits",
                 lambda: math.isclose(_valeur(torch.zeros(3, 8), torch.tensor([0, 3, 7])), 3.0,
                                      rel_tol=1e-5)),
        verifier("une prédiction certaine et juste coûte presque 0 bit",
                 lambda: _valeur(torch.tensor([[50.0, 0.0, 0.0]]), torch.tensor([0])) < 1e-6),
        verifier("pas de débordement avec de grands logits",
                 lambda: math.isfinite(_valeur(torch.tensor([[1000.0, 0.0]]), torch.tensor([1])))),
    ], titre="Tests — entropie_croisee")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Exercice — `perplexite`

    `perplexite(texte)` retourne la perplexité **par jeton** de Qwen sur un
    texte :

    1. découpez le texte en jetons : `tok(texte, return_tensors="pt").input_ids`;
    2. calculez les logits sans gradient : `with torch.no_grad(): modele(ids).logits`;
    3. les logits de la position $t$ prédisent le jeton $t + 1$ : alignez-les
       avec les bons jetons (le premier jeton n'est pas prédit);
    4. appliquez `entropie_croisee`, puis $2^H$.
    """)
    return


@app.cell
def _(entropie_croisee, modele, tok, torch):
    def perplexite(texte):
        # À compléter
        ...
    return (perplexite,)


@app.cell(hide_code=True)
def _(math, modele, perplexite, rapport, tok, torch, verifier):
    _phrase = "Le chat noir qui dormait sur le tapis s'est réveillé."

    def _reference(texte):
        _ids = tok(texte, return_tensors="pt").input_ids
        with torch.no_grad():
            return math.exp(float(modele(_ids, labels=_ids).loss))

    rapport([
        verifier("égale à la perplexité calculée par transformers",
                 lambda: math.isclose(float(perplexite(_phrase)), _reference(_phrase), rel_tol=1e-3)),
        verifier("l'accord correct est moins surprenant que l'erreur",
                 lambda: perplexite(_phrase) < perplexite(_phrase.replace("réveillé", "réveillée"))),
        verifier("une phrase vaut mieux que ses mots mélangés",
                 lambda: perplexite(_phrase) < perplexite("tapis le s'est qui noir réveillé sur chat dormait Le.")),
    ], titre="Tests — perplexite")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Expérience — les têtes du jeton précédent

    Dans la page, la tête 7 de la couche 8 regardait presque toujours le jeton
    précédent. L'a-t-on trouvée par chance? Cherchons systématiquement, parmi
    les 24 × 14 = 336 têtes, celles qui le font, puis vérifions qu'elles le font
    aussi sur des phrases qu'on n'a pas utilisées pour les trouver.

    `score_precedent(texte)` retourne un tenseur $24 \times 14$ : pour chaque
    tête, le poids moyen qu'elle accorde au jeton précédent.
    """)
    return


@app.cell
def _(modele, tok, torch):
    def score_precedent(texte):
        ids = tok(texte, return_tensors="pt").input_ids
        with torch.no_grad():
            poids = torch.stack(modele(ids, output_attentions=True).attentions)[:, 0]
        n = ids.shape[1]
        # poids[c, h, i, i - 1] pour i = 1 .. n - 1
        diagonale = poids[:, :, torch.arange(1, n), torch.arange(0, n - 1)]
        return diagonale.mean(dim=-1)
    return (score_precedent,)


@app.cell
def _(score_precedent, torch):
    TROUVER = "Le chat noir qui dormait sur le tapis s'est réveillé."
    VERIFIER = [
        "Les enfants que nous avons vus hier jouent dehors.",
        "def moyenne(valeurs):\n    return sum(valeurs) / len(valeurs)",
        "The quick brown fox jumps over the lazy dog.",
    ]

    _scores = score_precedent(TROUVER)
    meilleures = [divmod(int(i), _scores.shape[1]) for i in torch.argsort(_scores.flatten(), descending=True)[:5]]
    verification = torch.stack([score_precedent(t) for t in VERIFIER])
    return TROUVER, VERIFIER, meilleures, verification


@app.cell(hide_code=True)
def _(TROUVER, VERIFIER, meilleures, mo, score_precedent, verification):
    _s = score_precedent(TROUVER)
    _entete = "| couche, tête | phrase de départ | " + " | ".join(
        f"vérif. {i + 1}" for i in range(len(VERIFIER))) + " |\n"
    _entete += "|:--|--:|" + "--:|" * len(VERIFIER) + "\n"
    _lignes = "\n".join(
        f"| {c}, {h} | {_s[c, h]:.0%} | "
        + " | ".join(f"{verification[i, c, h]:.0%}" for i in range(len(VERIFIER))) + " |"
        for c, h in meilleures
    )
    mo.md(
        "Les cinq têtes qui regardent le plus le jeton précédent dans la phrase de "
        f"départ, et leur score sur les phrases de vérification :\n\n{_entete}{_lignes}\n\n"
        + "\n".join(f"{i + 1}. `{t!r}`" for i, t in enumerate(VERIFIER))
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions.**

    1. Les têtes trouvées sur une seule phrase gardent-elles leur rôle sur du
       code et sur de l'anglais? Qu'est-ce que ça suggère sur ce qu'elles ont
       appris?
    2. Adaptez `score_precedent` pour mesurer le poids accordé au **premier**
       jeton (le puits d'attention). Dans quelles couches les puits sont-ils
       les plus fréquents?
    3. Tentez la même démarche pour la tête « du sujet » (couche 5, tête 5) :
       quel score faudrait-il calculer, et sur quelles phrases la vérifier?
    """)
    return


if __name__ == "__main__":
    app.run()
