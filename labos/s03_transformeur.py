import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Semaine 3 — le transformeur")


@app.cell
def _():
    import math
    from collections import Counter, defaultdict

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from agentique.verif import rapport, verifier
    from agentique.viz import courbe_perte
    return Counter, F, courbe_perte, defaultdict, math, mo, nn, np, rapport, torch, verifier


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Laboratoire 3 — le transformeur

    Quatre parties :

    1. écrire l'attention causale en NumPy;
    2. la réécrire en PyTorch, et laisser PyTorch calculer un gradient à votre
       place;
    3. entraîner le mini-transformeur de la page, et mesurer ce que lui
       rapporte son contexte;
    4. ouvrir un vrai modèle, Qwen2.5-0.5B. La partie 4 le télécharge à la
       première exécution (environ 1 Go), puis le garde en cache.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Exercice — l'attention en NumPy

    $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

    Le softmax par ligne vous est donné (vous l'avez écrit au laboratoire 2).
    `attention(Q, K, V, causal)` retourne le couple `(sortie, poids)` :

    - `poids` : la matrice $n \times n$ des poids d'attention;
    - `sortie` : `poids @ V`.

    $d$ est le nombre de colonnes de $K$. Avec `causal=True`, le jeton $i$ ne
    peut pas regarder les jetons $j > i$ : mettez leur score à $-\infty$
    **avant** le softmax (`np.tri` vous sera utile).
    """)
    return


@app.cell
def _(np):
    def softmax(x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    return (softmax,)


@app.cell
def _(np, softmax):
    def attention(Q, K, V, causal=True):
        # À compléter
        ...
    return (attention,)


@app.cell(hide_code=True)
def _(attention, np, rapport, verifier):
    _rng = np.random.default_rng(0)
    _Q, _K, _V = (_rng.normal(size=(4, 8)) for _ in range(3))

    def _poids(causal):
        return attention(_Q, _K, _V, causal)[1]

    rapport([
        verifier("retourne (sortie, poids) aux bonnes dimensions",
                 lambda: attention(_Q, _K, _V)[0].shape == (4, 8)
                 and _poids(True).shape == (4, 4)),
        verifier("chaque ligne de poids somme à 1",
                 lambda: np.allclose(_poids(True).sum(axis=-1), 1)),
        verifier("masque causal : aucun poids vers l'avenir",
                 lambda: np.allclose(np.triu(_poids(True), k=1), 0)),
        verifier("le premier jeton ne regarde que lui-même",
                 lambda: np.isclose(_poids(True)[0, 0], 1)),
        verifier("sans masque, tous les poids sont positifs",
                 lambda: (_poids(False) > 0).all()),
        verifier("sortie = poids @ V",
                 lambda: np.allclose(attention(_Q, _K, _V)[0], _poids(True) @ _V)),
        verifier("division par √d",
                 lambda: np.allclose(
                     _poids(False)[0],
                     np.exp(_Q[0] @ _K.T / np.sqrt(8)) / np.exp(_Q[0] @ _K.T / np.sqrt(8)).sum())),
    ], titre="Tests — attention")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Exercice — la même chose en PyTorch

    PyTorch manipule des **tenseurs** (*tensors*), des tableaux à plusieurs
    dimensions qui ressemblent beaucoup à ceux de NumPy : `@`, `.T`,
    `.shape` s'utilisent de la même façon. Quelques équivalents :

    | NumPy | PyTorch |
    |:--|:--|
    | `np.tri(n, dtype=bool)` | `torch.tril(torch.ones(n, n, dtype=torch.bool))` |
    | `np.where(masque, a, -np.inf)` | `a.masked_fill(~masque, float("-inf"))` |
    | votre `softmax(x)` | `torch.softmax(x, dim=-1)` |
    | `np.sqrt(d)` | `math.sqrt(d)` |

    Écrivez `attention_torch(Q, K, V)`, l'attention **causale**, qui retourne
    `(sortie, poids)` comme en partie 1, mais avec des tenseurs.
    """)
    return


@app.cell
def _(math, torch):
    def attention_torch(Q, K, V):
        # À compléter
        ...
    return (attention_torch,)


@app.cell(hide_code=True)
def _(F, attention, attention_torch, np, rapport, torch, verifier):
    _g = torch.Generator().manual_seed(0)
    _Q, _K, _V = (torch.randn(5, 8, generator=_g) for _ in range(3))

    rapport([
        verifier("même résultat que votre version NumPy",
                 lambda: np.allclose(attention_torch(_Q, _K, _V)[0].numpy(),
                                     attention(_Q.numpy(), _K.numpy(), _V.numpy())[0], atol=1e-5)),
        verifier("même résultat que l'attention de PyTorch",
                 lambda: torch.allclose(attention_torch(_Q, _K, _V)[0],
                                        F.scaled_dot_product_attention(_Q, _K, _V, is_causal=True),
                                        atol=1e-5)),
        verifier("masque causal : aucun poids vers l'avenir",
                 lambda: torch.allclose(torch.triu(attention_torch(_Q, _K, _V)[1], diagonal=1),
                                        torch.zeros(5, 5))),
    ], titre="Tests — attention_torch")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Le gradient, calculé par PyTorch

    À la semaine 2, nous avons écrit la rétropropagation à la main. Le cœur en
    était le gradient de l'entropie croisée par rapport aux logits : $p - y$, la
    distribution prédite moins la « bonne réponse » (1 pour le bon jeton, 0 pour
    les autres), divisé par le nombre d'exemples $n$ quand la perte est une
    moyenne.

    PyTorch, lui, calcule les gradients tout seul : c'est la
    **différentiation automatique** (*automatic differentiation*). On marque les
    tenseurs dont on veut le gradient (`requires_grad=True`), on calcule la
    perte, puis `perte.backward()` remplit leur attribut `.grad`. La cellule
    suivante le fait pour des logits tirés au hasard. `F.cross_entropy` calcule
    l'entropie croisée avec le logarithme naturel plutôt qu'en base 2; le
    gradient est alors exactement $p - y$, sans facteur $\ln 2$.
    """)
    return


@app.cell
def _(F, torch):
    torch.manual_seed(0)
    logits = torch.randn(4, 6, requires_grad=True)  # 4 exemples, 6 jetons possibles
    cibles = torch.tensor([0, 3, 5, 3])             # le bon jeton de chaque exemple
    perte = F.cross_entropy(logits, cibles)
    perte.backward()
    logits.grad
    return cibles, logits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Écrivez `gradient_a_la_main(logits, cibles)`, qui calcule le même gradient
    sans `backward()` : $(p - y) / n$, où $p$ est le softmax de chaque ligne et
    $y$ le vecteur one-hot du bon jeton (`F.one_hot(cibles, V)` le construit).
    """)
    return


@app.cell
def _(F, torch):
    def gradient_a_la_main(logits, cibles):
        # À compléter
        ...
    return (gradient_a_la_main,)


@app.cell(hide_code=True)
def _(cibles, gradient_a_la_main, logits, rapport, torch, verifier):
    rapport([
        verifier("égal au gradient calculé par PyTorch",
                 lambda: torch.allclose(gradient_a_la_main(logits.detach(), cibles), logits.grad,
                                        atol=1e-6)),
        verifier("chaque ligne somme à 0 (la probabilité ne fait que se déplacer)",
                 lambda: torch.allclose(gradient_a_la_main(logits.detach(), cibles).sum(dim=1),
                                        torch.zeros(4), atol=1e-6)),
        verifier("négatif pour le bon jeton, positif ailleurs",
                 lambda: bool((gradient_a_la_main(logits.detach(), cibles)
                               .gather(1, cibles[:, None]) < 0).all())),
    ], titre="Tests — gradient_a_la_main")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pour le bigramme, on pouvait encore écrire ce gradient à la main. Pour un
    transformeur, avec ses attentions, ses normalisations et ses connexions
    résiduelles, personne ne le fait : `backward()` s'en charge, pour tous les
    paramètres à la fois.

    ## 3. Expérience — le mini-transformeur

    Voici le mini-transformeur de la page : un bloc, deux têtes, des vecteurs de
    dimension 16. `nn.MultiheadAttention` est l'attention que vous venez
    d'écrire, avec plusieurs têtes et ses propres matrices $W_Q$, $W_K$ et
    $W_V$; le petit réseau est celui de la semaine 2. Une seule chose change
    d'une expérience à l'autre : la longueur du **contexte**, le nombre de
    jetons que le modèle voit avant de prédire.
    """)
    return


@app.cell
def _(Counter, defaultdict, math, torch):
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
    jetons = ["."] + CORPUS.split()
    vocabulaire = sorted(set(jetons))
    index = {m: i for i, m in enumerate(vocabulaire)}
    V = len(vocabulaire)
    ids = torch.tensor([index[m] for m in jetons])

    # Le plancher : l'entropie croisée du bigramme par comptage (semaine 2)
    _comptes = defaultdict(Counter)
    for _a, _b in zip(jetons, jetons[1:]):
        _comptes[_a][_b] += 1
    plancher = -sum(math.log2(_comptes[_a][_b] / sum(_comptes[_a].values()))
                    for _a, _b in zip(jetons, jetons[1:])) / (len(jetons) - 1)
    return V, ids, index, plancher, vocabulaire


@app.cell
def _(V, nn, torch):
    D, TETES = 16, 2


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
        def __init__(self, contexte):
            super().__init__()
            self.plongements = nn.Embedding(V, D)
            self.positions = nn.Embedding(contexte, D)
            self.bloc = Bloc()
            self.norme = nn.LayerNorm(D)
            self.sortie = nn.Linear(D, V)

        def forward(self, x):
            h = self.plongements(x) + self.positions(torch.arange(x.shape[1]))
            return self.sortie(self.norme(self.bloc(h)))
    return (MiniTransformeur,)


@app.cell
def _(F, MiniTransformeur, V, ids, math, torch):
    def entrainer(contexte, pas=300, taux=0.01, graine=0):
        torch.manual_seed(graine)
        modele = MiniTransformeur(contexte)
        # Toutes les fenêtres de contexte + 1 jetons : chaque position prédit la suivante
        fenetres = torch.stack([ids[i:i + contexte + 1] for i in range(len(ids) - contexte)])
        optimiseur = torch.optim.Adam(modele.parameters(), lr=taux)
        pertes = []
        for _ in range(pas):
            logits = modele(fenetres[:, :-1])
            perte = F.cross_entropy(logits.reshape(-1, V), fenetres[:, 1:].reshape(-1))
            optimiseur.zero_grad()
            perte.backward()      # la rétropropagation, faite par PyTorch
            optimiseur.step()     # un pas d'Adam sur tous les paramètres
            pertes.append(perte.item() / math.log(2))  # en bits
        return modele, pertes
    return (entrainer,)


@app.cell
def _(mo):
    contexte = mo.ui.slider(1, 8, value=8, label="Contexte (jetons)", show_value=True)
    contexte
    return (contexte,)


@app.cell
def _(contexte, courbe_perte, entrainer, mo, plancher):
    modele_mini, pertes_mini = entrainer(contexte.value)
    mo.vstack([
        courbe_perte(pertes_mini, plancher, haut=5),
        mo.md(f"Contexte de {contexte.value} jeton(s) : perte finale "
              f"**{pertes_mini[-1]:.3f} bits** (plancher du bigramme : {plancher:.3f})."),
    ])
    return (modele_mini,)


@app.cell
def _(F, contexte, index, modele_mini, torch, vocabulaire):
    def predire(debut, n=3):
        """Les n jetons les plus probables après un début de phrase."""
        x = [index[m] for m in [".", *debut.split()]][-contexte.value:]
        with torch.no_grad():
            p = F.softmax(modele_mini(torch.tensor([x]))[0, -1], dim=-1)
        meilleurs = torch.argsort(p, descending=True)[:n]
        return ", ".join(f"{vocabulaire[i]} ({p[i]:.0%})" for i in meilleurs)


    {d: predire(d) for d in ["le chat mange le", "le chien mange le",
                             "la souris dort dans le", "le chien court dans le"]}
    return


@app.cell
def _(F, contexte, index, math, modele_mini, torch):
    def perte_sur(texte):
        """Perte moyenne du mini-transformeur sur un texte (en bits), jeton par jeton."""
        jetons_texte = [".", *texte.split()]
        total = 0.0
        with torch.no_grad():
            for t in range(1, len(jetons_texte)):
                x = [index[m] for m in jetons_texte[max(0, t - contexte.value):t]]
                logits = modele_mini(torch.tensor([x]))[0, -1]
                total += F.cross_entropy(logits[None], torch.tensor([index[jetons_texte[t]]])).item()
        return total / (len(jetons_texte) - 1) / math.log(2)


    perte_sur("le chien mange la souris . le chat dort sur le canapé .")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Questions.**

    1. Réglez le contexte à 1 jeton. Où s'arrête la courbe? Pourquoi
       exactement là? (Relisez la section sur l'entropie croisée de la
       semaine 2.)
    2. Faites passer le contexte de 1 à 8, et notez chaque fois la perte finale.
       Que rapporte chaque jeton de contexte supplémentaire?
    3. `perte_sur` mesure la perte sur un texte de votre choix, fait des mots
       du corpus. Écrivez trois phrases **absentes** du corpus (comme « le chien
       mange la souris . »), et comparez leur perte à la perte d'entraînement,
       pour plusieurs longueurs de contexte. Le modèle a-t-il appris les
       régularités du corpus, ou l'a-t-il appris par cœur?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Expérience — un vrai modèle

    La cellule suivante charge **Qwen2.5-0.5B**, le modèle de base ouvert dans
    la page.
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
    ### Exercice — `perplexite`

    `perplexite(texte)` retourne la perplexité **par jeton** de Qwen sur un
    texte :

    1. découpez le texte en jetons : `tok(texte, return_tensors="pt").input_ids`;
    2. calculez les logits sans gradient : `with torch.no_grad(): modele(ids).logits`;
    3. les logits de la position $t$ prédisent le jeton $t + 1$ : alignez-les
       avec les bons jetons (le premier jeton n'est pas prédit);
    4. calculez l'entropie croisée avec `F.cross_entropy` (en nats), puis la
       perplexité : $e^H$, qui vaut la même chose que $2^H$ quand $H$ est en
       bits.
    """)
    return


@app.cell
def _(F, math, modele, tok, torch):
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
    ### Expérience — les têtes du jeton précédent

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
