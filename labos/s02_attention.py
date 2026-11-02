import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Semaine 2 — attention")


@app.cell
def _():
    import marimo as mo
    import numpy as np

    from c1.verif import rapport, verifier
    return mo, np, rapport, verifier


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Laboratoire 2 — l'attention

    $$\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

    ## 1. Exercice — `softmax`

    Le softmax d'un vecteur $x$ est $e^{x_i} / \sum_j e^{x_j}$, appliqué ici à
    **chaque ligne** d'une matrice. Pour éviter les débordements, soustrayez
    d'abord le maximum de chaque ligne : le résultat ne change pas.
    """)
    return


@app.cell
def _(np):
    def softmax(x):
        # À compléter
        ...
    return (softmax,)


@app.cell(hide_code=True)
def _(np, rapport, softmax, verifier):
    _x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    rapport([
        verifier("chaque ligne somme à 1",
                 lambda: np.allclose(softmax(_x).sum(axis=-1), 1)),
        verifier("une ligne constante donne une distribution uniforme",
                 lambda: np.allclose(softmax(_x)[1], 1 / 3)),
        verifier("l'ordre des valeurs est conservé",
                 lambda: (np.diff(softmax(_x)[0]) > 0).all()),
        verifier("pas de débordement avec de grandes valeurs",
                 lambda: np.isfinite(softmax(np.array([[1000.0, 1001.0]]))).all()),
        verifier("−∞ donne un poids nul",
                 lambda: softmax(np.array([[0.0, -np.inf]]))[0, 1] == 0),
    ], titre="Tests — softmax")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Exercice — `attention`

    `attention(Q, K, V, causal)` retourne le couple `(sortie, poids)` :

    - `poids` : la matrice $n \times n$ des poids d'attention;
    - `sortie` : `poids @ V`.

    Avec `causal=True`, le jeton $i$ ne peut pas regarder les jetons $j > i$ :
    mettez leur score à $-\infty$ **avant** le softmax (`np.tri` vous sera
    utile).
    """)
    return


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
    ## 3. Expérience — la carte d'attention

    La phrase et la matrice $M$ de la page de la semaine. Modifiez $M$ pour que
    « tapis » regarde « sur » plutôt que « le », puis observez la carte.
    """)
    return


@app.cell
def _(np):
    CATEGORIES = ["dét", "nom", "adj", "verbe", "prép"]
    PHRASE = [("le", "dét"), ("chat", "nom"), ("noir", "adj"), ("dort", "verbe"),
              ("sur", "prép"), ("le", "dét"), ("tapis", "nom")]
    X = np.array([[c == cat for c in CATEGORIES] for _, cat in PHRASE], dtype=float)

    M = np.zeros((5, 5))
    M[2, 1] = 4  # l'adjectif cherche un nom
    M[3, 1] = 4  # le verbe cherche un nom
    M[1, 0] = 3  # le nom cherche un déterminant
    M[4, 3] = 3  # la préposition cherche un verbe
    M[0, 4] = 2  # le déterminant cherche une préposition
    return M, PHRASE, X


@app.cell(hide_code=True)
def _(M, PHRASE, X, attention, mo, np):
    mo.stop(
        not isinstance(attention(X, X, X), tuple),
        mo.md("*La carte apparaîtra quand `attention` fonctionnera.*"),
    )
    _, _poids = attention(X @ M, X, X)
    _mots = [m for m, _ in PHRASE]

    def _cellule(p):
        return (f"<td style='background:rgba(59,130,246,{p:.2f});text-align:center;"
                f"padding:4px 8px'>{p:.2f}</td>")

    _entete = "<tr><th></th>" + "".join(f"<th>{m}</th>" for m in _mots) + "</tr>"
    _lignes = "".join(
        f"<tr><th style='text-align:right'>{m}</th>"
        + "".join(_cellule(p) for p in _poids[i]) + "</tr>"
        for i, m in enumerate(_mots)
    )
    mo.vstack([
        mo.md("Ligne : le mot qui regarde. Colonne : le mot regardé."),
        mo.Html(f"<table style='border-collapse:collapse'>{_entete}{_lignes}</table>"),
        mo.md(f"« tapis » regarde surtout : **{_mots[int(np.argmax(_poids[6]))]}**"),
    ])
    return


if __name__ == "__main__":
    app.run()
