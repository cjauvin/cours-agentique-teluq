import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Semaine 1 — température")


@app.cell
def _():
    import math
    import random
    from collections import Counter, defaultdict

    import marimo as mo

    from agentique.verif import rapport, verifier
    from agentique.viz import barres
    return Counter, barres, defaultdict, math, mo, random, rapport, verifier


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Laboratoire 1 — la température

    Dans la page de la semaine, la fonction `probabilites` vous était donnée.
    Ici, vous l'écrivez, puis vous mesurez l'effet de la température sur le
    texte généré.
    """)
    return


@app.cell
def _(Counter, defaultdict):
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
    jetons = CORPUS.split()
    comptes = defaultdict(Counter)
    for precedent, suivant in zip(jetons, jetons[1:]):
        comptes[precedent][suivant] += 1
    return (comptes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Exercice — `probabilites`

    Chaque compte $c_i$ devient $c_i^{1/T}$, puis on normalise pour que la somme
    fasse 1 :

    $$p_i = \frac{c_i^{1/T}}{\sum_j c_j^{1/T}}$$
    """)
    return


@app.cell
def _():
    def probabilites(compteur, temperature):
        # À compléter
        ...
    return (probabilites,)


@app.cell(hide_code=True)
def _(probabilites, rapport, verifier):
    _c = {"chat": 6, "chien": 3, "fromage": 1}
    rapport([
        verifier("renvoie un dictionnaire",
                 lambda: isinstance(probabilites(_c, 1.0), dict)),
        verifier("la somme fait 1",
                 lambda: abs(sum(probabilites(_c, 0.7).values()) - 1) < 1e-9),
        verifier("à température 1, ce sont les fréquences",
                 lambda: abs(probabilites(_c, 1.0)["chat"] - 0.6) < 1e-9),
        verifier("à basse température, le plus fréquent domine",
                 lambda: probabilites(_c, 0.1)["chat"] > 0.99),
        verifier("à haute température, on s'approche de l'uniforme",
                 lambda: probabilites(_c, 100)["fromage"] > 0.3),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mot = mo.ui.text(value="le", label="Mot précédent")
    temperature = mo.ui.slider(
        0.1, 3.0, step=0.1, value=1.0, label="Température", show_value=True
    )
    mo.hstack([mot, temperature], justify="start", gap=2)
    return mot, temperature


@app.cell(hide_code=True)
def _(barres, comptes, mo, mot, probabilites, temperature):
    mo.stop(
        not isinstance(probabilites({"a": 1}, 1.0), dict),
        mo.md("*Les barres apparaîtront quand `probabilites` fonctionnera.*"),
    )
    barres(probabilites(comptes[mot.value], temperature.value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Expérience — entropie et diversité

    Pour chaque température, on mesure :

    - l'**entropie** de la distribution après « le », en bits : 0 si un seul
      mot est possible, $\log_2 n$ si les $n$ mots sont équiprobables;
    - la probabilité du mot le plus probable après « le »;
    - sur 200 phrases générées : la proportion de phrases **distinctes** et
      leur **longueur** moyenne.
    """)
    return


@app.cell
def _(comptes, math, probabilites, random):
    def entropie(probs):
        return -sum(p * math.log2(p) for p in probs.values() if p > 0)

    def generer(temperature, rng):
        phrase = ["le"]
        while phrase[-1] != "." and len(phrase) < 20:
            probs = probabilites(comptes[phrase[-1]], temperature)
            phrase.append(rng.choices(list(probs), weights=list(probs.values()))[0])
        return phrase

    def mesurer(temperature, n=200, graine=0):
        rng = random.Random(graine)
        phrases = [generer(temperature, rng) for _ in range(n)]
        apres_le = probabilites(comptes["le"], temperature)
        return {
            "température": temperature,
            "entropie après « le » (bits)": round(entropie(apres_le), 2),
            "p du plus probable": round(max(apres_le.values()), 2),
            "phrases distinctes": len({" ".join(p) for p in phrases}) / n,
            "longueur moyenne": sum(map(len, phrases)) / n,
        }
    return (mesurer,)


@app.cell
def _(mesurer, mo, probabilites):
    mo.stop(
        not isinstance(probabilites({"a": 1}, 1.0), dict),
        mo.md("*L'expérience s'exécutera quand `probabilites` fonctionnera.*"),
    )
    mo.ui.table([mesurer(t) for t in (0.1, 0.5, 1.0, 1.5, 2.0, 3.0)], selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Questions

    Notez vos réponses dans votre carnet de laboratoire.

    1. Décrivez l'entropie en fonction de la température. Quelles sont ses
       valeurs limites, quand $T \to 0$ et quand $T \to \infty$?
    2. À température 0,1, les phrases ne sont pas toutes identiques. Pourquoi?
       (Regardez les comptes après « le ».)
    3. Pourquoi la longueur moyenne des phrases change-t-elle avec la
       température?
    4. Quelle température choisiriez-vous pour un agent qui doit modifier du code?
       Pour un remue-méninges? Justifiez avec vos mesures.
    """)
    return


if __name__ == "__main__":
    app.run()
