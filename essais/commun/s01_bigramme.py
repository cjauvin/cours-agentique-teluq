import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Semaine 1 — le bigramme")


@app.cell
def _():
    import random
    from collections import Counter, defaultdict

    import marimo as mo
    return Counter, defaultdict, mo, random


@app.cell
def _():
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
    return (jetons,)


@app.cell
def _(Counter, defaultdict, jetons):
    comptes = defaultdict(Counter)
    for precedent, suivant in zip(jetons, jetons[1:]):
        comptes[precedent][suivant] += 1
    vocabulaire = sorted(comptes)
    return comptes, vocabulaire


@app.cell
def _(mo, vocabulaire):
    mot = mo.ui.dropdown(vocabulaire, value="le", label="Mot précédent")
    temperature = mo.ui.slider(
        0.1, 3.0, step=0.1, value=1.0, label="Température", show_value=True
    )
    mo.hstack([mot, temperature], justify="start", gap=2)
    return mot, temperature


@app.cell
def _(comptes, mo, mot, temperature):
    def probabilites(compteur, temperature):
        poids = {m: c ** (1 / temperature) for m, c in compteur.items()}
        total = sum(poids.values())
        return {m: p / total for m, p in poids.items()}

    def barres(probs):
        lignes = []
        for m, p in sorted(probs.items(), key=lambda x: -x[1]):
            lignes.append(
                f"<div style='display:flex;align-items:center;gap:.5em;margin:2px 0'>"
                f"<code style='width:6em'>{m}</code>"
                f"<div style='background:#3b82f6;height:1em;width:{p * 300:.0f}px'></div>"
                f"<span>{p:.0%}</span></div>"
            )
        return mo.Html("".join(lignes))

    barres(probabilites(comptes[mot.value], temperature.value))
    return (probabilites,)


@app.cell
def _(mo):
    generer = mo.ui.run_button(label="Générer une phrase")
    generer
    return (generer,)


@app.cell
def _(comptes, generer, mo, probabilites, random, temperature):
    mo.stop(not generer.value, mo.md("*Clique sur le bouton pour générer.*"))
    phrase = ["le"]
    while phrase[-1] != "." and len(phrase) < 20:
        probs = probabilites(comptes[phrase[-1]], temperature.value)
        phrase.append(random.choices(list(probs), weights=list(probs.values()))[0])
    mo.md(f"**{' '.join(phrase)}**")
    return


if __name__ == "__main__":
    app.run()
