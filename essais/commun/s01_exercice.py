import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Semaine 1 — exercice")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    def probabilites(compteur, temperature):
        # À compléter : chaque compte c devient c ** (1 / temperature),
        # puis on normalise pour que la somme fasse 1.
        ...
    return (probabilites,)


@app.cell
def _(mo, probabilites):
    def verifier(nom, condition):
        try:
            ok = bool(condition())
        except Exception as e:
            return f"❌ {nom} — erreur : {e}"
        return f"{'✅' if ok else '❌'} {nom}"

    compteur = {"chat": 6, "chien": 3, "fromage": 1}
    resultats = [
        verifier("renvoie un dictionnaire",
                 lambda: isinstance(probabilites(compteur, 1.0), dict)),
        verifier("la somme fait 1",
                 lambda: abs(sum(probabilites(compteur, 0.7).values()) - 1) < 1e-9),
        verifier("à température 1, ce sont les fréquences",
                 lambda: abs(probabilites(compteur, 1.0)["chat"] - 0.6) < 1e-9),
        verifier("à basse température, le plus fréquent domine",
                 lambda: probabilites(compteur, 0.1)["chat"] > 0.99),
        verifier("à haute température, on s'approche de l'uniforme",
                 lambda: probabilites(compteur, 100)["fromage"] > 0.3),
    ]
    mo.md("\n\n".join(resultats))
    return


if __name__ == "__main__":
    app.run()
