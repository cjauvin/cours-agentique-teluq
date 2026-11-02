"""Vérifications autocorrigées, affichées ✅/❌ dans les notebooks."""

import marimo as mo


def verifier(description, test):
    """Exécute `test()` et retourne une ligne ✅/❌; une exception compte comme un échec."""
    try:
        ok = bool(test())
    except Exception as erreur:
        return f"❌ {description} — erreur : {erreur}"
    return ("✅ " if ok else "❌ ") + description


def rapport(resultats, titre="Tests"):
    """Affiche une liste de lignes produites par `verifier`."""
    reussis = sum(r.startswith("✅") for r in resultats)
    lignes = "\n".join(f"- {r}" for r in resultats)
    return mo.md(f"### {titre} ({reussis}/{len(resultats)})\n\n{lignes}")
