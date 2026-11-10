"""Visualisations partagées : distributions et traces d'agent."""

import marimo as mo


def barres(probs, largeur=300):
    """Barres horizontales d'une distribution {jeton: probabilité}, de la plus probable à la moins probable."""
    lignes = []
    for jeton, p in sorted(probs.items(), key=lambda x: -x[1]):
        lignes.append(
            f"<div style='display:flex;align-items:center;gap:.5em;margin:2px 0'>"
            f"<code style='width:6em'>{jeton}</code>"
            f"<div style='background:#3b82f6;height:1em;width:{p * largeur:.0f}px'></div>"
            f"<span>{p:.0%}</span></div>"
        )
    return mo.Html("".join(lignes))


def afficher_message(message):
    role = message["role"]
    if role == "system":
        return mo.callout(mo.md(f"**système** — {message['content']}"), kind="neutral")
    if role == "user":
        return mo.callout(mo.md(f"**personne** — {message['content']}"), kind="info")
    if role == "tool":
        return mo.callout(
            mo.md(f"**outil** (résultat de `{message['tool_call_id']}`)\n\n"
                  f"```\n{message['content']}\n```"),
            kind="warn",
        )
    if message.get("tool_calls"):
        demandes = "\n".join(
            f"- `{a['function']['name']}({a['function']['arguments']})`"
            for a in message["tool_calls"]
        )
        return mo.callout(
            mo.md(f"**modèle** — demande d'outil :\n\n{demandes}"), kind="success"
        )
    return mo.callout(mo.md(f"**modèle** — {message['content']}"), kind="success")


def afficher_trace(messages):
    """Affiche une conversation d'agent, un encadré par message."""
    return mo.vstack([afficher_message(m) for m in messages])


def courbe_perte(pertes, plancher=None, haut=None, largeur=520, hauteur=200):
    """Courbe de perte en SVG (en bits), avec un plancher en pointillé rouge."""
    haut = haut or max(pertes)

    def px(i):
        return 40 + i / max(len(pertes) - 1, 1) * (largeur - 50)

    def py(v):
        return 10 + (1 - min(v, haut) / haut) * (hauteur - 40)

    trace = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(pertes))
    graduations = "".join(
        f"<text x='34' y='{py(v) + 4:.0f}' text-anchor='end' font-size='11'>{v}</text>"
        for v in range(0, int(haut) + 1)
    )
    ligne_plancher = "" if plancher is None else (
        f"<line x1='40' y1='{py(plancher):.1f}' x2='{largeur - 10}' y2='{py(plancher):.1f}' "
        f"stroke='#ef4444' stroke-dasharray='4 4'/>"
        f"<text x='{largeur - 12}' y='{py(plancher) - 5:.1f}' text-anchor='end' font-size='11' "
        f"fill='#ef4444'>plancher : {plancher:.2f} bits</text>"
    )
    return mo.Html(
        f"<svg width='{largeur}' height='{hauteur}' fill='currentColor' style='font-family:sans-serif'>"
        f"<line x1='40' y1='{py(0)}' x2='{largeur - 10}' y2='{py(0)}' stroke='#94a3b8'/>"
        f"<line x1='40' y1='10' x2='40' y2='{py(0)}' stroke='#94a3b8'/>{graduations}"
        f"<text x='46' y='14' font-size='11'>bits</text>{ligne_plancher}"
        f"<polyline points='{trace}' fill='none' stroke='#3b82f6' stroke-width='2'/>"
        f"<text x='{largeur / 2}' y='{hauteur - 5}' text-anchor='middle' font-size='11'>"
        f"pas d'entraînement</text></svg>"
    )
