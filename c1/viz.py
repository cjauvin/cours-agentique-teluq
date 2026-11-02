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
