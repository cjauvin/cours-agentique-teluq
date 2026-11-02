# C1 — IA agentique : fondements, conception et développement

Cours de la TÉLUQ (programme court en développement logiciel assisté par l'IA).

## Structure

| Dossier | Contenu |
|---|---|
| `index.qmd`, `mise-en-route.qmd`, `modules/` | le site du cours (livre Quarto, une page par semaine) |
| `labos/` | les laboratoires (notebooks marimo), à exécuter dans un vrai Python |
| `c1/` | la bibliothèque commune des laboratoires (`verif`, `llm`, `viz`) |
| `_extensions/` | l'extension [quarto-marimo](https://github.com/marimo-team/quarto-marimo) |
| `documents/` | dossiers de présentation et avant-projet |
| `essais/` | l'essai comparatif Hugo / Quarto (novembre 2026) |
| `archives/` | l'ancien squelette Quarto |

## Commandes

```bash
quarto preview                          # le site, avec rechargement
uv run marimo edit labos/s01_temperature.py   # un laboratoire
```

Ne pas lancer `quarto render` pendant qu'un `quarto preview` tourne : l'aperçu
peut réécrire une version périmée d'une page.
