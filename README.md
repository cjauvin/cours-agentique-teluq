# C1 — IA agentique : fondements, conception et développement

Cours de la TÉLUQ (programme court en développement logiciel assisté par l'IA).

## Structure

| Dossier | Contenu |
|---|---|
| `index.qmd`, `mise-en-route.qmd`, `modules/` | le site du cours (livre Quarto, une page par semaine) |
| `labos/` | les laboratoires (notebooks marimo), à exécuter dans un vrai Python |
| `c1/` | la bibliothèque commune des laboratoires (`verif`, `llm`, `viz`) |
| `.devcontainer/` | l'environnement GitHub Codespaces des laboratoires |
| `_extensions/` | l'extension [quarto-marimo](https://github.com/marimo-team/quarto-marimo) |
| `documents/` | dossiers de présentation et avant-projet |
| `essais/` | l'essai comparatif Hugo / Quarto (novembre 2026) |
| `archives/` | l'ancien squelette Quarto |

## Commandes

```bash
quarto preview                          # le site, avec rechargement
uv run marimo edit labos/s01_temperature.py   # un laboratoire
uv run python outils/precalculer_s03.py       # données de la page 3
```

Ne pas lancer `quarto render` pendant qu'un `quarto preview` tourne : l'aperçu
peut réécrire une version périmée d'une page. Si une modification n'apparaît
pas, arrêter l'aperçu, faire `quarto render`, puis relancer l'aperçu.

## Code modifiable dans les pages

Tout le code affiché dans une page doit être modifiable. L'éditeur natif de
quarto-marimo ne réexécutant pas le code modifié, chaque page inclut
`_includes/code-modifiable.qmd` après sa cellule d'imports, et chaque bloc de
code affiché s'écrit en deux cellules :

````markdown
```python {.marimo}
code_decoder = code_modifiable(r'''
def decoder(compteur, temperature=1.0):
    ...
''')
code_decoder
```

```python {.marimo}
(decoder,) = executer(code_decoder, {}, "decoder")
```
````

Le deuxième argument de `executer` donne les noms dont le code a besoin; les
suivants, ceux qu'il doit exporter vers le reste de la page. Pas de
`#| echo: true`.
