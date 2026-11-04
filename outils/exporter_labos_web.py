"""Exporte les laboratoires sans PyTorch en notebooks marimo qui tournent dans le
navigateur (WebAssembly), dans <site>/labos/<nom>.html.

Le paquet agentique n'existe pas dans Pyodide : on l'ajoute à une copie du
notebook, dans une cellule de configuration (masquée) qui écrit ses fichiers
sur le système de fichiers du navigateur. Les fichiers de labos/ ne changent pas.

Lancé par Quarto après chaque rendu (post-render dans _quarto.yml), ou à la
main :

    uv run python outils/exporter_labos_web.py [dossier_du_site]
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
LABOS = ["s01_temperature", "s02_attention"]
MODULES = ["__init__.py", "verif.py", "viz.py"]


def variante_web(source):
    """Ajoute au notebook une cellule de configuration qui installe agentique."""
    fichiers = {nom: (RACINE / "agentique" / nom).read_text() for nom in MODULES}
    configuration = (
        "with app.setup(hide_code=True):\n"
        "    # Version navigateur : la bibliothèque agentique du cours, écrite\n"
        "    # dans le système de fichiers de Pyodide.\n"
        "    import pathlib as _pathlib\n"
        "    import sys as _sys\n"
        "\n"
        "    _dossier = _pathlib.Path('/home/pyodide/lib-agentique/agentique')\n"
        "    _dossier.mkdir(parents=True, exist_ok=True)\n"
        f"    for _nom, _texte in {fichiers!r}.items():\n"
        "        (_dossier / _nom).write_text(_texte)\n"
        "    _sys.path.insert(0, str(_dossier.parent))\n"
    )
    entete, reste = source.split("\n\n\n@app.cell", 1)
    return f"{entete}\n\n{configuration}\n\n@app.cell{reste}"


def main():
    site = Path(sys.argv[1] if len(sys.argv) > 1
                else os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", RACINE / "_site"))
    sortie = site / "labos"
    sortie.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        for nom in LABOS:
            copie = Path(temp) / f"{nom}.py"
            copie.write_text(variante_web((RACINE / "labos" / f"{nom}.py").read_text()))
            subprocess.run(
                [sys.executable, "-m", "marimo", "export", "html-wasm", str(copie),
                 "-o", str(sortie / f"{nom}.html"), "--mode", "edit", "--single-file", "-f"],
                check=True, capture_output=True,
            )
            print(f"labos/{nom}.html")


if __name__ == "__main__":
    main()
