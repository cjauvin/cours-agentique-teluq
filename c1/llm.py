"""Modèles interchangeables de signature llm(messages, schemas) -> message.

Les messages suivent le format de l'API Chat Completions d'OpenAI, que la
plupart des fournisseurs (Groq, OpenRouter, Ollama, vLLM…) acceptent.
"""

import json
import os


def appel_outil(id_appel, nom, **arguments):
    """Fabrique une réponse du modèle qui demande un outil."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": id_appel,
                "type": "function",
                "function": {"name": nom, "arguments": json.dumps(arguments)},
            }
        ],
    }


def reponse(texte):
    """Fabrique une réponse finale du modèle."""
    return {"role": "assistant", "content": texte}


def modele_simule(script):
    """Retourne un « modèle » qui joue les réponses du script, une par appel."""
    restantes = list(script)

    def llm(messages, schemas):
        return restantes.pop(0)

    return llm


def modele_openai(client, modele):
    """Adapte un client compatible OpenAI à la signature llm(messages, schemas)."""

    def llm(messages, schemas):
        reponse_api = client.chat.completions.create(
            model=modele, messages=messages, tools=schemas
        )
        message = reponse_api.choices[0].message
        resultat = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            resultat["tool_calls"] = [
                {
                    "id": appel.id,
                    "type": "function",
                    "function": {
                        "name": appel.function.name,
                        "arguments": appel.function.arguments,
                    },
                }
                for appel in message.tool_calls
            ]
        return resultat

    return llm


def configuration():
    """Lit C1_BASE_URL, C1_API_KEY et C1_MODELE dans l'environnement."""
    return {
        "base_url": os.environ.get("C1_BASE_URL"),
        "api_key": os.environ.get("C1_API_KEY"),
        "modele": os.environ.get("C1_MODELE"),
    }


def modele_reel(config=None):
    """Retourne un vrai modèle configuré par l'environnement, ou None s'il manque une variable."""
    config = config or configuration()
    if not all(config.values()):
        return None
    from openai import OpenAI

    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    return modele_openai(client, config["modele"])
