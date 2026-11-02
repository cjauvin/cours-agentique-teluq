---
title: "Semaine 1 — Prédire le mot suivant"
weight: 1
---

# Semaine 1 — Prédire le mot suivant

Un modèle de langue ne fait qu'une chose : étant donné un début de texte, il
attribue une probabilité à chaque jeton qui pourrait suivre. Tout le reste — la
conversation, le raisonnement, les appels d'outils de la semaine 7 — est
construit sur cette seule opération. Cette semaine, on la réalise avec le
modèle le plus simple qui soit : le **bigramme**.

## Le bigramme

Le bigramme ne regarde que le mot précédent. Il estime la probabilité du mot
suivant en comptant, dans un corpus, combien de fois chaque paire apparaît :

$$P(w_t \mid w_{t-1}) = \frac{C(w_{t-1}, w_t)}{C(w_{t-1})}$$

Pour générer du texte, on tire le mot suivant au hasard selon cette
distribution, puis on recommence. La **température** $T$ modifie la
distribution avant le tirage : chaque compte $c_i$ devient $c_i^{1/T}$, puis on
renormalise. C'est exactement le paramètre `temperature` des API de LLM.

Choisis un mot, puis fais varier la température : observe comment la
distribution s'aplatit ou se concentre, et ce que ça change aux phrases
générées.

{{< jouet nom="s01_bigramme" hauteur="560" titre="Jouet : bigramme et température" >}}

## Exercice

Complète la fonction `probabilites` ci-dessous. Les vérifications se mettent à
jour dès que tu exécutes la cellule.

{{< jouet nom="s01_exercice" hauteur="520" titre="Exercice : la température" >}}

## Limites

- La phrase « le chat dort sur le canapé du chien . » peut être générée, même
  si elle n'apparaît nulle part dans le corpus. Pourquoi?
- Que faudrait-il pour que le modèle accorde un adjectif avec un nom situé trois
  mots plus tôt?
