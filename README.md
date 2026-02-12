# ChefBot — Guide du TP

Ce dépôt contient l'implémentation progressive de *ChefBot*, l'agent cuisinier du TP. Ce README explique, pour chaque question de `ENONCE_TP.md`, ce qui est attendu et où trouver l'implémentation ou un exemple dans le dépôt.

**Important — Prérequis**
- Créez un fichier `.env` à la racine avec les variables indiquées dans `ENONCE_TP.md` (clés Groq et Langfuse).
- Installez les dépendances requises (par exemple `groq`, `langfuse`, `python-dotenv`, `smolagents` selon vos besoins).

Exemples d'exécution rapide :
```powershell
python chefbot.py
python Partie4-6.py
python Partie5
python Partie7
```
Le script `chefbot.py` contient des points d'entrée pour exécuter des exemples et l'expérience d'évaluation.

**Plan du README**

- **Partie 1 — Premier contact (1.1, 1.2, 1.3)**
	- Objectif : appel LLM simple, system prompt, et traçage Langfuse.
	- Où : implémentation principale dans [chefbot.py](chefbot.py). Voir la fonction `ask_chef()`.

- **Partie 2 — Le Chef qui réfléchit (2.1, 2.2)**
	- Objectif : planificateur multi-étapes (plan → exécution par étape → synthèse) et gestion d'erreurs / retry.
	- Où : `plan_weekly_menu()` est implémentée dans [chefbot.py](chefbot.py). La logique est divisée en `_plan`, `_execute_step`, `_synthesize` et chaque étape est tracée avec `@observe()`.

- **Partie 3 — Évaluation et qualité (3.1–3.4)**
	- Objectif : créer un dataset d'évaluation, écrire un évaluateur programmatique et un juge LLM, exécuter l'expérience.
    - Où : implémentation principale dans [chefbot.py](chefbot.py). Voir les fonctions `create_chefbot_dataset`, `rule_evaluator`, `llm_judge` et `run_evaluation`.

- **Partie 4 — Tool use et smolagents (4.1–4.3)**
	- Objectif : définir des outils simulés et implémenter une boucle manuelle de tool-calling, puis migrer vers `smolagents`.
    - Où : implémentation principale dans [Partie4-6.py](Partie4-6.py). Voir le début du fichier qui est dédié à la Partie 4.

- **Partie 5 — Le Restaurant intelligent (5.1–5.3)**
	- Objectif : `MenuDatabaseTool` (classe `Tool`), agent planificateur et mode conversationnel.
	- Où : consultez [Partie5.py](Partie5.py) pour l'outil de base de données et les exemples d'agent.

- **Partie 6 — Architecture multi-agent (6.1–6.2)**
	- Objectif : manager + 3 agents spécialisés (nutritionist, chef_agent, budget_agent) et test complexe.
    - Où : implémentation principale dans [Partie4-6.py](Partie4-6.py). Voir la fin du fichier qui est dédié à la Partie 6.

- **Partie 7 — Boss final : évaluation end-to-end (7.1–7.3)**
	- Objectif : dataset de scenarios, juge LLM 5 critères, comparaison de configurations.
    - Où : consultez [Partie7.py](Partie7.py) pour le boss final.
