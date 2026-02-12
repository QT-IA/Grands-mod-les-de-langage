from dotenv import load_dotenv
from datetime import datetime
from groq import Groq
from langfuse import observe, get_client, propagate_attributes, Evaluation
from typing import List, Dict, Any
import json
import time

load_dotenv()

groq_client = Groq()
client = get_client()


@observe(name="Quentin & Arthur - Partie 7")
def create_multiagent_dataset():
    name = "chefbot-multiagent-eval_quentin-arthur"

    scenarios = [
        {
            "input": {"constraints": "Dîner simple pour 2 personnes : entrée, plat, dessert. Pas d'allergies."},
            "expected_output": {"must_respect": [], "expected_services": 3, "max_budget": 40},
            "metadata": {"difficulty": "facile"},
        },
        {
            "input": {"constraints": "Repas pour 4 personnes, 1 invité allergique aux noix. Entrée+plat+dessert."},
            "expected_output": {"must_respect": ["noix"], "expected_services": 3, "max_budget": 80},
            "metadata": {"difficulty": "moyen"},
        },
        {
            "input": {"constraints": "Dîner pour 6 personnes : 2 végétariens, 1 intolérant au gluten, budget limité."},
            "expected_output": {"must_respect": ["sans gluten", "options végétariennes"], "expected_services": 4, "max_budget": 120},
            "metadata": {"difficulty": "difficile"},
        },
        {
            "input": {"constraints": "Événement 12 personnes : contraintes culturelles (halal), allergies (fruits à coque), budget serré."},
            "expected_output": {"must_respect": ["halal", "pas de fruits à coque"], "expected_services": 4, "max_budget": 360},
            "metadata": {"difficulty": "extreme"},
        },
    ]

    try:
        with propagate_attributes(tags=["Quentin & Arthur", "Partie 7"]):
            client.create_dataset(name=name, description="Dataset multi-agent pour ChefBot - Partie 7")
    except Exception:
        pass

    ds = client.get_dataset(name)
    if len(ds.items) >= len(scenarios):
        return ds

    for s in scenarios:
        try:
            client.create_dataset_item(dataset_name=name, input=s["input"], expected_output=s["expected_output"], metadata=s.get("metadata", {}))
        except Exception:
            pass

    ds = client.get_dataset(name)
    return ds


@observe(name="Quentin & Arthur - Partie 7")
def llm_judge_multiagent(question: str, output: str, expected: Dict[str, Any], judge_model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> Dict[str, Any]:
    prompt = (
        "Tu es un évaluateur expert. Donne une note entre 0.0 et 1.0 pour chacun des critères suivants:\n"
        "1) respect_contraintes: les restrictions alimentaires sont-elles respectées?\n"
        "2) completude: tous les services demandés sont-ils proposés (entrée, plat, dessert, etc.)?\n"
        "3) budget: le budget maximum est-il respecté?\n"
        "4) coherence: le menu forme-t-il un ensemble harmonieux?\n"
        "5) faisabilite: les recettes sont-elles réalisables par un amateur?\n"
        "Réponds STRICTEMENT en JSON avec ces clés: respect_contraintes, completude, budget, coherence, faisabilite, comment.\n"
        f"Contrainte / question: {question}\n"
        f"Expected: {json.dumps(expected, ensure_ascii=False)}\n"
        f"Sortie du système (menu): {output}\n"
    )

    response = groq_client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": "Tu es un évaluateur objectif qui ne répond qu'en JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    text = response.choices[0].message.content
    try:
        result = json.loads(text)
    except Exception:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            result = json.loads(text[start:end])
        except Exception as e:
            result = {
                "respect_contraintes": 0.0,
                "completude": 0.0,
                "budget": 0.0,
                "coherence": 0.0,
                "faisabilite": 0.0,
                "comment": f"parse_error: {e}",
            }

    return result


def _safe_json_dumps(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


@observe(name="Quentin & Arthur - Partie 7")
def generate_menu_three_step(constraints: str, model_name: str) -> Dict[str, Any]:
    # Reprend le pattern du plan -> exécution -> synthèse en 3 appels
    try:
        with propagate_attributes(tags=["Quentin & Arthur", "Partie 7"], ):
            client.update_current_trace(metadata={"type": "generate_menu", "model": model_name, "constraints": constraints})
    except Exception:
        pass

    def _plan():
        prompt = (
            "Décompose la tâche de création d'un menu en 3 à 6 étapes claires (format JSON list of {step,title,instruction}).\n"
            f"Contrainte: {constraints}"
        )
        resp = groq_client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.2)
        text = resp.choices[0].message.content
        try:
            plan = json.loads(text)
        except Exception:
            try:
                start = text.index('[')
                end = text.rindex(']') + 1
                plan = json.loads(text[start:end])
            except Exception:
                plan = [{"step": 1, "title": "Générer menu", "instruction": "Proposer un menu complet"}]
        return plan

    def _execute_step(step, context_text):
        prompt = f"Exécute l'étape '{step.get('title')}'. Instruction: {step.get('instruction')}. Contexte: {context_text}"
        resp = groq_client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.5)
        return resp.choices[0].message.content

    def _synthesize(results):
        prompt = (
            "A partir des résultats ci-dessous, fournis un menu hebdomadaire ou un menu pour l'événement au format JSON valide.\n"
            "Format attendu: {\"services\": [\"entrée\", \"plat\", \"dessert\"], \"menu\": {...}}\n"
            "Réponds STRICTEMENT en JSON.\n"
            f"Résultats: {json.dumps(results, ensure_ascii=False)}"
        )
        resp = groq_client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature=0.3)
        text = resp.choices[0].message.content
        try:
            menu = json.loads(text)
        except Exception:
            try:
                start = text.index('{')
                end = text.rindex('}') + 1
                menu = json.loads(text[start:end])
            except Exception:
                menu = {"menu_text": text}
        return menu

    plan = _plan()
    results = []
    context_text = f"constraints: {constraints}"
    for step in plan:
        out = _execute_step(step, context_text)
        results.append({"step": step, "output": out})
        context_text += "\n" + out

    menu = _synthesize(results)

    try:
        client.flush()
    except Exception:
        pass

    return menu


@observe(name="Quentin & Arthur - Partie 7")
def run_partie7_comparison(models: List[str] = None):
    if models is None:
        models = [
            "meta-llama/llama-3.3-70b-versatile",
            "meta-llama/llama-4-scout-17b-16e-instruct",
        ]

    dataset = create_multiagent_dataset()

    comparison_results = {}

    for model_name in models:
        try:
            with propagate_attributes(tags=["Quentin & Arthur", "Partie 7"]):
                client.update_current_trace(metadata={"experiment_model": model_name})
        except Exception:
            pass

        def task(*, item):
            constraints = item.input.get("constraints")
            menu = generate_menu_three_step(constraints, model_name)
            return _safe_json_dumps(menu)

        def evaluator_llm(**kwargs):
            item = kwargs.get("item")
            question = item.input.get("constraints") if item and hasattr(item, "input") else kwargs.get("input", {}).get("constraints")
            output = kwargs.get("output")
            expected = kwargs.get("expected_output")
            scores = llm_judge_multiagent(question, output, expected)
            return [
                Evaluation(name="respect_contraintes", value=scores.get("respect_contraintes", 0.0)),
                Evaluation(name="completude", value=scores.get("completude", 0.0)),
                Evaluation(name="budget", value=scores.get("budget", 0.0)),
                Evaluation(name="coherence", value=scores.get("coherence", 0.0)),
                Evaluation(name="faisabilite", value=scores.get("faisabilite", 0.0)),
            ]

        exp_name = f"chefbot-multiagent-eval-{model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            results = client.run_experiment(name=exp_name, data=dataset.items, task=task, evaluators=[evaluator_llm], description=f"Partie 7 comparison for model {model_name}", metadata={"model": model_name})
        except Exception as e:
            results = None
            try:
                client.log(message=f"Experiment failed for {model_name}: {e}", level="ERROR")
            except Exception:
                pass

        # Aggregate simple averages
        agg = {"respect_contraintes": 0.0, "completude": 0.0, "budget": 0.0, "coherence": 0.0, "faisabilite": 0.0}
        count = 0
        if results and hasattr(results, "items"):
            for it in results.items:
                # each item has evaluations list
                evs = getattr(it, "evaluations", [])
                for ev in evs:
                    name = ev.name
                    val = ev.value if ev.value is not None else 0.0
                    if name in agg:
                        agg[name] += float(val)
                count += 1
            if count > 0:
                for k in agg:
                    agg[k] = round(agg[k] / count, 4)

        comparison_results[model_name] = {"aggregates": agg, "raw_results": results}

        # Log summary to Langfuse trace
        try:
            client.update_current_trace(metadata={"comparison_summary": agg, "model": model_name})
            client.log(message=f"Partie 7 - résumé pour {model_name}: {agg}", level="INFO")
            client.flush()
        except Exception:
            pass

    # Simple analysis between first two models
    analysis = ""
    if len(models) >= 2:
        m0, m1 = models[0], models[1]
        a0 = comparison_results[m0]["aggregates"]
        a1 = comparison_results[m1]["aggregates"]
        wins = {m0: 0, m1: 0}
        for crit in ["respect_contraintes", "completude", "budget", "coherence", "faisabilite"]:
            if a0[crit] > a1[crit]:
                wins[m0] += 1
            elif a1[crit] > a0[crit]:
                wins[m1] += 1

        analysis = f"Comparaison {m0} vs {m1}: scores agrégés {m0}={a0}, {m1}={a1}. Victoires par critère: {wins}."
        try:
            client.update_current_trace(metadata={"partie7_analysis": analysis})
            client.log(message=analysis, level="INFO")
            client.flush()
        except Exception:
            pass

    return comparison_results


if __name__ == "__main__":
    # Exécute la partie 7 : création dataset + comparaison
    ds = create_multiagent_dataset()
    models_to_compare = [
        "meta-llama/llama-3.3-70b-versatile",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    ]
    results = run_partie7_comparison(models=models_to_compare)
    print("Comparison finished. Summary:")
    for m, r in results.items():
        print(m, r.get("aggregates"))
