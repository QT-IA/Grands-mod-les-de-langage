from dotenv import load_dotenv
from datetime import datetime
from groq import Groq
from langfuse import observe, get_client, propagate_attributes, Evaluation
from typing import List, Dict, Any
import json

load_dotenv()

# Initialise clients
groq_client = Groq()

client = get_client()

modele = "meta-llama/llama-4-scout-17b-16e-instruct"

# Partie 1 :
@observe(name="Quentin & Arthur")
def ask_chef(question: str, saison : str, temperature: float = 0.5) -> str:

    # Ajout de metadata et tags pour Langfuse
    try:
        with propagate_attributes(tags=["Quentin & Arthur", "Partie 1"]):
            client.update_current_trace(
                metadata={
                    "type": "ask_chef",
                    "season": saison,
                    "temperature": temperature,
                }
            )
    except Exception:
        # Ne pas échouer si Langfuse n'est pas configuré
        pass

    system_prompt = (
        "Tu es ChefBot, un chef cuisinier français spécialisé en cuisine de saison. "
        "Réponds en français, propose des ingrédients de saison, techniques de cuisson, "
        "variantes et suggestions de présentation. Sois clair, pratique et concis."
    )

    response = groq_client.chat.completions.create(
        model=modele,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
    )

    # Extraire le contenu de la première sélection
    content = response.choices[0].message.content

    # Flush Langfuse (best-effort)
    try:
        client.flush()
    except Exception:
        pass

    return str(content)

# Partie 2 :
@observe(name="Quentin & Arthur")
def plan_weekly_menu(constraints: str) -> Dict[str, Any]:
    # Ajout de metadata et tags pour Langfuse - Trace principale
    try:
        with propagate_attributes(tags=["Quentin & Arthur", "Partie 2"]):
            client.update_current_trace(
                metadata={
                    "type": "planifier le menu hebdomadaire",
                    "partie": "Partie 2",
                    "constraints": constraints,
                }
            )
    except Exception:
        pass

    def _log_langfuse_error(message: str) -> None:
        try:
            client.log(message=message, level="ERROR")
        except Exception:
            try:
                client.update_current_trace(
                    metadata={
                        "error": message, 
                        "level": "ERROR"
                        }
                    )
            except Exception:
                pass

    @observe(name="Quentin & Arthur, Partie 2 - plan")
    def _plan(constraints: str) -> List[Dict[str, Any]]:
        """Étape 1: Planification"""
        try:
            client.update_current_observation(
                metadata={
                    "step": "planning", 
                    "constraints": constraints
                    }
            )
        except Exception:
            pass

        prompt = (
            "Décompose la tâche de création d'un menu hebdomadaire en étapes claires. "
            "Renvoie strictement un JSON de la forme [ {\"step\": 1, \"title\": \"...\", \"instruction\": \"...\"}, ... ] et rien d'autre. "
            "Prends en compte ces contraintes: " + constraints
        )

        contexte = (
            "Contexte: Tu es ChefBot, un assistant expert en planification de menus. "
            "Tu dois créer un plan d'action étape par étape pour générer un menu hebdomadaire en respectant les contraintes données."
        )

        for attempt in range(2):
            try:
                resp = groq_client.chat.completions.create(
                    model=modele,
                    messages=[
                        {"role": "system", "content": contexte},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                text = resp.choices[0].message.content
                try:
                    plan = json.loads(text)
                except json.JSONDecodeError:
                    # Fallback: try to extract JSON list
                    start = text.index('[')
                    end = text.rindex(']') + 1
                    plan = json.loads(text[start:end])
                
                if isinstance(plan, list):
                    return plan
                else:
                    raise ValueError("Plan JSON is not a list")
            except Exception as e:
                _log_langfuse_error(f"Plan parsing error (attempt {attempt+1}): {e}")
                if attempt == 0:
                    continue
                else:
                    raise

    @observe(name="Quentin & Arthur, Partie 2 - execution")
    def _execute_step(step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Étape 2: Exécution"""
        title = step.get("title", "Étape")
        instruction = step.get("instruction", "")
        
        try:
            client.update_current_observation(
                metadata={
                    "step": "execution",
                    "step_number": step.get("step", "?"),
                    "step_title": title
                }
            )
        except Exception:
            pass

        ctx_text = json.dumps(context, ensure_ascii=False)
        prompt = (
            f"Tu es ChefBot. Exécute l'étape '{title}': {instruction}.\n"
            f"Contexte: {ctx_text}\n"
            "Réponds en français et sois précis."
        )

        resp = groq_client.chat.completions.create(
            model=modele,
            messages=[
                {"role": "system", "content": "Tu es ChefBot, un chef cuisinier français expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )

        return resp.choices[0].message.content

    @observe(name="Quentin & Arthur, Partie 2 - synthese")
    def _synthesize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Étape 3: Synthèse"""
        try:
            client.update_current_observation(
                metadata={
                    "step": "synthese",
                    "num_results": len(results)
                }
            )
        except Exception:
            pass
            
        prompt = (
            "Tu es ChefBot. En te basant sur les résultats suivants, génère un menu hebdomadaire cohérent. "
            "RÉPONDS STRICTEMENT EN JSON VALIDE, et rien d'autre (aucune explication).\n"
            "Format attendu (exemple) :\n"
            "{\n"
            "  \"week_menu\": {\n"
            "    \"lundi\": {\"dejeuner\": \"...\", \"diner\": \"...\"},\n"
            "    \"mardi\": {\"dejeuner\": \"...\", \"diner\": \"...\"},\n"
            "    ...\n"
            "  }\n"
            "}\n"
            "Chaque repas doit être une courte description listant les plats et ingrédients principaux.\n"
            "Respecte les contraintes fournies et inclus des champs optionnels comme 'calories_estimees' ou 'notes_pratiques' si utile.\n"
            "Résultats: " + json.dumps(results, ensure_ascii=False)
        )

        for attempt in range(2):
            try:
                resp = groq_client.chat.completions.create(
                    model=modele,
                    messages=[
                        {"role": "system", "content": "Tu es ChefBot, synthétiseur de menus. Il faut créer un menu hebdomadaire à partir des résultats d'exécution."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                )
                text = resp.choices[0].message.content
                
                # Tenter de parser le JSON directement
                try:
                    menu = json.loads(text)
                    return menu
                except json.JSONDecodeError:
                    # Si échec, essayer d'extraire le JSON (par exemple s'il est entouré de ```json ... ```)
                    try:
                        start = text.index('{')
                        end = text.rindex('}') + 1
                        json_str = text[start:end]
                        menu = json.loads(json_str)
                        return menu
                    except (ValueError, json.JSONDecodeError):
                        _log_langfuse_error(f"Failed to extract JSON from response: {text}")
                        raise ValueError("Synthesis response is not valid JSON")

            except Exception as e:
                _log_langfuse_error(f"Synthesis JSON parsing error (attempt {attempt+1}): {e}")
                if attempt == 0:
                    continue
                else:
                    raise
    try:
        plan = _plan(constraints)
        # Mettre à jour la trace avec le nombre d'étapes planifiées
        try:
            client.update_current_trace(
                metadata={
                    "type": "planifier le menu hebdomadaire",
                    "partie": "Partie 2",
                    "constraints": constraints,
                    "num_steps_planned": len(plan) if isinstance(plan, list) else 0,
                    "plan_steps": [s.get("title", "?") for s in plan] if isinstance(plan, list) else []
                }
            )
        except Exception:
            pass
    except Exception as e:
        _log_langfuse_error(f"Failed to produce plan: {e}")
        raise

    results: List[Dict[str, Any]] = []
    context: Dict[str, Any] = {"constraints": constraints}
    for step in plan:
        try:
            out = _execute_step(step, context)
        except Exception as e:
            _log_langfuse_error(f"Error executing step {step}: {e}")
            out = f"Error: {e}"
        results.append({"step": step, "output": out})
        context[f"step_{step.get('step', len(results))}"] = out

    try:
        menu = _synthesize(results)
        # Mettre à jour la trace avec le succès de la synthèse
        try:
            client.update_current_trace(
                metadata={
                    "type": "planifier le menu hebdomadaire",
                    "partie": "Partie 2",
                    "constraints": constraints,
                    "num_steps_planned": len(plan) if isinstance(plan, list) else 0,
                    "num_steps_executed": len(results),
                    "status": "completed",
                    "plan_steps": [s.get("title", "?") for s in plan] if isinstance(plan, list) else [],
                    "has_menu": "week_menu" in menu if isinstance(menu, dict) else False
                }
            )
        except Exception:
            pass
    except Exception as e:
        _log_langfuse_error(f"Failed to synthesize menu: {e}")
        raise

    try:
        client.flush()
    except Exception:
        pass

    return menu

# Partie 3 :
@observe(name="Quentin & Arthur - Dataset")
def create_chefbot_dataset():
    name = "chefbot-menu-eval_quentin-arthur"
    
    test_cases = [
        {
            "input": {"constraints": "repas pour diabetique, sans sucre ajouté, portion individuelle"},
            "expected_output": {"must_avoid": ["sucre", "miel", "sirop"], "must_include": ["légumes", "protéine maigre"], "max_calories_per_meal": 600},
            "metadata": {"case": "diabete"},
        },
        {
            "input": {"constraints": "menu économique pour 4 personnes, budget 10 euros par personne"},
            "expected_output": {"must_avoid": ["produits premium"], "must_include": ["légumineuses", "légumes de saison"], "max_price_per_person": 10},
            "metadata": {"case": "budget"},
        },
        {
            "input": {"constraints": "allergie aux noix et aux crustacés, 2 adultes 2 enfants"},
            "expected_output": {"must_avoid": ["noix", "crevettes", "crustacés"], "must_include": ["légumes", "féculents"], "servings": 4},
            "metadata": {"case": "allergies"},
        },
        {
            "input": {"constraints": "régime végétarien pour une semaine, 1 personne, riche en protéines"},
            "expected_output": {"must_avoid": ["viande", "poisson"], "must_include": ["tofu", "légumineuses", "légumes"], "protein_per_day_g": 60},
            "metadata": {"case": "vegetarien"},
        },
        {
            "input": {"constraints": "préférences culturelles: cuisine méditerranéenne, saison: été, 2 personnes"},
            "expected_output": {"must_avoid": [], "must_include": ["huile d'olive", "légumes frais", "poisson"], "style": "mediterranean"},
            "metadata": {"case": "mediterranean"},
        },
    ]

    try:
        with propagate_attributes(tags=["Quentin & Arthur", "Partie 3"]):
            client.create_dataset(
                name=name,
                description="Dataset d'évaluation pour ChefBot"
            )
    except Exception:
        pass

    dataset = client.get_dataset(name)

    # Check if dataset already has the correct number of items
    if len(dataset.items) == len(test_cases):
        return dataset

    for case in test_cases:
        try:
            client.create_dataset_item(
                dataset_name=name,
                input=case["input"],
                expected_output=case["expected_output"],
                metadata=case.get("metadata", {}),
            )
        except Exception:
            pass
    
    ds = client.get_dataset(name)
    print(f"Dataset '{name}' contains {len(ds.items)} items.")
    return ds

@observe(name="Quentin & Arthur - Evaluator")
def rule_evaluator(output: str, expected: Dict) -> Dict:
    text = output.lower()
    scores = {}

    avoid = expected.get("must_avoid", [])
    if avoid:
        avoid_violations = [a for a in avoid if a.lower() in text]
        scores["avoid_score"] = 0.0 if avoid_violations else 1.0
        scores["avoid_violations"] = avoid_violations
    else:
        scores["avoid_score"] = 1.0

    include = expected.get("must_include", [])
    if include:
        included = [i for i in include if i.lower() in text]
        scores["include_ratio"] = len(included) / len(include)
        scores["included_items"] = included
    else:
        scores["include_ratio"] = 1.0

    scores["overall"] = (scores["avoid_score"] + scores["include_ratio"]) / 2

    return scores

@observe(name="Quentin & Arthur - LLM Judge")
def llm_judge(question: str, output: str, expected: Dict) -> Dict:
    prompt = (
        "Tu es un évaluateur expert. Note la réponse fournie sur ces critères (0.0 à 1.0):\n"
        "- pertinence: la réponse respecte-t-elle les contraintes fournies?\n"
        "- creativite: les recettes sont-elles variées et originales?\n"
        "- praticite: les recettes sont-elles réalistes pour un non-professionnel?\n"
        "Réponds STRICTEMENT en JSON avec les clés: pertinence, creativite, praticite, comment.\n"
        f"Question/contrainte: {question}\n"
        f"Critères expectés: {json.dumps(expected, ensure_ascii=False)}\n"
        f"Sortie: {output}\n"
    )

    response = groq_client.chat.completions.create(
        model=modele,
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
        # fallback: try to extract JSON between braces
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            result = json.loads(text[start:end])
        except Exception as e:
            result = {"pertinence": 0.0, "creativite": 0.0, "praticite": 0.0, "comment": f"parse_error: {e}"}

    return result

@observe(name="Quentin & Arthur - Rule Evaluator")
def run_evaluation():
    dataset = create_chefbot_dataset()

    @observe(name="Quentin & Arthur - Task")
    def task(*, item):
        # item.input contains {'constraints': '...'}
        constraints = item.input.get("constraints")
        
        # Appel direct sans import circulaire ni re-decoration
        menu = plan_weekly_menu(constraints)
        
        # return as string for evaluators
        return json.dumps(menu, ensure_ascii=False)

    def evaluator_rule(**kwargs):
        output = kwargs.get("output")
        expected = kwargs.get("expected_output")
        scores = rule_evaluator(output, expected)
        return [
            Evaluation(name="avoid_score", value=scores.get("avoid_score")),
            Evaluation(name="include_ratio", value=scores.get("include_ratio")),
            Evaluation(name="overall_rule", value=scores.get("overall")),
        ]

    def evaluator_llm(**kwargs):
        # Sécurisation de la récupération par item ou input direct
        item = kwargs.get("item")
        if item and hasattr(item, "input"):
            question = item.input.get("constraints")
        else:
            inp = kwargs.get("input", {})
            question = inp.get("constraints") if isinstance(inp, dict) else str(inp)

        output = kwargs.get("output")
        expected = kwargs.get("expected_output")
        
        if not question:
            question = "No constraints provided"
            
        scores = llm_judge(question, output, expected)
        return [
            Evaluation(name="pertinence", value=scores.get("pertinence", 0.0)),
            Evaluation(name="creativite", value=scores.get("creativite", 0.0)),
            Evaluation(name="praticite", value=scores.get("praticite", 0.0)),
        ]

    # Run the experiment using the built-in runner
    results = client.run_experiment(
        name=f"chefbot-menu-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        data=dataset.items,
        task=task,
        evaluators=[evaluator_rule, evaluator_llm],
        description="Evaluation of ChefBot weekly menu planner",
        metadata={"model": modele},
    )

    try:
        client.flush()
    except Exception:
        pass
    return results

# Partie 4 :

# Test :
if __name__ == "__main__":

    saison = "printemps"
    example_q = f"Que proposerais-tu pour un dîner de {saison} avec des asperges et du saumon ?"
    temps = [0.1, 0.7, 1.2]


    # for temp in temps:
    #     print(ask_chef(example_q, saison, temperature=temp))


    # plan_weekly_menu(f"Le menu doit être végétarien, adapté pour une famille de 4 personnes, et utiliser des ingrédients de saison pour le {saison}. Inclure des options pour le déjeuner et le dîner, ainsi que des suggestions de desserts légers.")


    run_evaluation()