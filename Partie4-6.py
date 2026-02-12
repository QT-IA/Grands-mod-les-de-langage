
from dotenv import load_dotenv
from groq import Groq
from langfuse import observe, get_client, propagate_attributes
import json
from typing import List, Dict, Any, Callable
from smolagents import CodeAgent, LiteLLMModel, tool, Tool
from Partie5 import MenuDatabaseTool, calculate

load_dotenv()

groq_client = Groq()
langfuse_client = get_client()
modele = "openai/gpt-oss-120b"

# PARTIE 4 - TOOL CALLING MANUEL VS SMOLAGENTS

# 4.1 - Définitions des 3 outils

def check_fridge() -> List[str]:
    """Retourne la liste des ingrédients disponibles dans le frigo"""
    return [
        "oeufs (6)",
        "lait (1L)",
        "beurre (200g)",
        "fromage râpé (150g)",
        "tomates (4)",
        "courgettes (2)",
        "carottes (5)",
        "oignons (3)",
        "ail (1 tête)",
        "poulet (600g)",
        "crème fraîche (200ml)",
        "champignons (250g)",
        "épinards frais (300g)"
    ]

def get_recipe(dish_name: str) -> str:
    """Retourne une recette détaillée pour un plat donné"""
    recipes = {
        "omelette": """
        Omelette aux champignons et fromage
        
        Ingrédients:
        - 3 œufs
        - 100g de champignons
        - 50g de fromage râpé
        - 20g de beurre
        - Sel, poivre
        
        Instructions:
        1. Émincer les champignons et les faire revenir dans du beurre
        2. Battre les œufs avec sel et poivre
        3. Verser les œufs dans la poêle avec les champignons
        4. Parsemer de fromage râpé
        5. Cuire 3-4 minutes, plier et servir
        """,
        "poulet": """
        Poulet à la crème et champignons
        
        Ingrédients:
        - 600g de poulet
        - 200g de champignons
        - 200ml de crème fraîche
        - 1 oignon
        - 2 gousses d'ail
        - 30g de beurre
        - Sel, poivre, herbes de Provence
        
        Instructions:
        1. Couper le poulet en morceaux
        2. Faire revenir l'oignon et l'ail dans le beurre
        3. Ajouter le poulet et faire dorer
        4. Ajouter les champignons émincés
        5. Verser la crème, assaisonner
        6. Mijoter 20 minutes
        """,
        "gratin": """
        Gratin de courgettes
        
        Ingrédients:
        - 2 courgettes
        - 200ml de crème fraîche
        - 100g de fromage râpé
        - 2 gousses d'ail
        - Sel, poivre, muscade
        
        Instructions:
        1. Couper les courgettes en rondelles
        2. Disposer dans un plat à gratin
        3. Mélanger crème, ail haché, sel, poivre, muscade
        4. Verser sur les courgettes
        5. Parsemer de fromage
        6. Cuire 30 min à 180°C
        """
    }
    
    dish_lower = dish_name.lower()
    for key in recipes:
        if key in dish_lower:
            return recipes[key]
    
    return f"Désolé, je n'ai pas de recette pour '{dish_name}' dans ma base de données."

def check_dietary_info(ingredient: str) -> str:
    """Retourne les informations nutritionnelles et allergéniques d'un ingrédient"""
    dietary_info = {
        "oeufs": {
            "calories_pour_100g": 155,
            "proteines": "13g",
            "lipides": "11g",
            "glucides": "1g",
            "allergenes": ["œufs"],
            "convient_pour": ["régime protéiné"],
            "ne_convient_pas_pour": ["végan", "végétalien"]
        },
        "lait": {
            "calories_pour_100ml": 61,
            "proteines": "3.2g",
            "lipides": "3.3g",
            "glucides": "4.8g",
            "allergenes": ["lactose", "protéines laitières"],
            "convient_pour": ["végétarien"],
            "ne_convient_pas_pour": ["végan", "intolérant au lactose"]
        },
        "poulet": {
            "calories_pour_100g": 165,
            "proteines": "31g",
            "lipides": "3.6g",
            "glucides": "0g",
            "allergenes": [],
            "convient_pour": ["régime protéiné", "sans gluten"],
            "ne_convient_pas_pour": ["végan", "végétarien"]
        },
        "champignons": {
            "calories_pour_100g": 22,
            "proteines": "3.1g",
            "lipides": "0.3g",
            "glucides": "3.3g",
            "allergenes": [],
            "convient_pour": ["végan", "végétarien", "sans gluten", "régime faible en calories"],
            "ne_convient_pas_pour": []
        },
        "fromage": {
            "calories_pour_100g": 402,
            "proteines": "25g",
            "lipides": "33g",
            "glucides": "1.3g",
            "allergenes": ["lactose", "protéines laitières"],
            "convient_pour": ["végétarien"],
            "ne_convient_pas_pour": ["végan", "intolérant au lactose"]
        },
        "courgettes": {
            "calories_pour_100g": 17,
            "proteines": "1.2g",
            "lipides": "0.3g",
            "glucides": "3.1g",
            "allergenes": [],
            "convient_pour": ["végan", "végétarien", "sans gluten", "régime faible en calories"],
            "ne_convient_pas_pour": []
        }
    }
    
    ingredient_lower = ingredient.lower()
    for key in dietary_info:
        if key in ingredient_lower:
            info = dietary_info[key]
            result = f"\nInformations nutritionnelles pour {ingredient}:\n"
            result += f"- Calories: {info['calories_pour_100g']} kcal/100g\n"
            result += f"- Protéines: {info['proteines']}\n"
            result += f"- Lipides: {info['lipides']}\n"
            result += f"- Glucides: {info['glucides']}\n"
            result += f"- Allergènes: {', '.join(info['allergenes']) if info['allergenes'] else 'Aucun'}\n"
            result += f"- Convient pour: {', '.join(info['convient_pour'])}\n"
            result += f"- Ne convient pas pour: {', '.join(info['ne_convient_pas_pour']) if info['ne_convient_pas_pour'] else 'Tout le monde'}\n"
            return result
    
    return f"Pas d'informations disponibles pour '{ingredient}'"


# 4.2 - Boucle de tool calling manuelle

tools = [
    {
        "type": "function",
        "function": {
            "name": "check_fridge",
            "description": "Consulte le contenu du frigo et retourne la liste des ingrédients disponibles",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recipe",
            "description": "Retourne une recette détaillée pour un plat spécifique",
            "parameters": {
                "type": "object",
                "properties": {
                    "dish_name": {
                        "type": "string",
                        "description": "Le nom du plat pour lequel obtenir la recette"
                    }
                },
                "required": ["dish_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_dietary_info",
            "description": "Retourne les informations nutritionnelles et allergéniques d'un ingrédient",
            "parameters": {
                "type": "object",
                "properties": {
                    "ingredient": {
                        "type": "string",
                        "description": "Le nom de l'ingrédient à analyser"
                    }
                },
                "required": ["ingredient"]
            }
        }
    }
]

# Mapping des noms d'outils vers les fonctions
TOOLS_MAP: Dict[str, Callable] = {
    "check_fridge": check_fridge,
    "get_recipe": get_recipe,
    "check_dietary_info": check_dietary_info
}


@observe(name="execute_tool_call")
def execute_tool_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Exécute un appel d'outil et retourne le résultat"""
    try:
        langfuse_client.update_current_observation(
            metadata={"tool_name": str(tool_name), "arguments": arguments}
        )
    except Exception:
        pass
    
    if tool_name not in TOOLS_MAP:
        return f"Erreur: outil '{tool_name}' inconnu"
    
    try:
        function_to_call = TOOLS_MAP[tool_name]
        result = function_to_call(**arguments)
        if isinstance(result, list):
            return json.dumps(result, ensure_ascii=False)
        return str(result)
    except Exception as e:
        return f"Erreur lors de l'exécution de {tool_name}: {str(e)}"


@observe(name="Approche Manuelle")
def manual_tool_calling(question: str, max_iterations: int = 5) -> str:
    # Ajout de metadata pour le span
    try:
        langfuse_client.update_current_observation(
            metadata={
                "approach": "manual",
                "max_iterations": max_iterations
            }
        )
    except Exception:
        pass
    
    messages = [
        {
            "role": "system",
            "content": (
                "Tu es ChefBot, un assistant culinaire intelligent. "
                "Tu as accès à des outils pour consulter le frigo, obtenir des recettes, "
                "et vérifier les informations nutritionnelles. Utilise-les judicieusement."
            )
        },
        {"role": "user", "content": question}
    ]
    print("\nApproche Manuelle")
    print(f"Question: {question}\n")
    
    for iteration in range(max_iterations):
        print(f"--- Itération {iteration + 1}/{max_iterations} ---")
        
        # Appel au LLM avec les outils disponibles
        response = groq_client.chat.completions.create(
            model=modele,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.3
        )
        
        response_message = response.choices[0].message
        
        # Vérifier si le LLM veut utiliser un outil
        if response_message.tool_calls:
            # Ajouter la réponse du LLM aux messages
            messages.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            })
            
            # Exécuter chaque outil demandé
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                print(f"-- Appel de l'outil: {tool_name}")
                print(f"-- Arguments: {tool_args}")
                
                # Exécuter l'outil
                tool_result = execute_tool_call(tool_name, tool_args)
                print(f"-- Résultat: {tool_result[:100]}...")
                
                # Ajouter le résultat aux messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })
        else:
            final_response = response_message.content
            print(f"\nRéponse: {final_response}\n")
            
            # Mise à jour du span avec le succès
            try:
                langfuse_client.update_current_observation(
                    metadata={
                        "approach": "manual",
                        "max_iterations": max_iterations,
                        "iterations_used": iteration + 1,
                        "status": "completed"
                    }
                )
            except Exception:
                pass
            
            return final_response
    
    print(f"\nLimite d'itérations atteinte ({max_iterations})")
    
    # Mise à jour du span en cas de limite atteinte
    try:
        langfuse_client.update_current_observation(
            metadata={
                "approach": "manual",
                "max_iterations": max_iterations,
                "iterations_used": max_iterations,
                "status": "max_iterations_reached"
            }
        )
    except Exception:
        pass
    
    return "Désolé, je n'ai pas pu compléter la requête dans le nombre d'itérations autorisé."


# 4.3 - Version avec smolagents
@tool
def check_fridge_tool() -> List[str]:
    """Consulte le contenu du frigo et retourne la liste des ingrédients disponibles"""
    return check_fridge()

@tool
def get_recipe_tool(dish_name: str) -> str:
    """
    Retourne une recette détaillée pour un plat spécifique
    
    Args:
        dish_name: Le nom du plat pour lequel obtenir la recette
    """
    return get_recipe(dish_name)

@tool
def check_dietary_info_tool(ingredient: str) -> str:
    """
    Retourne les informations nutritionnelles et allergéniques d'un ingrédient
    
    Args:
        ingredient: Le nom de l'ingrédient à analyser
    """
    return check_dietary_info(ingredient)


@observe(name="Smolagents")
def smolagents_approach(question: str) -> str:
    try:
        langfuse_client.update_current_observation(
            metadata={"approach": "smolagents"}
        )
    except Exception:
        pass
    
    print("Smolagents")
    print(f"Question: {question}\n")
    
    # Créer le modèle LiteLLM pointant vers Groq
    model = LiteLLMModel(
        model_id=f"groq/{modele}",
        temperature=0.3
    )
    
    # Créer l'agent avec les outils
    agent = CodeAgent(
        tools=[check_fridge_tool, get_recipe_tool, check_dietary_info_tool],
        model=model,
        additional_authorized_imports=["json", "typing"],
        max_steps=5
    )
    
    # Exécuter la requête
    try:
        result = agent.run(question)
        print(f"\nRéponse:\n{result}\n")
        
        # Mise à jour du span avec le succès
        try:
            langfuse_client.update_current_observation(
                metadata={
                    "approach": "smolagents",
                    "status": "completed"
                }
            )
        except Exception:
            pass
        
        return result
    except Exception as e:
        error_msg = f"Erreur lors de l'exécution de l'agent: {str(e)}"
        print(f"\n{error_msg}\n")
        
        # Mise à jour du span en cas d'erreur
        try:
            langfuse_client.update_current_observation(
                metadata={
                    "approach": "smolagents",
                    "status": "error",
                    "error": str(e)
                }
            )
        except Exception:
            pass
        
        return error_msg


# Tests et comparaison

@observe(name="Quentin & Arthur - Partie 4 Comparaison")
def compare_approaches():
    """Compare les deux approches"""
    with propagate_attributes(tags=["Quentin & Arthur", "Partie 4"]):
        # Ajout de metadata et tags pour la trace principale
        try:
            langfuse_client.update_current_trace(
                metadata={
                    "type": "comparaison_tool_calling",
                    "partie": "Partie 4",
                    "approaches": ["manual", "smolagents"]
                }
            )
        except Exception:
            pass
        
        test_question = (
            "J'ai faim. Qu'est-ce que je peux cuisiner avec ce que j'ai dans mon frigo ? "
            "Propose-moi une recette et dis-moi si elle convient pour un végétarien."
        )
        
        print("\n")
        print("COMPARAISON DES DEUX APPROCHES")
        
        # Approche manuelle
        print("\n[1/2] Test de l'approche manuelle...")
        manual_result = manual_tool_calling(test_question)
        
        # Approche smolagents
        print("\n[2/2] Test de l'approche smolagents...")
        smolagents_result = smolagents_approach(test_question)

        # l'approche manuelle permet d'avoir le controle et une transparence totale sur ce qui se passe, 
        # mais c'est vite une galère dès qu'il faut gérer les erreurs ou les appels d'outils à la main dans 150 lignes de code. 

        # Smolagents permet code super court qui s'occupe de tout automatiquement, 
        # même si tu perds un peu en contrôle. 

        # Au final, smolagents est plus pratique et rapide à faire et la méthode manuelle est plutot
        # pour les projets tordus qui demandent un contrôle total sur chaque étape.



# PARTIE 6 - L'EMPIRE CHEFBOT

# Outils importés depuis Partie5.py (Partie 5): MenuDatabaseTool et calculate

# 6.1 - Construction du système multi-agent
def build_chefbot_empire():
    """
    Construit le système multi-agent ChefBot avec un manager et 3 agents spécialisés.
    
    Returns:
        Le manager agent qui coordonne les agents spécialisés
    """
    
    # Créer le modèle LLM
    model = LiteLLMModel(model_id=f"groq/{modele}", temperature=0.3)
    
    # --- Agent 1: Nutritionist ---
    # Vérifie l'équilibre nutritionnel et les allergènes
    nutritionist = CodeAgent(
        tools=[check_dietary_info_tool],
        model=model,
        name="nutritionist",
        description=(
            "Nutritionniste expert qui vérifie l'équilibre nutritionnel et les allergènes. "
            "Utilise check_dietary_info_tool pour analyser les ingrédients. "
            "Peut recommander des alternatives pour les allergies et intolérances."
        ),
        max_steps=5,
    )
    
    # --- Agent 2: Chef Agent ---
    # Propose des recettes et consulte le frigo
    chef_agent = CodeAgent(
        tools=[check_fridge_tool, get_recipe_tool],
        model=model,
        name="chef_agent",
        description=(
            "Chef cuisinier expert qui propose des recettes et consulte le frigo. "
            "Utilise check_fridge_tool pour voir les ingrédients disponibles et "
            "get_recipe_tool pour obtenir des recettes détaillées."
        ),
        max_steps=5,
    )
    
    # --- Agent 3: Budget Agent ---
    # Calcule les coûts et respecte le budget
    menu_db = MenuDatabaseTool()
    budget_agent = CodeAgent(
        tools=[calculate, menu_db],
        model=model,
        name="budget_agent",
        description=(
            "Expert en gestion de budget qui calcule les coûts et consulte le menu. "
            "Utilise menu_database pour trouver des plats selon le budget et "
            "calculate pour effectuer les calculs de coûts totaux."
        ),
        max_steps=5,
    )
    
    # --- Manager ---
    # Coordonne les 3 agents spécialisés, n'a aucun outil propre
    manager = CodeAgent(
        tools=[],  # Pas d'outils propres
        model=model,
        managed_agents=[nutritionist, chef_agent, budget_agent],
        name="chefbot_manager",
        description=(
            "Tu es le manager de ChefBot, un système de restauration intelligent. "
            "Tu coordonnes 3 agents spécialisés:\n"
            "- nutritionist: pour vérifier l'équilibre nutritionnel et les allergènes\n"
            "- chef_agent: pour proposer des recettes et consulter le frigo\n"
            "- budget_agent: pour gérer le budget et trouver des plats dans le menu\n\n"
            "Délègue intelligemment les tâches aux agents appropriés. "
            "Pour une demande complexe, consulte plusieurs agents et synthétise leurs réponses."
        ),
        max_steps=10,
    )
    
    return manager


# 6.2 - Test du système multi-agent
@observe(name="Quentin & Arthur - Partie 6")
def test_empire_chefbot():
    try:
        with propagate_attributes(tags=["Quentin & Arthur", "Partie 6"]):
            langfuse_client.update_current_trace(
                metadata={
                    "type": "multi_agent_system",
                    "partie": "Partie 6",
                    "agents": ["nutritionist", "chef_agent", "budget_agent"],
                    "manager": "chefbot_manager"
                }
            )
    except Exception:
        pass
    
    manager = build_chefbot_empire()

    complex_query = (
        "Je reçois 8 personnes samedi soir. Parmi eux : 2 végétariens, "
        "1 intolérant au gluten, 1 allergique aux fruits à coque. "
        "Budget total : 120 euros. "
        "Je veux un apéritif, une entrée, un plat principal et un dessert. "
        "Il faut que tout le monde puisse manger chaque service."
    )
    
    print("Requête complexe :\n")
    print(f"\n{complex_query}\n")
    
    try:
        result = manager.run(complex_query)
        
        print("Résultat :")
        print(f"\n{result}\n")
        
        # Mise à jour de la trace avec succès
        try:
            langfuse_client.update_current_trace(
                metadata={
                    "type": "multi_agent_system",
                    "partie": "Partie 6",
                    "agents": ["nutritionist", "chef_agent", "budget_agent"],
                    "manager": "chefbot_manager",
                    "status": "completed",
                    "query_complexity": "high",
                    "constraints": ["végétariens", "sans gluten", "sans fruits à coque", "budget 120€"]
                }
            )
        except Exception:
            pass
        
        return result
        
    except Exception as e:
        error_msg = f"Erreur lors de l'exécution: {str(e)}"
        print(f"\n{error_msg}\n")
        
        # Mise à jour de la trace en cas d'erreur
        try:
            langfuse_client.update_current_trace(
                metadata={
                    "type": "multi_agent_system",
                    "partie": "Partie 6",
                    "status": "error",
                    "error": str(e)
                }
            )
        except Exception:
            pass
        
        return error_msg


if __name__ == "__main__":
    print("PARTIE 4 ")
    
    # Exécuter la comparaison de la partie 4
    compare_approaches()
    
    print("PARTIE 6 - L'EMPIRE CHEFBOT")
    
    # Exécuter le test du système multi-agent
    test_empire_chefbot()
    
    # Flush final pour s'assurer que toutes les traces sont envoyées
    try:
        langfuse_client.flush()
        print("\nTraces Langfuse envoyées")
    except Exception as e:
        print(f"\nErreur lors du flush Langfuse: {e}")
    
    print("PARTIES 4 ET 6 TERMINÉES")
