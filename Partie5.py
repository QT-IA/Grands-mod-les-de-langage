import os
from dotenv import load_dotenv
from smolagents import CodeAgent, Tool, LiteLLMModel, tool
from typing import Optional, List
import json

load_dotenv()

# PARTIE 5 - Restaurant Intellignent

# 5.1 Outil Base de Données
class MenuDatabaseTool(Tool):
    name = "menu_db"
    description = "Consulte la base de données des plats du restaurant pour trouver des plats selon des critères (catégorie, prix maximum, restrictions alimentaires)."
    inputs = {
        "category": {
            "type": "string",
            "description": "La catégorie du plat recherché (ex: 'Entrée', 'Plat', 'Dessert'). Si non spécifié, cherche dans tout.",
            "nullable": True
        },
        "max_price": {
            "type": "number",
            "description": "Le prix maximum toléré pour le plat (en euros).",
            "nullable": True
        },
        "dietary_restriction": {
            "type": "string",
            "description": "Restriction alimentaire (ex: 'vegetarien', 'sans gluten', 'vegan').",
            "nullable": True
        }
    }
    output_type = "string"

    def __init__(self):
        super().__init__()
        # Initialisation de la base de données (au moins 10 plats)
        # Champs: nom, prix, temps de preparation, allergenes, categorie
        self.menu_data = [
            {"name": "Salade César", "price": 12, "prep_time": 10, "allergens": ["gluten", "lait", "oeuf"], "category": "Entrée", "tags": []},
            {"name": "Soupe à l'oignon", "price": 10, "prep_time": 15, "allergens": ["gluten", "lait"], "category": "Entrée", "tags": ["vegetarien"]},
            {"name": "Carpaccio de Boeuf", "price": 14, "prep_time": 10, "allergens": [], "category": "Entrée", "tags": ["sans gluten"]},
            {"name": "Escargots de Bourgogne", "price": 16, "prep_time": 15, "allergens": ["beurre"], "category": "Entrée", "tags": []},
            
            {"name": "Boeuf Bourguignon", "price": 22, "prep_time": 120, "allergens": ["sulfites"], "category": "Plat", "tags": ["sans gluten"]},
            {"name": "Filet de Saumon", "price": 20, "prep_time": 20, "allergens": ["poisson"], "category": "Plat", "tags": ["sans gluten"]},
            {"name": "Risotto aux Champignons", "price": 18, "prep_time": 25, "allergens": ["lait"], "category": "Plat", "tags": ["vegetarien", "sans gluten"]},
            {"name": "Ratatouille Provencale", "price": 16, "prep_time": 30, "allergens": [], "category": "Plat", "tags": ["vegetarien", "vegan", "sans gluten"]},
            {"name": "Poulet Rôti", "price": 19, "prep_time": 40, "allergens": [], "category": "Plat", "tags": ["sans gluten"]},
            
            {"name": "Mousse au Chocolat", "price": 8, "prep_time": 10, "allergens": ["oeuf", "lait"], "category": "Dessert", "tags": ["vegetarien", "sans gluten"]},
            {"name": "Tarte Tatin", "price": 9, "prep_time": 45, "allergens": ["gluten", "lait"], "category": "Dessert", "tags": ["vegetarien"]},
            {"name": "Salade de Fruits", "price": 7, "prep_time": 10, "allergens": [], "category": "Dessert", "tags": ["vegetarien", "vegan", "sans gluten"]},
            {"name": "Crème Brûlée", "price": 9, "prep_time": 60, "allergens": ["lait", "oeuf"], "category": "Dessert", "tags": ["vegetarien", "sans gluten"]}
        ]

    def forward(self, category: str = None, max_price: float = None, dietary_restriction: str = None) -> str:
        results = []
        for dish in self.menu_data:
            keep = True
            
            # Filtre par catégorie
            if category and dish["category"].lower() != category.lower():
                keep = False
            
            # Filtre par prix
            if max_price is not None and dish["price"] > max_price:
                keep = False
            
            # Filtre par restriction (basé sur tags ou allergènes)
            if dietary_restriction:
                req = dietary_restriction.lower()
                
                # Check tags explicitly
                dish_tags = [t.lower() for t in dish["tags"]]
                
                if "vegetarien" in req:
                    if "vegetarien" not in dish_tags and "vegan" not in dish_tags:
                        keep = False
                
                if "vegan" in req:
                    if "vegan" not in dish_tags:
                        keep = False

                if "gluten" in req and ("sans" in req or "free" in req):
                    if "sans gluten" not in dish_tags: # ou vérifier allergènes
                        # Fallback check allergens
                        if "gluten" in [a.lower() for a in dish["allergens"]]:
                            keep = False
                        # If not explicitly tagged sans gluten and no gluten allergen, assume ok? 
                        # Safe approach: if not tagged sans gluten, exclude. But let's be lenient if no allergens.
                        if "gluten" in [a.lower() for a in dish["allergens"]]:
                            keep = False

            if keep:
                results.append(dish)
        
        if not results:
            return "Aucun plat trouvé correspondant exactement aux critères. Essayez d'élargir la recherche."
        
        # Formatage lisible pour le LLM
        return json.dumps(results, ensure_ascii=False)

@tool
def calculate(expression: str) -> str:
    """
    Calcule le résultat d'une expression mathématique simple. Utile pour additionner les prix.
    Args:
        expression: L'expression mathématique (ex: '20 + 15 + 8').
    """
    try:
        # Restriction pour sécurité basique
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "Erreur: Caractères non autorisés dans le calcul."
        return str(eval(expression))
    except Exception as e:
        return f"Erreur de calcul: {e}"

# Main
def main():
    # Définition du modèle
    # Ajout d'une limite de tokens de sortie pour éviter les erreurs de quota (TPM / taille de requête)
    # On passe max_tokens directement à LiteLLMModel pour limiter la réponse
    model = LiteLLMModel(model_id="groq/qwen/qwen3-32b", max_tokens=5500)
    
    menu_tool = MenuDatabaseTool()
    
    # 5.2 Agent avec planification
    system_prompt = (
        "Tu es un maître d'hôtel expérimenté dans un restaurant gastronomique. "
        "Ton rôle est de conseiller les clients, de prendre leur commande et de vérifier qu'elle correspond à leurs besoins (budget, allergies). "
        "Utilise les outils à ta disposition pour vérifier le menu et calculer les prix. "
        "Sois courtois, professionnel et précis."
    )

    print("\n" + "="*50)
    print("5.2 - Agent avec Planification")
    print("="*50)

    planner_agent = CodeAgent(
        tools=[menu_tool, calculate],
        model=model,
        planning_interval=2, 
        max_steps=12,
        verbosity_level=1,
        additional_authorized_imports=['json']
    )

    query_5_2 = "On est 3. Un vegetarien, un sans gluten, et moi je mange de tout. Budget max 60 euros pour le groupe. Proposez-nous un menu complet (Entrée + Plat ou Plat + Dessert chacun) qui rentre dans le budget."
    
    print(f"QUERY: {query_5_2}\n")
    
    # Capture pour le fichier log
    logs = []
    logs.append("=== PARTIE 5.2 ===")
    logs.append(f"Query: {query_5_2}")
    
    try:
        res_5_2 = planner_agent.run(query_5_2)
        print(f"\nRESULTAT 5.2: {res_5_2}")
        logs.append(f"Result: {res_5_2}")
    except Exception as e:
        print(f"Erreur 5.2: {e}")
        logs.append(f"Erreur: {e}")

    # 5.3 Agent Conversationnel
    print("\n" + "="*50)
    print("5.3 - Agent Conversationnel (Multi-tours)")
    print("="*50)
    
    chat_agent = CodeAgent(
        tools=[menu_tool, calculate],
        model=model,
        planning_interval=2,
        max_steps=10,
        verbosity_level=1,
        additional_authorized_imports=['json']
    )
    
    dialogue_turns = [
        "Bonjour, avez-vous des suggestions pour un plat à base de poisson ?",
        "Je vois, mais finalement je n'aime pas trop le saumon. Avez-vous autre chose ou sinon une viande sans sulfites ?",
        "Parfait, je prends ça. Combien cela coûtera avec une mousse au chocolat en dessert ?"
    ]
    
    logs.append("\n=== PARTIE 5.3 ===")
    
    for i, user_input in enumerate(dialogue_turns):
        print(f"\n--- Tour {i+1} ---")
        print(f"User: {user_input}")
        logs.append(f"\nUser [{i+1}]: {user_input}")
        
        # reset=False permet de garder l'historique de la conversation
        response = chat_agent.run(user_input, reset=False)
        
        print(f"Agent: {response}")
        logs.append(f"Agent [{i+1}]: {response}")

    # Sauvegarde des traces
    with open("run_restaurant.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    print("\nTrace sauvegardée dans 'run_restaurant.txt'.")

if __name__ == "__main__":
    main()
