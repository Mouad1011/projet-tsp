from tsp_graph_init import Lieu, Graph, Route, Affichage

if __name__ == "__main__":
    graphe = Graph("C:/ISEN/5ème année/S9/DP IA/Traitement des données/Fichier_CSV/graph_5.csv")
    # graphe = Graph(nb_lieux=NB_LIEUX)
    premiere_route = Route.depuis_plus_proche_voisin(graphe)
    print("Premier résultat :", premiere_route)
    affichage = Affichage(graphe, titre="- plus proche voisin")
    affichage.mettre_a_jour(meilleure_route=premiere_route,
    message=f"Plus proche voisin : distance = {premiere_route.distance:.2f}")
    affichage.ecrire(f"Ordre de visite : {premiere_route.ordre}")
    affichage.lancer()
