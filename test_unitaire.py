from tsp_graph_init import Lieu, Graph, Route, Affichage
from tsp_aco import TSP_ACO

NB_LIEUX = 5

############ Programme principal ############
if __name__ == "__main__":
    # graphe = Graph("C:/ISEN/5ème année/S9/DP IA/Traitement des données/Fichier_CSV/graph_20.csv")
    graphe = Graph(nb_lieux=NB_LIEUX)
    aco = TSP_ACO(graphe)
    # Sans fenêtre (résultats dans le terminal) :
    # aco.resoudre(nb_iterations=100)
    # Avec fenêtre :
    fenetre = Affichage(graphe, titre="- colonie de fourmis")
    aco.resoudre_avec_affichage(fenetre, nb_iterations=100)
