from tsp_graph_init import Lieu, Graph, Route, Affichage
from tsp_aco import TSP_ACO

NB_LIEUX = 500

############ Programme principal ############
if __name__ == "__main__":
    graphe = Graph(nb_lieux=NB_LIEUX)
    aco = TSP_ACO(graphe)
    # Sans fenêtre (résultats dans le terminal) :
    # aco.resoudre(nb_iterations=100)
    # Avec fenêtre :
    fenetre = Affichage(graphe, titre="- colonie de fourmis")
    aco.resoudre_avec_affichage(fenetre, nb_iterations=100)
