from tsp_graph_init import Lieu, Graph, Route, Affichage

tau_initial = 0.1
point_depart = "0"
alpha = 1
beta = 1
rho = 0.5
Q = 100
NB_LIEUX = 5

######## Fonctions utilitaires ##########
def creation_liste_lieux(csv):
    ## Création la liste des lieux grâce au fichier csv chargé
    lieux = []
    for index, row in csv.iterrows():
        lieu = Lieu(row["x"], row["y"], index)
        lieux.append(lieu)

    return lieux

def creation_matrice_phéromones(matrice_od, tau_initial):
    matrice_pheromones = {}
    for cle in matrice_od.keys():
        matrice_pheromones[cle] = tau_initial

    return matrice_pheromones

test = TSP_HERISTIQUE(graph, route, point_depart)
test.start()




# Classe TSP_ACO : à venir
class TSP_ACO:
    def __init__(self, pheromones, matrice_od):
        self.pheromones = pheromones
        self.matrice_od = matrice_od

    
    