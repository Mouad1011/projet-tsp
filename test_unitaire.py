from tsp_graph_init import Lieu, Graph, Route, Affichage

######## Fonction utilitaire ##########
def creation_liste_lieux(csv):
    ## Création la liste des lieux grâce au fichier csv chargé
    lieux = []
    for index, row in csv.iterrows():
        lieu = Lieu(row["x"], row["y"], index)
        lieux.append(lieu)

    return lieux

csv = Graph.charger_graph("C:/ISEN/5ème année/S9/DP IA/Traitement des données/Fichier_CSV", 5)
liste_lieux = creation_liste_lieux(csv)
graph = Graph(liste_lieux)
graph.calcul_matrice_cout_od()
print("Matrice OD : ", graph.matrice_od)
proche, distance = graph.plus_proche_voisin("0")
print(proche)
print(distance)
ordre = [0, 1, 2, 3, 4, 0]
distance_total = graph.calcul_distance_route(ordre)
print(distance_total)

point_depart = '0'
csv = Graph.charger_graph("C:/ISEN/5ème année/S9/DP IA/Traitement des données/Fichier_CSV", 5)
liste_lieux = creation_liste_lieux(csv)
graph = Graph(liste_lieux)
graph.calcul_matrice_cout_od()
route = Route([int(point_depart)], 0, point_depart, graph)
route.heuristique_depart(5)
