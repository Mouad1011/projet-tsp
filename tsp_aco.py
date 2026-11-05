from tsp_graph_init import Lieu, Graph, Route, Affichage
import random
import numpy as np
from tkinter import TclError
from concurrent.futures import ProcessPoolExecutor

def _lot_de_fourmis(aco, nb, graine):
    """Exécutée dans un processus séparé : fait travailler `nb` fourmis."""
    random.seed(graine)  # indispensable, sinon tous les processus tirent les mêmes routes
    routes = []
    for _ in range(nb):
        route = aco.construire_route()
        if aco.utiliser_2opt:
            route = aco.deux_opt(route)
        routes.append(route)
    return routes

# Classe TSP_ACO : à venir
class TSP_ACO:
    def __init__(self, graph, nb_fourmis=20, alpha=1.0, beta=3.0, evaporation=0.1, poids_elite=5,
                 utiliser_2opt=True, nb_processus=4):
        self.graph = graph
        self.n = len(graph.liste_lieux)
        self.nb_fourmis = nb_fourmis          # nombre de fourmis par itération
        self.alpha = alpha                    # importance des phéromones
        self.beta = beta                      # importance de la distance
        self.evaporation = evaporation        # part des phéromones qui s'évapore à chaque itération
        self.poids_elite = poids_elite        # la meilleure route dépose comme 5 fourmis (0 = désactivé)
        self.utiliser_2opt = utiliser_2opt    # True = chaque route de fourmi est améliorée par le 2-opt
 
        # Au départ, la même quantité de phéromone (1) sur tous les arcs
        self.pheromones = np.ones((self.n, self.n))
 
        # Point de départ : l'heuristique du plus proche voisin
        self.meilleure_route = Route.depuis_plus_proche_voisin(graph)
        self.longueur_reference = self.meilleure_route.distance  # sert d'échelle pour les dépôts
        self.iteration_meilleure = 0

        self.nb_processus = nb_processus
        self.pool = ProcessPoolExecutor(max_workers=nb_processus)
 
    # ------------------------------------------------------------------
    # Étape 2 : une fourmi construit une route
    # ------------------------------------------------------------------
    def choisir_lieu_suivant(self, lieu_actuel, non_visites):
        """Tire au sort le prochain lieu, selon phéromone et distance (la roulette)."""
        candidats = list(non_visites)
        poids = []
        for lieu in candidats:
            distance = self.graph.matrice_od[lieu_actuel][lieu]
            visibilite = 1.0 / distance
            poids.append((self.pheromones[lieu_actuel][lieu] ** self.alpha)
                         * (visibilite ** self.beta))
        return random.choices(candidats, weights=poids)[0]
 
    def construire_route(self):
        """Une fourmi part de 0, visite tous les lieux une fois, et revient à 0."""
        ordre = [0]
        non_visites = set(range(1, self.n))
        lieu_actuel = 0
        while non_visites:
            suivant = self.choisir_lieu_suivant(lieu_actuel, non_visites)
            ordre.append(suivant)
            non_visites.remove(suivant)
            lieu_actuel = suivant
        ordre.append(0)
        route = Route(ordre)
        route.distance = self.graph.calcul_distance_route(route)
        return route
 
    # ------------------------------------------------------------------
    # Étape 6 : le 2-opt (heuristique d'optimisation)
    # ------------------------------------------------------------------
    def deux_opt(self, route):
        """
        Améliore une route en "décroisant" ses arcs.
        Pour chaque paire d'arcs (a→b) et (c→d) : si (a→c) + (b→d) est plus court,
        on inverse le morceau de route entre b et c. On recommence tant que ça s'améliore.
        """
        d = self.graph.matrice_od
        ordre = list(route.ordre)  # copie de la liste, ex : [0, 3, 1, 2, 0]
        amelioration = True
        while amelioration:
            amelioration = False
            for i in range(1, len(ordre) - 2):          # b = ordre[i] (on ne bouge jamais le 0 de départ)
                for j in range(i + 1, len(ordre) - 1):  # c = ordre[j] (ni le 0 d'arrivée)
                    a, b = ordre[i - 1], ordre[i]
                    c, e = ordre[j], ordre[j + 1]
                    avant = d[a][b] + d[c][e]
                    apres = d[a][c] + d[b][e]
                    if apres < avant - 1e-9:  # plus court : on inverse le morceau b ... c
                        ordre[i:j + 1] = reversed(ordre[i:j + 1])
                        amelioration = True
        nouvelle_route = Route(ordre)
        nouvelle_route.distance = self.graph.calcul_distance_route(nouvelle_route)
        return nouvelle_route
 
    # ------------------------------------------------------------------
    # Étape 3 : les phéromones
    # ------------------------------------------------------------------
    def mettre_a_jour_pheromones(self, routes):
        """Évaporation partout, puis chaque fourmi dépose sur les arcs de sa route."""
        # 1. Évaporation : toutes les phéromones diminuent un peu
        self.pheromones = self.pheromones * (1 - self.evaporation)
 
        # 2. Dépôt : plus la route est courte, plus on dépose
        for route in routes:
            depot = self.longueur_reference / route.distance
            self.deposer(route, depot)
 
        # 3. Fourmi élitiste : la meilleure route trouvée depuis le début dépose aussi,
        # comme si elle était parcourue par plusieurs fourmis (poids_elite).
        # La colonie ne l'"oublie" plus, et explore autour d'elle.
        depot_elite = self.poids_elite * self.longueur_reference / self.meilleure_route.distance
        self.deposer(self.meilleure_route, depot_elite)
 
    def deposer(self, route, depot):
        """Ajoute `depot` de phéromone sur chaque arc de la route (dans les deux sens)."""
        for k in range(len(route.ordre) - 1):
            i = route.ordre[k]
            j = route.ordre[k + 1]
            self.pheromones[i][j] += depot
            self.pheromones[j][i] += depot  # même arc dans l'autre sens
 
    def iteration(self):
        """Une itération : toutes les fourmis font leur tour, puis mise à jour des phéromones."""
        base, reste = divmod(self.nb_fourmis, self.nb_processus)
        tailles = [base + (1 if k < reste else 0) for k in range(self.nb_processus)]
        graines = [random.getrandbits(32) for _ in tailles]

        lots = self.pool.map(_lot_de_fourmis, [self] * self.nb_processus, tailles, graines)
        routes = [r for lot in lots for r in lot]
 
        # On met à jour les phéromones avec les routes de ce tour
        self.mettre_a_jour_pheromones(routes)
 
        # On garde la meilleure route vue depuis le début
        meilleure_du_tour = min(routes, key=lambda r: r.distance)
        if meilleure_du_tour.distance < self.meilleure_route.distance:
            self.meilleure_route = meilleure_du_tour
            return routes, True  # True = on a trouvé mieux
        return routes, False
 
    def resoudre(self, nb_iterations=100):
        print(f"Départ (plus proche voisin) : {self.meilleure_route.distance:.2f}")
        for it in range(1, nb_iterations + 1):
            routes, amelioration = self.iteration()
            if amelioration:
                self.iteration_meilleure = it
                print(f"Itération {it:3d} : nouvelle meilleure route = {self.meilleure_route.distance:.2f}")
        print(f"Fin : meilleure distance = {self.meilleure_route.distance:.2f} "
              f"(trouvée à l'itération {self.iteration_meilleure})")
        self.pool.shutdown()
        return self.meilleure_route
 
    # ------------------------------------------------------------------
    # Étape 4 : la même chose, mais en affichant chaque itération dans la fenêtre
    # ------------------------------------------------------------------
    def resoudre_avec_affichage(self, affichage, nb_iterations=100, delai_ms=50):
        """Lance une itération, met à jour la fenêtre, attend delai_ms, recommence."""
        affichage.mettre_a_jour(meilleure_route=self.meilleure_route,
                                pheromones=self.pheromones,
                                message=f"Départ (plus proche voisin) : {self.meilleure_route.distance:.2f}")
        self.numero_iteration = 0
 
        def une_etape():
            self.numero_iteration += 1
            routes, amelioration = self.iteration()
            if amelioration:
                self.iteration_meilleure = self.numero_iteration
            routes_triees = sorted(routes, key=lambda r: r.distance)
            affichage.mettre_a_jour(
                meilleure_route=self.meilleure_route,
                autres_routes=routes_triees[:5],   # pour la touche R
                pheromones=self.pheromones,        # pour la touche P
                message=(f"Itération {self.numero_iteration}/{nb_iterations} | "
                         f"meilleure distance : {self.meilleure_route.distance:.2f} "
                         f"(trouvée à l'itération {self.iteration_meilleure})"))
            if self.numero_iteration < nb_iterations:
                affichage.fenetre.after(delai_ms, une_etape)  # on programme l'itération suivante
            else:
                affichage.ecrire(f"TERMINÉ : meilleure distance = {self.meilleure_route.distance:.2f}")
                affichage.ecrire(f"Ordre de visite : {self.meilleure_route.ordre}")
 
        affichage.fenetre.after(delai_ms, une_etape)  # première itération dans delai_ms
        affichage.lancer()  # la fenêtre reste ouverte (Échap pour quitter)
        return self.meilleure_route

    def __getstate__(self):
        etat = self.__dict__.copy()
        etat["pool"] = None
        return etat
    
    