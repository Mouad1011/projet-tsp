import random
import time
import numpy as np
from tsp_graph_init import Graph, Route, Affichage


class TSP_ACO:
    def __init__(self, graph, nb_fourmis=20, alpha=1.0, beta=3.0, evaporation=0.1, poids_elite=5,
                 utiliser_2opt=True, nb_voisins=10, q0=0.9):
        self.graph = graph
        self.n = len(graph.liste_lieux)
        self.nb_fourmis = nb_fourmis          # nombre de fourmis par itération
        self.alpha = alpha                    # importance des phéromones
        self.beta = beta                      # importance de la distance
        self.evaporation = evaporation        # part des phéromones qui s'évapore à chaque itération
        self.poids_elite = poids_elite        # la meilleure route dépose comme 5 fourmis (0 = désactivé)
        self.utiliser_2opt = utiliser_2opt    # True = chaque route de fourmi est améliorée par le 2-opt
        self.q0 = q0                          # probabilité de prendre directement le meilleur choix (0 = toujours la roulette)

        # Petits/moyens graphes : copie de la matrice en liste Python (lecture d'une case plus rapide).
        # Gros graphes : on lit directement la matrice numpy, pour ne pas doubler la mémoire utilisée.
        self.dist = graph.matrice_od.tolist() if self.n <= 2000 else graph.matrice_od

        # Étape 7 : LISTES DE VOISINS. Pour chaque lieu, ses nb_voisins lieux les plus proches,
        # triés du plus proche au plus loin. Calculées une seule fois, comme la matrice OD.
        self.voisins = self.calculer_voisins(min(nb_voisins, self.n - 1))

        # Au départ, la même quantité de phéromone (1) sur tous les arcs
        # (float32 sur les gros graphes : 2 fois moins de mémoire, largement assez précis)
        self.pheromones = np.ones((self.n, self.n), dtype=np.float64 if self.n <= 2000 else np.float32)

        # Point de départ : l'heuristique du plus proche voisin (+ 2-opt si activé)
        self.meilleure_route = Route.depuis_plus_proche_voisin(graph)
        self.longueur_reference = self.meilleure_route.distance  # sert d'échelle pour les dépôts
        if self.utiliser_2opt:
            self.meilleure_route = self.deux_opt(self.meilleure_route)
        self.iteration_meilleure = 0

    def calculer_voisins(self, k):
        """Pour chaque lieu, la liste de ses k lieux les plus proches (du plus proche au plus loin)."""
        voisins = []
        matrice = self.graph.matrice_od
        for debut in range(0, self.n, 500):  # par paquets de 500 lignes (moins de mémoire)
            lignes = matrice[debut:debut + 500]
            # argpartition : trouve les k+1 plus petites distances de chaque ligne sans trier toute la ligne
            proches = np.argpartition(lignes, k, axis=1)[:, :k + 1]
            for numero, cand in enumerate(proches, start=debut):
                cand = sorted((int(v) for v in cand if v != numero), key=lambda v: matrice[numero, v])
                voisins.append(cand[:k])
        return voisins

    # ------------------------------------------------------------------
    # Étape 2 (+ étape 7) : une fourmi construit une route
    # ------------------------------------------------------------------
    def choisir_lieu_suivant(self, lieu_actuel, visite):
        """
        La fourmi tire au sort parmi ses VOISINS PROCHES non visités (roulette).
        Si tous ses voisins proches sont déjà visités, elle va au lieu libre le plus proche.
        """
        candidats = [v for v in self.voisins[lieu_actuel] if not visite[v]]
        if candidats:
            tau = self.pheromones[lieu_actuel]
            poids = [(tau[v] ** self.alpha) * ((1.0 / self.dist[lieu_actuel][v]) ** self.beta)
                     for v in candidats]
            # Étape 8 (Ant Colony System) : avec la probabilité q0, la fourmi EXPLOITE
            # (elle prend directement le meilleur choix) ; sinon elle EXPLORE (roulette).
            if random.random() < self.q0:
                return candidats[poids.index(max(poids))]
            return random.choices(candidats, weights=poids)[0]
        # Plus de voisin proche libre : le lieu non visité le plus proche (calcul numpy rapide)
        distances = self.graph.matrice_od[lieu_actuel].copy()
        distances[visite] = np.inf
        return int(np.argmin(distances))

    def construire_route(self):
        """Une fourmi part de 0, visite tous les lieux une fois, et revient à 0."""
        visite = np.zeros(self.n, dtype=bool)  # visite[i] = True si le lieu i est déjà visité
        visite[0] = True
        ordre = [0]
        lieu_actuel = 0
        for _ in range(self.n - 1):
            suivant = self.choisir_lieu_suivant(lieu_actuel, visite)
            ordre.append(suivant)
            visite[suivant] = True
            lieu_actuel = suivant
        ordre.append(0)
        route = Route(ordre)
        route.distance = self.graph.calcul_distance_route(route)
        return route

    # ------------------------------------------------------------------
    # Étape 6 : le 2-opt simple (toutes les paires d'arcs) - gardé pour comparer
    # ------------------------------------------------------------------
    def deux_opt_simple(self, route):
        """Teste TOUTES les paires d'arcs (≈ n² tests par passage). Simple mais lent si n est grand."""
        d = self.dist
        ordre = list(route.ordre)
        amelioration = True
        while amelioration:
            amelioration = False
            for i in range(1, len(ordre) - 2):
                for j in range(i + 1, len(ordre) - 1):
                    a, b = ordre[i - 1], ordre[i]
                    c, e = ordre[j], ordre[j + 1]
                    if d[a][c] + d[b][e] < d[a][b] + d[c][e] - 1e-9:
                        ordre[i:j + 1] = reversed(ordre[i:j + 1])
                        amelioration = True
        nouvelle_route = Route(ordre)
        nouvelle_route.distance = self.graph.calcul_distance_route(nouvelle_route)
        return nouvelle_route

    # ------------------------------------------------------------------
    # Étape 7 : le 2-opt rapide (seulement vers les voisins proches)
    # ------------------------------------------------------------------
    def deux_opt(self, route):
        """
        Même idée que le 2-opt simple : remplacer les arcs (a→b) et (c→e) par (a→c) et (b→e)
        si c'est plus court. Mais pour chaque lieu a, on ne teste que c parmi ses VOISINS PROCHES :
        relier a à un lieu lointain ne peut presque jamais raccourcir la route.
        On ne re-teste que les lieux autour desquels la route vient de changer (file d'attente).
        """
        d = self.dist
        t = list(route.ordre[:-1])  # le circuit, sans le 0 final répété
        n = len(t)
        if n < 5:
            return self.deux_opt_simple(route)

        position = [0] * n  # position[v] = place du lieu v dans le circuit
        for p, v in enumerate(t):
            position[v] = p

        def inverser(i, j):
            """Inverse le morceau du circuit entre les places i et j (le circuit est une boucle)."""
            longueur = (j - i) % n + 1
            if 2 * longueur > n:  # inverser l'autre côté de la boucle revient au même, et c'est plus court
                i, j = (j + 1) % n, (i - 1) % n
                longueur = n - longueur
            for _ in range(longueur // 2):
                t[i], t[j] = t[j], t[i]
                position[t[i]], position[t[j]] = i, j
                i, j = (i + 1) % n, (j - 1) % n

        a_tester = list(t)  # au départ, on teste tous les lieux
        dans_la_file = [True] * n
        while a_tester:
            a = a_tester.pop()
            dans_la_file[a] = False
            for sens in (1, -1):  # on regarde l'arc vers le lieu suivant, puis vers le précédent
                b = t[(position[a] + sens) % n]
                d_ab = d[a][b]
                change = False
                for c in self.voisins[a]:
                    if d[a][c] >= d_ab:
                        break  # voisins triés : les suivants sont encore plus loin, inutile
                    e = t[(position[c] + sens) % n]
                    if c == b or e == a:
                        continue
                    if d[a][c] + d[b][e] < d_ab + d[c][e] - 1e-9:  # plus court ?
                        if sens == 1:
                            inverser(position[b], position[c])
                        else:
                            inverser(position[a], position[e])
                        for v in (a, b, c, e):  # la route a changé autour d'eux
                            if not dans_la_file[v]:
                                dans_la_file[v] = True
                                a_tester.append(v)
                        change = True
                        break
                if change:
                    break

        k = t.index(0)  # on refait commencer la route par le lieu 0
        ordre = t[k:] + t[:k] + [0]
        nouvelle_route = Route(ordre)
        nouvelle_route.distance = self.graph.calcul_distance_route(nouvelle_route)
        return nouvelle_route

    # ------------------------------------------------------------------
    # Étapes 3 et 5 : les phéromones (+ fourmi élitiste)
    # ------------------------------------------------------------------
    def mettre_a_jour_pheromones(self, routes):
        """Évaporation partout, puis chaque fourmi dépose sur les arcs de sa route."""
        self.pheromones *= (1 - self.evaporation)  # 1. évaporation
        for route in routes:                       # 2. dépôt des fourmis
            self.deposer(route, self.longueur_reference / route.distance)
        depot_elite = self.poids_elite * self.longueur_reference / self.meilleure_route.distance
        self.deposer(self.meilleure_route, depot_elite)  # 3. fourmi élitiste

    def deposer(self, route, depot):
        """Ajoute `depot` de phéromone sur chaque arc de la route (dans les deux sens)."""
        ordre = np.array(route.ordre)
        self.pheromones[ordre[:-1], ordre[1:]] += depot
        self.pheromones[ordre[1:], ordre[:-1]] += depot

    def iteration(self):
        """Une itération : toutes les fourmis font leur tour, puis mise à jour des phéromones."""
        routes = []
        for _ in range(self.nb_fourmis):
            route = self.construire_route()
            if self.utiliser_2opt:
                route = self.deux_opt(route)  # on "nettoie" la route de la fourmi
            routes.append(route)

        self.mettre_a_jour_pheromones(routes)

        meilleure_du_tour = min(routes, key=lambda r: r.distance)
        if meilleure_du_tour.distance < self.meilleure_route.distance - 1e-9:
            self.meilleure_route = meilleure_du_tour
            return routes, True  # True = on a trouvé mieux
        return routes, False

    # ------------------------------------------------------------------
    # Lancement : l'algorithme tourne sans limite (Ctrl+C dans le terminal pour arrêter)
    # ------------------------------------------------------------------
    def resoudre(self):
        """Version sans fenêtre : les résultats s'affichent dans le terminal."""
        self.debut = time.time()
        self.numero_iteration = 0
        print(f"Départ : {self.meilleure_route.distance:.2f}  (Ctrl+C pour arrêter)")
        try:
            while True:
                self.numero_iteration += 1
                routes, amelioration = self.iteration()
                if amelioration:
                    self.iteration_meilleure = self.numero_iteration
                    print(f"Itération {self.numero_iteration:4d} : nouvelle meilleure route = "
                          f"{self.meilleure_route.distance:.2f} ({time.time() - self.debut:.1f} s)")
        except KeyboardInterrupt:
            pass
        print(f"ARRÊT : meilleure distance = {self.meilleure_route.distance:.2f}, "
              f"trouvée à l'itération {self.iteration_meilleure} ({time.time() - self.debut:.1f} s)")
        return self.meilleure_route

    # ------------------------------------------------------------------
    # Étape 4 : avec la fenêtre, mise à jour à chaque itération
    # ------------------------------------------------------------------
    def resoudre_avec_affichage(self, affichage, delai_ms=20):
        """Lance une itération, met à jour la fenêtre, attend delai_ms, recommence."""
        self.debut = time.time()
        self.numero_iteration = 0
        affichage.mettre_a_jour(meilleure_route=self.meilleure_route, pheromones=self.pheromones,
                                message=f"{self.n} lieux | départ : {self.meilleure_route.distance:.2f}")

        def une_etape():
            self.numero_iteration += 1
            routes, amelioration = self.iteration()
            if amelioration:
                self.iteration_meilleure = self.numero_iteration
            routes_triees = sorted(routes, key=lambda r: r.distance)
            affichage.mettre_a_jour(
                meilleure_route=self.meilleure_route,
                autres_routes=routes_triees[:5],  # pour la touche R
                pheromones=self.pheromones,       # pour la touche P
                message=(f"Itération {self.numero_iteration} | "
                         f"meilleure distance : {self.meilleure_route.distance:.2f} "
                         f"(trouvée à l'itération {self.iteration_meilleure}) | "
                         f"{time.time() - self.debut:.1f} s"))
            affichage.fenetre.after(delai_ms, une_etape)  # on programme l'itération suivante

        affichage.fenetre.after(delai_ms, une_etape)
        affichage.lancer()  # la fenêtre reste ouverte (Échap pour quitter)
        return self.meilleure_route


if __name__ == "__main__":
    # ---------------- Réglages ----------------
    FICHIER_CSV = "graph_20.csv"  # ou None pour des lieux au hasard
    NB_LIEUX = 100                # utilisé seulement si FICHIER_CSV = None
    AVEC_FENETRE = True
    # ------------------------------------------

    graphe = Graph(FICHIER_CSV) if FICHIER_CSV else Graph(nb_lieux=NB_LIEUX)
    aco = TSP_ACO(graphe)
    if AVEC_FENETRE:
        fenetre = Affichage(graphe, titre="- colonie de fourmis")
        aco.resoudre_avec_affichage(fenetre)
    else:
        aco.resoudre()