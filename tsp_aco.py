"""
tsp_aco.py
==========
Résolution du voyageur de commerce par une colonie de fourmis (ACO).

Historique des améliorations (sur graph_20.csv) :
  plus proche voisin = 3177 -> fourmis + phéromones ≈ 2870 -> + fourmi élitiste ≈ 2860
  -> + 2-opt = 2814 (meilleure route connue) -> + listes de voisins : beaucoup plus rapide
  -> + règle q0 (exploiter 90 % du temps, explorer 10 %) : meilleur sur les gros graphes
  -> + plus d'arrêt automatique : tourne jusqu'à ce qu'on dise stop (Espace = pause / résultat)
  -> + Numba (si installé) : fourmis et 2-opt compilés, fourmis en parallèle

Lancer : python tsp_aco.py   (les réglages sont en bas du fichier)
"""
import random
import time
import numpy as np
from tsp_graph_init import Graph, Route, Affichage

# ---------------------------------------------------------------------------
# Étape 10 : NUMBA (facultatif). Numba compile les fonctions marquées @njit en code machine
# (aussi rapide que du C), et `prange` répartit les fourmis sur les cœurs du processeur.
# Si Numba n'est pas installé, ces fonctions ne sont pas utilisées : le programme garde
# la version Python habituelle (méthodes de la classe TSP_ACO).
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    NUMBA_DISPONIBLE = True
except ImportError:
    NUMBA_DISPONIBLE = False

    def njit(*args, **kwargs):  # sans Numba : décorateur qui ne fait rien
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f

    prange = range


@njit(cache=True)
def construire_route_nb(dist, voisins, tau, alpha, beta, q0):
    """Même règle que choisir_lieu_suivant / construire_route, sur des tableaux numpy."""
    n = dist.shape[0]
    k = voisins.shape[1]
    visite = np.zeros(n, np.bool_)
    ordre = np.empty(n + 1, np.int64)
    candidats = np.empty(k, np.int64)
    poids = np.empty(k, np.float64)
    courant = 0
    visite[0] = True
    ordre[0] = 0
    for etape in range(1, n):
        m = 0
        for t in range(k):  # 1. filtre : les voisins proches libres
            v = voisins[courant, t]
            if not visite[v]:
                d = dist[courant, v]
                if d < 1e-12:
                    d = 1e-12
                candidats[m] = v
                poids[m] = (tau[courant, v] ** alpha) * ((1.0 / d) ** beta)  # 2. odeur × distance
                m += 1
        if m > 0:
            if np.random.random() < q0:  # 3a. exploiter : le meilleur poids
                meilleur = 0
                for t in range(1, m):
                    if poids[t] > poids[meilleur]:
                        meilleur = t
                suivant = candidats[meilleur]
            else:  # 3b. explorer : la roulette
                total = 0.0
                for t in range(m):
                    total += poids[t]
                r = np.random.random() * total
                cumul = 0.0
                suivant = candidats[m - 1]
                for t in range(m):
                    cumul += poids[t]
                    if cumul >= r:
                        suivant = candidats[t]
                        break
        else:  # aucun voisin libre : le lieu libre le plus proche
            suivant = -1
            meilleure_d = np.inf
            for v in range(n):
                if not visite[v] and dist[courant, v] < meilleure_d:
                    meilleure_d = dist[courant, v]
                    suivant = v
        ordre[etape] = suivant
        visite[suivant] = True
        courant = suivant
    ordre[n] = 0
    return ordre


@njit(cache=True)
def deux_opt_nb(ordre, dist, voisins):
    """Même 2-opt rapide que la méthode deux_opt (voisins proches + file d'attente)."""
    n = ordre.shape[0] - 1
    if n < 5:
        return ordre.copy()
    k = voisins.shape[1]
    t = ordre[:n].copy()
    position = np.empty(n, np.int64)
    for p in range(n):
        position[t[p]] = p
    pile = t.copy()  # lieux à tester
    sommet = n
    dans_la_pile = np.ones(n, np.bool_)
    while sommet > 0:
        sommet -= 1
        a = pile[sommet]
        dans_la_pile[a] = False
        for s in range(2):
            sens = 1 if s == 0 else -1
            b = t[(position[a] + sens + n) % n]
            d_ab = dist[a, b]
            change = False
            for q in range(k):
                c = voisins[a, q]
                d_ac = dist[a, c]
                if d_ac >= d_ab:
                    break
                e = t[(position[c] + sens + n) % n]
                if c == b or e == a:
                    continue
                if d_ac + dist[b, e] < d_ab + dist[c, e] - 1e-9:
                    if sens == 1:
                        i = position[b]
                        j = position[c]
                    else:
                        i = position[a]
                        j = position[e]
                    longueur = (j - i + n) % n + 1  # inverser le morceau i..j (côté le plus court)
                    if 2 * longueur > n:
                        i, j = (j + 1) % n, (i - 1 + n) % n
                        longueur = n - longueur
                    for _ in range(longueur // 2):
                        ti = t[i]
                        tj = t[j]
                        t[i] = tj
                        t[j] = ti
                        position[tj] = i
                        position[ti] = j
                        i = (i + 1) % n
                        j = (j - 1 + n) % n
                    for v in (a, b, c, e):  # la route a changé autour de ces 4 lieux
                        if not dans_la_pile[v]:
                            dans_la_pile[v] = True
                            pile[sommet] = v
                            sommet += 1
                    change = True
                    break
            if change:
                break
    resultat = np.empty(n + 1, np.int64)  # on refait commencer la route par 0
    debut = position[0]
    for p in range(n):
        resultat[p] = t[(debut + p) % n]
    resultat[n] = 0
    return resultat


@njit(cache=True, parallel=True)
def fourmis_nb(dist, voisins, tau, alpha, beta, q0, nb_fourmis, avec_2opt):
    """Toutes les fourmis d'une itération, EN PARALLÈLE (elles ne lisent que les phéromones)."""
    n = dist.shape[0]
    ordres = np.empty((nb_fourmis, n + 1), np.int64)
    for f in prange(nb_fourmis):
        o = construire_route_nb(dist, voisins, tau, alpha, beta, q0)
        if avec_2opt:
            o = deux_opt_nb(o, dist, voisins)
        ordres[f, :] = o
    return ordres


class TSP_ACO:
    def __init__(self, graph, nb_fourmis=None, alpha=1.0, beta=3.0, evaporation=0.1, poids_elite=5,
                 utiliser_2opt=True, nb_voisins=10, q0=0.9, utiliser_numba=True):
        self.graph = graph
        self.n = len(graph.liste_lieux)
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

        # Étape 10 : Numba si disponible (sinon, ou en cas d'erreur, la version Python)
        self.voisins_np = np.array(self.voisins, dtype=np.int64)
        self.avec_numba = utiliser_numba and NUMBA_DISPONIBLE and self.preparer_numba()
        self.mode = "Numba (fourmis en parallèle)" if self.avec_numba else "Python"

        # Nombre de fourmis par itération. Sans Numba, sur un gros graphe, on en prend moins
        # pour que la fenêtre se mette à jour (et réponde à Espace) plus souvent.
        if nb_fourmis is None:
            nb_fourmis = 20 if (self.avec_numba or self.n <= 2000) else 5
        self.nb_fourmis = nb_fourmis

        # Point de départ : l'heuristique du plus proche voisin (+ 2-opt si activé)
        self.meilleure_route = Route.depuis_plus_proche_voisin(graph)
        self.longueur_reference = self.meilleure_route.distance  # sert d'échelle pour les dépôts
        if self.utiliser_2opt:
            self.meilleure_route = self.deux_opt(self.meilleure_route)
        self.iteration_meilleure = 0

    def preparer_numba(self):
        """Compile les fonctions Numba sur ce graphe. En cas de problème : on reste en Python."""
        try:
            fourmis_nb(self.graph.matrice_od, self.voisins_np, self.pheromones,
                       self.alpha, self.beta, self.q0, 1, self.utiliser_2opt)
            return True
        except Exception as erreur:  # on ne veut JAMAIS planter à cause de Numba
            print("Numba indisponible pour ce calcul, version Python utilisée :", erreur)
            return False

    def routes_depuis_ordres(self, ordres):
        """Transforme les tableaux renvoyés par Numba en objets Route (avec leur coût)."""
        d = self.graph.matrice_od
        longueurs = d[ordres[:, :-1], ordres[:, 1:]].sum(axis=1)
        routes = []
        for ordre, longueur in zip(ordres, longueurs):
            route = Route(ordre.tolist())
            route.distance = float(longueur)
            routes.append(route)
        return routes

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
        if self.avec_numba:
            return self.routes_depuis_ordres(
                deux_opt_nb(np.array(route.ordre, dtype=np.int64), self.graph.matrice_od,
                            self.voisins_np)[None, :])[0]

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
        if self.avec_numba:  # toutes les fourmis d'un coup, en parallèle
            routes = self.routes_depuis_ordres(
                fourmis_nb(self.graph.matrice_od, self.voisins_np, self.pheromones, self.alpha,
                           self.beta, self.q0, self.nb_fourmis, self.utiliser_2opt))
        else:
            routes = self.routes_python()

        self.mettre_a_jour_pheromones(routes)

        meilleure_du_tour = min(routes, key=lambda r: r.distance)
        if meilleure_du_tour.distance < self.meilleure_route.distance - 1e-9:
            self.meilleure_route = meilleure_du_tour
            return routes, True  # True = on a trouvé mieux
        return routes, False

    def routes_python(self):
        """Version Python : les fourmis l'une après l'autre."""
        routes = []
        for _ in range(self.nb_fourmis):
            route = self.construire_route()
            if self.utiliser_2opt:
                route = self.deux_opt(route)  # on "nettoie" la route de la fourmi
            routes.append(route)
        return routes

    # ------------------------------------------------------------------
    # Étape 9 : tourner sans s'arrêter (c'est le prof qui dit stop)
    # ------------------------------------------------------------------
    def demarrer(self):
        self.debut = time.time()
        self.numero_iteration = 0

    def une_iteration_complete(self):
        """Une itération + mémorisation de l'itération où l'on a trouvé mieux."""
        self.numero_iteration += 1
        routes, amelioration = self.iteration()
        if amelioration:
            self.iteration_meilleure = self.numero_iteration
        return routes, amelioration

    def resoudre(self, temps_max=None):
        """
        Version sans fenêtre : tourne jusqu'à Ctrl + C (ou temps_max secondes si donné),
        puis affiche la meilleure route trouvée.
        """
        self.demarrer()
        print(f"Mode : {self.mode} | {self.nb_fourmis} fourmis | départ : "
              f"{self.meilleure_route.distance:.2f}  (Ctrl + C pour arrêter)")
        try:
            while temps_max is None or time.time() - self.debut < temps_max:
                _, amelioration = self.une_iteration_complete()
                if amelioration:
                    print(f"Itération {self.numero_iteration:5d} : nouvelle meilleure route = "
                          f"{self.meilleure_route.distance:.2f} ({time.time() - self.debut:.1f} s)")
        except KeyboardInterrupt:
            pass
        print(f"RÉSULTAT : meilleure distance = {self.meilleure_route.distance:.2f}, trouvée à "
              f"l'itération {self.iteration_meilleure} ({time.time() - self.debut:.1f} s)")
        return self.meilleure_route

    # ------------------------------------------------------------------
    # Étape 4 (+ 9) : avec la fenêtre, mise à jour à chaque itération
    # ------------------------------------------------------------------
    def resoudre_avec_affichage(self, affichage, delai_ms=20):
        """
        Tourne sans s'arrêter. Espace = pause (on montre le résultat) / reprise. Échap = quitter.
        """
        self.demarrer()
        affichage.mettre_a_jour(meilleure_route=self.meilleure_route, pheromones=self.pheromones,
                                message=f"{self.n} lieux | mode : {self.mode} | {self.nb_fourmis} fourmis | "
                                        f"départ : {self.meilleure_route.distance:.2f} | [Espace] pause / résultat")

        def une_etape():
            if affichage.en_pause:  # pause : on attend sans calculer
                affichage.fenetre.after(200, une_etape)
                return
            routes, _ = self.une_iteration_complete()
            routes_triees = sorted(routes, key=lambda r: r.distance)
            affichage.mettre_a_jour(
                meilleure_route=self.meilleure_route,
                autres_routes=routes_triees[:5],  # pour la touche R
                pheromones=self.pheromones,       # pour la touche P
                message=(f"Itération {self.numero_iteration} | "
                         f"meilleure distance : {self.meilleure_route.distance:.2f} "
                         f"(trouvée à l'itération {self.iteration_meilleure}) | "
                         f"{time.time() - self.debut:.1f} s"))
            affichage.fenetre.after(delai_ms, une_etape)  # et on continue, sans fin

        def quand_pause(en_pause):
            """Appelée par l'affichage quand on appuie sur Espace."""
            if en_pause:
                affichage.ecrire(f"PAUSE - RÉSULTAT : meilleure distance = "
                                 f"{self.meilleure_route.distance:.2f} (trouvée à l'itération "
                                 f"{self.iteration_meilleure}, {time.time() - self.debut:.1f} s)")
                if self.n <= 60:
                    affichage.ecrire(f"Ordre de visite : {self.meilleure_route.ordre}")
            else:
                affichage.ecrire("REPRISE de la recherche")

        affichage.quand_pause = quand_pause
        affichage.fenetre.after(delai_ms, une_etape)
        affichage.lancer()  # la fenêtre reste ouverte (Échap pour quitter)
        print(f"Meilleure distance trouvée : {self.meilleure_route.distance:.2f}")
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
        aco.resoudre_avec_affichage(fenetre)  # Espace = pause / résultat, Échap = quitter
    else:
        aco.resoudre()  # Ctrl + C = arrêter et afficher le résultat