import csv
import random
import tkinter as tk
import numpy as np

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
LARGEUR = 800             # largeur de la zone de dessin
HAUTEUR = 600             # hauteur de la zone de dessin
NB_LIEUX = 10             # nombre de lieux si on les génère au hasard
NOM_GROUPE = "Groupe X"   # <-- mettre le nom du groupe ici
RAYON = 12                # taille des cercles des lieux


# ---------------------------------------------------------------------------
# Classe Lieu
# ---------------------------------------------------------------------------
class Lieu:
    """Un lieu à visiter : coordonnées x, y et un nom."""

    def __init__(self, x, y, nom=""):
        self.x = float(x)  # float() car le CSV donne du texte
        self.y = float(y)
        self.nom = str(nom)

    def calcul_distance(self, autre_lieu):
        """Distance euclidienne (Pythagore) entre ce lieu et un autre."""
        delta = np.array([self.x, self.y]) - np.array([autre_lieu.x, autre_lieu.y])
        return np.sqrt(np.dot(delta, delta))


# ---------------------------------------------------------------------------
# Classe Graph
# ---------------------------------------------------------------------------
class Graph:
    """Tous les lieux à visiter + la matrice des distances entre eux."""

    def __init__(self, fichier_csv=None, nb_lieux=NB_LIEUX):
        self.liste_lieux = []
        self.matrice_od = None
        if fichier_csv is not None:
            self.charger_graph(fichier_csv)            # lieux lus dans un CSV
        else:
            self.generer_lieux_aleatoires(nb_lieux)    # lieux au hasard
        # On calcule toutes les distances UNE SEULE FOIS, ici.
        self.calcul_matrice_cout_od()

    def generer_lieux_aleatoires(self, nb_lieux):
        """Crée nb_lieux lieux au hasard, qui tiennent dans LARGEUR x HAUTEUR."""
        self.liste_lieux = []
        for i in range(nb_lieux):
            x = random.uniform(RAYON, LARGEUR - RAYON)  # on garde une marge pour
            y = random.uniform(RAYON, HAUTEUR - RAYON)  # que le cercle ne sorte pas
            self.liste_lieux.append(Lieu(x, y, str(i)))

    def charger_graph(self, chemin_fichier):
        """Lit un CSV avec une colonne x et une colonne y (une ligne = un lieu)."""
        self.liste_lieux = []
        with open(chemin_fichier, "r", encoding="utf-8") as fichier:
            lecteur = csv.DictReader(fichier)  # lit chaque ligne comme {"x": ..., "y": ...}
            for numero, ligne in enumerate(lecteur):
                self.liste_lieux.append(Lieu(ligne["x"], ligne["y"], str(numero)))

    def calcul_matrice_cout_od(self):
        """Remplit matrice_od : matrice_od[i][j] = distance entre le lieu i et le lieu j."""
        # Même calcul que Lieu.calcul_distance (Pythagore), mais fait pour TOUS les couples
        # d'un coup avec numpy : beaucoup plus rapide qu'une double boucle sur un gros graphe.
        x = np.array([lieu.x for lieu in self.liste_lieux])
        y = np.array([lieu.y for lieu in self.liste_lieux])
        dx = x[:, None] - x[None, :]  # dx[i][j] = x_i - x_j
        dy = y[:, None] - y[None, :]  # dy[i][j] = y_i - y_j
        dx *= dx  # calculs "sur place" : pas de tableau intermédiaire en plus,
        dy *= dy  # ce qui économise beaucoup de mémoire sur un gros graphe
        dx += dy
        del dy
        self.matrice_od = np.sqrt(dx, out=dx)  # racine carrée -> distance (Pythagore)
        return self.matrice_od

    def plus_proche_voisin(self, lieu, deja_visites):
        """
        Renvoie le numéro du lieu NON visité le plus proche de `lieu`,
        en lisant la matrice des distances (aucun calcul de distance ici).
        """
        distances = self.matrice_od[lieu].copy()  # la ligne de la matrice : distances depuis `lieu`
        distances[lieu] = np.inf                  # on exclut le lieu lui-même...
        distances[list(deja_visites)] = np.inf    # ... et les lieux déjà visités
        meilleur_voisin = int(np.argmin(distances))  # le plus petit = le plus proche
        if np.isinf(distances[meilleur_voisin]):
            return None  # plus aucun lieu disponible
        return meilleur_voisin

    def calcul_distance_route(self, route):
        """Coût d'une route = somme des distances de chaque étape."""
        total = 0
        for k in range(len(route.ordre) - 1):
            depart = route.ordre[k]
            arrivee = route.ordre[k + 1]
            total += self.matrice_od[depart][arrivee]
        return total


# ---------------------------------------------------------------------------
# Classe Route
# ---------------------------------------------------------------------------
class Route:
    """
    Un trajet. ordre = liste des numéros de lieux visités,
    qui commence ET finit par 0. Exemple : [0, 3, 1, 2, 0]
    """

    def __init__(self, ordre):
        self.ordre = list(ordre)
        self.distance = None  # le coût, calculé par le Graph

    @staticmethod
    def depuis_plus_proche_voisin(graph):
        """
        HEURISTIQUE DE DÉPART : depuis où je suis, je vais toujours
        au lieu non visité le plus proche. Donne une première route rapidement.
        """
        ordre = [0]  # on part du lieu 0
        deja_visites = {0}
        lieu_actuel = 0
        while len(ordre) < len(graph.liste_lieux):
            suivant = graph.plus_proche_voisin(lieu_actuel, deja_visites)
            ordre.append(suivant)
            deja_visites.add(suivant)
            lieu_actuel = suivant
        ordre.append(0)  # retour au point de départ
        route = Route(ordre)
        route.distance = graph.calcul_distance_route(route)
        return route

    def __repr__(self):
        return f"Route(distance={self.distance:.2f}, ordre={self.ordre})"


# ---------------------------------------------------------------------------
# Classe Affichage
# ---------------------------------------------------------------------------
class Affichage:
    """
    Fenêtre Tkinter :
      zone de dessin LARGEUR x HAUTEUR : lieux, meilleure route, phéromones
      zone de texte en dessous : avancée de l'algorithme
    Touches : P = phéromones, R = N meilleures routes, Échap = quitter
    """

    def __init__(self, graph, titre=""):
        self.graph = graph
        self.meilleure_route = None
        self.autres_routes = []     # les N meilleures routes (en gris)
        self.pheromones = None      # matrice des phéromones (donnée par les fourmis)
        self.voir_pheromones = False
        self.voir_autres_routes = False
        self.en_pause = False       # Espace : pause / reprise de l'algorithme
        self.quand_pause = None     # fonction donnée par l'algorithme, appelée à chaque pause/reprise

        # Sur un gros graphe, on dessine des lieux plus petits et sans numéros (sinon illisible)
        n = len(graph.liste_lieux)
        self.rayon = RAYON if n <= 50 else 8 if n <= 100 else 4 if n <= 300 else 2
        self.avec_numeros = n <= 100

        self.fenetre = tk.Tk()
        self.fenetre.title(f"{NOM_GROUPE} - Voyageur de commerce {titre}")
        self.zone_dessin = tk.Canvas(self.fenetre, width=LARGEUR, height=HAUTEUR, bg="white")
        self.zone_dessin.pack()
        tk.Label(self.fenetre, text="[P] phéromones   [R] meilleures routes   "
                                    "[Espace] pause / résultat   [Échap] quitter").pack()
        self.zone_texte = tk.Text(self.fenetre, height=6)
        self.zone_texte.pack(fill="x")

        # Touches clavier
        self.fenetre.bind("<Escape>", lambda e: self.fenetre.destroy())
        self.fenetre.bind("<KeyPress-p>", lambda e: self.basculer_pheromones())
        self.fenetre.bind("<KeyPress-r>", lambda e: self.basculer_autres_routes())
        self.fenetre.bind("<space>", lambda e: self.basculer_pause())

        self.dessiner()

    # --- Ce que l'algorithme appelle -------------------------------------
    def ecrire(self, message):
        """Ajoute une ligne dans la zone de texte."""
        self.zone_texte.insert("end", message + "\n")
        self.zone_texte.see("end")

    def mettre_a_jour(self, meilleure_route=None, autres_routes=None, pheromones=None, message=None):
        """Les fourmis donnent leurs nouveaux résultats, on redessine tout."""
        if meilleure_route is not None:
            self.meilleure_route = meilleure_route
        if autres_routes is not None:
            self.autres_routes = autres_routes
        if pheromones is not None:
            self.pheromones = pheromones
        if message:
            self.ecrire(message)
        self.dessiner()
        self.fenetre.update()

    def lancer(self):
        """Garde la fenêtre ouverte."""
        self.fenetre.mainloop()

    # --- Touches -----------------------------------------------------------
    def basculer_pheromones(self):
        self.voir_pheromones = not self.voir_pheromones
        self.dessiner()

    def basculer_autres_routes(self):
        self.voir_autres_routes = not self.voir_autres_routes
        self.dessiner()

    def basculer_pause(self):
        """Espace : met l'algorithme en pause (pour montrer le résultat) ou le relance."""
        self.en_pause = not self.en_pause
        if self.quand_pause is not None:
            self.quand_pause(self.en_pause)

    # --- Dessin (ordre important : ce qui est dessiné en dernier passe devant) ---
    def dessiner(self):
        self.zone_dessin.delete("all")
        if self.voir_pheromones and self.pheromones is not None:
            self.dessiner_pheromones()
        if self.voir_autres_routes:
            for route in self.autres_routes[:5]:
                self.dessiner_route(route, couleur="lightgray", pointilles=None)
        if self.meilleure_route is not None:
            self.dessiner_route(self.meilleure_route, couleur="blue", pointilles=(6, 4))
        self.dessiner_lieux()
        if self.meilleure_route is not None and self.avec_numeros:
            self.dessiner_ordre_visite(self.meilleure_route)

    def dessiner_lieux(self):
        """Chaque lieu = un cercle avec son numéro au milieu (le 0 en rouge)."""
        r = self.rayon
        for numero, lieu in enumerate(self.graph.liste_lieux):
            couleur = "red" if numero == 0 else "yellow"
            self.zone_dessin.create_oval(lieu.x - r, lieu.y - r, lieu.x + r, lieu.y + r,
                                         fill=couleur, outline="black")
            if self.avec_numeros:
                self.zone_dessin.create_text(lieu.x, lieu.y, text=str(numero))

    def dessiner_route(self, route, couleur, pointilles):
        """Une ligne brisée qui passe par tous les lieux de la route, dans l'ordre."""
        points = []
        for numero in route.ordre:
            lieu = self.graph.liste_lieux[numero]
            points.append(lieu.x)
            points.append(lieu.y)
        if pointilles:
            self.zone_dessin.create_line(points, fill=couleur, width=2, dash=pointilles)
        else:
            self.zone_dessin.create_line(points, fill=couleur, width=1)

    def dessiner_ordre_visite(self, route):
        """Écrit au-dessus de chaque lieu sa position de passage (0, 1, 2, ...)."""
        for position, numero in enumerate(route.ordre[:-1]):
            lieu = self.graph.liste_lieux[numero]
            self.zone_dessin.create_text(lieu.x, lieu.y - self.rayon - 8,
                                         text=str(position), fill="blue")

    def dessiner_pheromones(self):
        """Chaque arête est dessinée d'autant plus épaisse qu'elle a de phéromones."""
        n = len(self.graph.liste_lieux)
        maximum = self.pheromones.max()
        if maximum <= 0:
            return
        if n <= 1500:
            # Toutes les arêtes (i < j) d'un coup avec numpy, au lieu d'une double boucle
            i_s, j_s = np.triu_indices(n, k=1)
        else:
            # Très gros graphe : pour chaque lieu, seulement ses 2 arêtes les plus parfumées
            j_s = np.concatenate([np.argpartition(-self.pheromones[d:d + 500], 2, axis=1)[:, :2]
                                  for d in range(0, n, 500)]).ravel()
            i_s = np.repeat(np.arange(n), 2)
        forces = self.pheromones[i_s, j_s] / maximum  # entre 0 et 1
        garder = np.where(forces >= 0.05)[0]          # on oublie les traces trop faibles
        if len(garder) > 4 * n:                       # sur un gros graphe : seulement les plus fortes
            garder = garder[np.argsort(forces[garder])[-4 * n:]]
        lieux = self.graph.liste_lieux
        for k in garder:
            a, b = lieux[i_s[k]], lieux[j_s[k]]
            self.zone_dessin.create_line(a.x, a.y, b.x, b.y,
                                         fill="orange", width=1 + 6 * forces[k])


# ---------------------------------------------------------------------------
# Test : afficher un PREMIER RÉSULTAT (plus proche voisin)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    graphe = Graph(nb_lieux=NB_LIEUX)  # ou : Graph("graph_20.csv")
    premiere_route = Route.depuis_plus_proche_voisin(graphe)
    print("Premier résultat :", premiere_route)
    affichage = Affichage(graphe, titre="- plus proche voisin")
    affichage.mettre_a_jour(meilleure_route=premiere_route,
                            message=f"Plus proche voisin : distance = {premiere_route.distance:.2f}")
    affichage.ecrire(f"Ordre de visite : {premiere_route.ordre}")
    affichage.lancer()