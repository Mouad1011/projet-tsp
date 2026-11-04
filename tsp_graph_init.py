###### Bibliothèques et variables #######
import numpy as np
import random
from time import time
import pandas as pd
from tkinter import * 
import csv

LARGEUR = 800
HAUTEUR = 600
NOM_GROUPE = "GROUPE_X" # A définir
NB_LIEUX = 20
RAYON = 12

######### Fonctions diverses ##########
def nettoyer_cle(cle, recherche):
    ## Fonction qui retourne l'élement inverse à la recherche dans les chemins
    elements = cle.split("-")
    return elements[1] if elements[0] == recherche else elements[0]

######### Classe Lieu #############
class Lieu:
    def __init__(self, x, y, nom):
        self.x = float(x)
        self.y = float(y)
        self.nom = str(nom)
        
    def calcul_distance(Lieu1, Lieu2):
        return float(np.hypot(Lieu1.x - Lieu2.x, Lieu1.y - Lieu2.y))
        # delta = np.array([Lieu1.x, Lieu1.y]) - np.array([Lieu2.x, Lieu2.y])
        # return np.sqrt(np.dot(delta, delta))

############ Class Graph ############
class Graph:
    def __init__(self, fichier_csv=None, nb_lieux=NB_LIEUX):
        self.liste_lieux = []
        self.matrice_od = None
        if fichier_csv is not None:
            self.charger_graph(fichier_csv) # lieux lus dans un CSV
        # On calcule toutes les distances UNE SEULE FOIS, ici.
        self.calcul_matrice_cout_od()


    def calcul_matrice_cout_od(self):
        """Remplit matrice_od : matrice_od[i][j] = distance entre le lieu i et le lieu j."""
        n = len(self.liste_lieux)
        self.matrice_od = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.matrice_od[i][j] = self.liste_lieux[i].calcul_distance(self.liste_lieux[j])
        return self.matrice_od

    def plus_proche_voisin(self, lieu, deja_visites):
        """
        Renvoie le numéro du lieu NON visité le plus proche de `lieu`,
        en lisant la matrice des distances (aucun calcul de distance ici).
        """
        meilleur_voisin = None
        meilleure_distance = float("inf") # "infini" au départ

        for autre in range(len(self.liste_lieux)):
            if autre == lieu or autre in deja_visites:
                continue # on saute le lieu lui-même et les visités
            distance = self.matrice_od[lieu][autre]
            if distance < meilleure_distance:
                meilleure_distance = distance
                meilleur_voisin = autre

        return meilleur_voisin    

    def charger_graph(self, chemin_fichier):
        """Lit un CSV avec une colonne x et une colonne y (une ligne = un lieu)."""
        self.liste_lieux = []
        with open(chemin_fichier, "r", encoding="utf-8") as fichier:
            lecteur = csv.DictReader(fichier) # lit chaque ligne comme {"x": ..., "y": ...}
            for numero, ligne in enumerate(lecteur):
                self.liste_lieux.append(Lieu(ligne["x"], ligne["y"], str(numero)))

    def calcul_distance_route(self, route):
        """Coût d'une route = somme des distances de chaque étape."""
        total = 0
        for k in range(len(route.ordre) - 1):
            depart = route.ordre[k]
            arrivee = route.ordre[k + 1]
            total += self.matrice_od[depart][arrivee]

        return total
            

############ Class Route ############
class Route:
    """
    Un trajet. ordre = liste des numéros de lieux visités,
    qui commence ET finit par 0. Exemple : [0, 3, 1, 2, 0]
    """
    def __init__(self, ordre):
        self.ordre = list(ordre)
        self.distance = None # le coût, calculé par le Graph
        
    @staticmethod
    def depuis_plus_proche_voisin(graph):
        """
        HEURISTIQUE DE DÉPART : depuis où je suis, je vais toujours
        au lieu non visité le plus proche. Donne une première route rapidement.
        """
        ordre = [0] # on part du lieu 0
        deja_visites = {0}
        lieu_actuel = 0
        while len(ordre) < len(graph.liste_lieux):
            suivant = graph.plus_proche_voisin(lieu_actuel, deja_visites)
            ordre.append(suivant)
            deja_visites.add(suivant)
            lieu_actuel = suivant

        ordre.append(0) # retour au point de départ
        route = Route(ordre)
        route.distance = graph.calcul_distance_route(route)
        return route

    def __repr__(self):
        return f"Route(distance={self.distance:.2f}, ordre={self.ordre})"


############ Class Affichage ############
class Affichage():
    def __init__(self, graph, titre=""):
        self.graph = graph
        self.meilleur_route = None
        self.autres_routes = []
        self.pheromones = None
        self.voir_pheromones = False
        self.voir_autres_routes = False
        self.fenetre = Tk()
        self.fenetre.title(f"{NOM_GROUPE} - Voyageur de commerce {titre}")
        self.zone_dessin = Canvas(self.fenetre, width=LARGEUR, height=HAUTEUR, bg="white")
        self.zone_dessin.pack()
        Label(self.fenetre, text="[P] phéromones [R] meilleures routes [Échap] quitter").pack()
        self.zone_texte = Text(self.fenetre, height=6)
        self.zone_texte.pack(fill="x")

        self.fenetre.bind("<Escape>", lambda e: self.fenetre.destroy())
        self.fenetre.bind("<KeyPress-p>", lambda e: self.basculer_pheromones())
        self.fenetre.bind("<KeyPress-r>", lambda e: self.basculer_autres_routes())
        self.dessiner()

    def ecrire(self, message):
        """Ajoute une ligne dans la zone de texte"""
        self.zone_texte.insert("end", message + "\n")
        self.zone_texte.see("end")

    def mettre_a_jour(self, meilleure_route=None, autres_routes=None, pheromones=None, message=None):
        """Les fourmis donnent leurs nouveaux résultats, on redissine toute."""
        if meilleure_route is not None:
            self.meilleur_route = meilleure_route
        if autres_routes is not None:
            self.autres_routes = autres_routes
        if pheromones is not None:
            self.pheromones = pheromones
        if message:
            self.ecrire(message)
        self.dessiner()
        self.fenetre.update()
    
    def lancer(self):
        """Garde la fen^tre ouverte"""
        self.fenetre.mainloop()

    def basculer_pheromones(self):
        self.voir_pheromones = not self.voir_pheromones
        self.dessiner()

    def basculer_autres_routes(self):
        self.voir_autres_routes = not self.voir_autres_routes
        self.dessiner()

    def dessiner(self):
        self.zone_dessin.delete("all")
        if self.voir_pheromones and self.pheromones is not None:
            self.dessiner_pheromones()
        if self.voir_autres_routes:
            for route in self.autres_routes[:5]:
                self.dessiner_route(route, couleur="lightgray", pointilles=None)
        if self.meilleur_route is not None:
            self.dessiner_route(self.meilleur_route, couleur="blue", pointilles=(6, 4))
        self.dessiner_lieux()
        if self.meilleur_route is not None:
            self.dessiner_ordre_visite(self.meilleur_route)
        
    def dessiner_lieux(self):
        """Chaque lieu = un cercle avec son numéro au milieu (le 0 en rouge)."""
        for numero, lieu in enumerate(self.graph.liste_lieux):
            couleur = "red" if numero == 0 else "yellow"
            self.zone_dessin.create_oval(lieu.x - RAYON, lieu.y - RAYON, lieu.x + RAYON, lieu.y + RAYON, fill=couleur, outline="black")
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
            self.zone_dessin.create_text(lieu.x, lieu.y - RAYON - 8, text=str(position), fill="blue")
    
    def dessiner_pheromones(self):
        """Chaque arête est dessinée d'autant plus épaisse qu'elle a de phéromones."""
        n = len(self.graph.liste_lieux)
        maximum = self.pheromones.max()
        if maximum <= 0:
            return
        for i in range(n):
            for j in range(i + 1, n):
                force = self.pheromones[i][j] / maximum # entre 0 et 1
                if force < 0.05:
                    continue # on ne dessine pas les traces trop faibles
                a, b = self.graph.liste_lieux[i], self.graph.liste_lieux[j]
                self.zone_dessin.create_line(a.x, a.y, b.x, b.y, fill="orange", width=1 + 6 * force)