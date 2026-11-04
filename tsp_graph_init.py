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

######### Fonctions diverses ##########
def nettoyer_cle(cle, recherche):
    ## Fonction qui retourne l'élement inverse à la recherche dans les chemins
    elements = cle.split("-")
    return elements[1] if elements[0] == recherche else elements[0]

######### Classe Lieu #############
class Lieu:
    def __init__(self, x, y, nom):
        self.x = x
        self.y = y
        self.nom = nom
        
    def calcul_distance(Lieu1, Lieu2):
        return float(np.hypot(Lieu1.x - Lieu2.x, Lieu1.y - Lieu2.y))
        # delta = np.array([Lieu1.x, Lieu1.y]) - np.array([Lieu2.x, Lieu2.y])
        # return np.sqrt(np.dot(delta, delta))

############ Class Graph ############
class Graph:
    def __init__(self, liste_lieux):
        self.liste_lieux = liste_lieux
        
    def calcul_matrice_cout_od(self):
        matrice_od = {} ## Dans un dictionnaire pour faire l'association chemins - distance
        delta = 0 ## Delta permet d'éviter de relire les mêmes chemins
        for position1 in self.liste_lieux:
            delta += 1
            for position2 in self.liste_lieux[delta:]: ## Pour la deuxième position, il faut prendre la position qui est à l'indice suivant (donc on supprime les delta-ème indice précédente)
                matrice_od[f"{position1.nom}-{position2.nom}"] = Lieu.calcul_distance(position1, position2) ## Format : "1-2" : 5

        self.matrice_od = matrice_od

    def plus_proche_voisin(self, nom):
        plus_proche = None
        distance_minimale = float("inf")

        for cle, valeur in self.matrice_od.items():
            noeud1, noeud2 = cle.split("-")
            if str(nom) == noeud1 or str(nom) == noeud2: ## Si le nom du lieu est dans la clé
                if valeur < distance_minimale: ## On regarde sa distance
                    distance_minimale = valeur
                    plus_proche = cle

        plus_proche = nettoyer_cle(plus_proche, nom) 
        return plus_proche, distance_minimale       

    def charger_graph(chemin_fichier, NB_LIEUX):
        return pd.read_csv(f'{chemin_fichier}/graph_{NB_LIEUX}.csv')

    def calcul_distance_route(self, ordre):
        distance = 0
        for i in range(len(ordre)-1):
            pos1 = ordre[i]
            pos2 = ordre[i+1]
            if pos2 == 0:
                continue
            if int(pos1) > int(pos2):
                distance += self.matrice_od[f"{pos2}-{pos1}"]
            else:
                distance += self.matrice_od[f"{pos1}-{pos2}"]

        return distance
            

############ Class Route ############
class Route:
    def __init__(self, ordre):
        self.ordre = ordre


############ Class Affichage ############
class Affichage():
    def __init__(self, largeur, hauteur):
        self.hauteur = hauteur
        self.largeur = largeur
        self.fenetre = None
        self.label = None
        self.canvas = None

    def creer_fenetre(self):
        self.fenetre = Tk()
        self.fenetre.title(NOM_GROUPE)
        self.canvas = Canvas(self.fenetre, width=LARGEUR, height=HAUTEUR)
        self.canvas.pack()
        self.label = Label(self.fenetre, text="Texte par défaut")
        self.label.pack()
        self.fenetre.mainloop()

    def modifier_texte(self, nouveau_texte):
        if self.label:
            self.label.config(text=str(nouveau_texte))

    ## TODO: Afficher et modifier le graphe
    ## TODO: Gérer les colorimétries et nombre de phéromones
    ## TODO: Gérer les différents affichages
