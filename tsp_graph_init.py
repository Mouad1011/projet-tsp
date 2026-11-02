###### Bibliothèques et variables #######
import numpy as np
import random
from time import time
import pandas as pd
from tkinter import * 
import csv

LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 5
NOM_GROUPE = "GROUPE_X" # A définir

######### Fonctions diverses ##########
def nettoyer_cle(cle, recherche):
    elements = cle.slip("-")
    return elements[1] if elements[0] == recherche else elements[0]

######### Classe Lieu #############
class Lieu:
    def __init__(self, x, y, nom):
        self.x = x
        self.y = y
        self.nom = nom
        
    def calcul_distance(Lieu1, Lieu2):
        delta = np.array([Lieu1.x, Lieu1.y]) - np.array([Lieu2.x, Lieu2.y])
        return np.sqrt(np.dot(delta, delta))

############ Class Graph ############
class Graph:
    def __init__(self, liste_lieux):
        self.liste_lieux = liste_lieux
        
    def calcul_matrice_cout_od(self):
        matrice_od = {}
        delta = 0
        for position1 in self.liste_lieux:
            delta += 1
        for position2 in self.liste_lieux[delta:]:
            matrice_od[f"{position1.nom}-{position2.nom}"] = Lieu.calcul_distance(position1, position2)

        self.matrice_od = matrice_od

    def plus_proche_voisin(self, nom):
        plus_proche = None
        distance_minimale = float("inf")

        for cle, valeur in self.matrice_od.items():
            if nom in cle:
                if valeur < distance_minimale:
                    distance_minimale = valeur
                    plus_proche = cle

        plus_proche = nettoyer_cle(plus_proche, nom) 
        return plus_proche, distance_minimale       

    def charger_graph(chemin_fichier):
        with open(f'{chemin_fichier}/graph_.{NB_LIEUX}csv', 'r', encoding='utf-8') as fichier:
            return csv.DictReader(fichier)

    def calcul_distance_route(self, ordre):
        distance = 0
        for i in range(len(ordre)):
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
    def __init__(self, hauteur, largeur):
        self.hauteur = hauteur
        self.largeur = largeur

    def creer_fenetre():
        fenetre = Tk()
        fenetre.title(NOM_GROUPE)
        canvas = Canvas(fenetre, width=LARGEUR, height=HAUTEUR)
        canvas.pack()
        label = Label(fenetre, text="Texte par défaut")
        label.pack()
        fenetre.mainloop()
