###### Bibliothèques et variables #######
import numpy as np
import random
from time import time
import pandas as pd
import tkinter
import csv

LARGEUR = 800
HAUTEUR = 600
NB_LIEUX = 5

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
      for position2 in self.liste_lieux[1:]:
        matrice_od[f"{position1.nom}-{position2.nom}"] = Lieu.calcul_distance(position1, position2)

    self.matrice_od = matrice_od


  def charger_graph(chemin_fichier):
    with open(f'graph_.{NB_LIEUX}csv', 'r', encoding='utf-8') as fichier:
      return csv.DictReader(fichier)