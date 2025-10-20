# Système de recommandation de films hybride (Streamlit + Python)
## 1 Objectif du projet

Le but du projet est de recommander des films à un utilisateur en se basant sur :

le comportement des autres utilisateurs (filtrage collaboratif)
 
le contenu des films (genres)

L’application Streamlit permet :

de générer des recommandations personnalisées selon un utilisateur,

de rechercher des films similaires à un titre donné.

## 2 Données utilisées

Dataset : MovieLens (jeu de données public pour les systèmes de recommandation).


 Fichier      Description                            Colonnes                                  

 ratings.dat  Notes attribuées par les utilisateurs  UserID, MovieID, Rating, Timestamp        
 movies.dat   Informations sur les films             MovieID, Title, Genres                    
 users.dat    Profil utilisateur                     UserID, Gender, Age, Occupation, Zip-code 

 ## 3 Méthodologie technique
 Étape 1 : Chargement et préparation

Lecture des fichiers avec pandas.read_csv().

Fusion des tables ratings, users, movies.

Nettoyage des titres et encodage des genres (get_dummies()).

 Étape 2 : Construction des matrices

Matrice utilisateur-film : lignes = utilisateurs, colonnes = films, valeurs = notes.

Similarités : calculées avec cosine_similarity() pour les utilisateurs et les films.

 Étape 3 : Système de recommandation

Collaboratif : moyenne pondérée des notes d’utilisateurs similaires.

Basé sur le contenu : similarité des genres des films aimés.

Hybride : combinaison pondérée : score_Hybride = α × Score_collaboratif​+ (1−α) × Score_contenu​ avec α réglable.

Étape 4 : Interface Streamlit

Deux onglets :

Recommandation personnalisée : choix d’un utilisateur + ajustement de α.

Films similaires : recherche par titre.
## 4 Technologies utilisées
 Catégorie  ---->        = Outils / Librairies              
 Manipulation de données = pandas, numpy                    
 Machine Learning        = scikit-learn (cosine_similarity) 
 Interface web           = Streamlit                        
 Visualisation           = Power BI, matplotlib             
 Style                   = CSS intégré (st.markdown)    

 ## 5 Avantages du modèle hybride
 
  Avantage                       Explication                                         
 
  Plus robuste                  Combine deux approches complémentaires              
  Gère la cold start partielle  Le modèle “contenu” compense l’absence d’historique 
  Interactif                    Ajustement du paramètre α en direct                 
  Visualisable                  Résultats exploitables dans Power BI                


## 6 Limites

Pas d’images/synopsis (MovieLens limité).

Problème de “cold start” complet (nouvel utilisateur).

Données anciennes.

## 7 Rôles de Power BI dans le projet

Power BI a été utilisé pour valoriser et interpréter les résultats du modèle.
Les données exportées depuis Python/Streamlit ont permis de créer plusieurs tableaux de bord analytiques.

 Rôle 1 — Analyse des données initiales

Statistiques descriptives des utilisateurs (âge, genre, occupation).

Histogrammes des notes (Rating distribution).

Répartition des genres de films.

 Rôle 2 — Visualisation de la similarité

Heatmap (carte de chaleur) entre utilisateurs pour visualiser leurs similarités.

Graphique réseau (Force-Directed Graph) montrant les connexions entre utilisateurs proches.

Bar chart des “Top utilisateurs similaires” à un profil donné.

## Résumé

« Ce projet est un système de recommandation de films hybride combinant filtrage collaboratif et contenu.
L’utilisateur peut ajuster le paramètre α via une interface Streamlit moderne.
Les résultats sont également exportés et visualisés dans Power BI pour analyse.
Ce travail illustre mes compétences en traitement de données, machine learning et visualisation. »
