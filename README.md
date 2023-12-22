# app_credit_scoring
*Projet développé dans le cadre de la formation Data Scientist OC (RNCP niveau 7)*

[lien vers le repository principal de ce projet](https://github.com/estellec18/modele_de_scoring)

Ce repository recense les fichiers nécessaires au déploiement de l'API :
- [main.py](main.py) : fichier python API
- [test_main.py](test_main.py) : fichier de test de l'API
- [requirements.txt](requirements.txt), [Procfile](Procfile), [runtime.txt](runtime.txt) : fichiers indispensables pour le déploiement via Heroku

Dans le dossier data se trouvent les fichiers necessaires au backend de l'API:
- un subset des données clients soumis au modèle pour tester l'API [sub_test.csv](data/sub_test.csv)
- une définition des différentes features du modèle [features.csv](data/features.csv)
- le modèle entrainé [best_xgb_1_2.joblib](data/best_xgb_1_2.joblib)
- l'explainer shap du modèle [explainer_xgb_1_2.joblib](data/explainer_xgb_1_2.joblib)

#

Pour rappel, dans le [repository principal](https://github.com/estellec18/modele_de_scoring) nous avons :
* pris connaissance des données (01_EDA.ipynb)
* entrainé un modèle de classification et analysé les features qui contribuent le plus à ce modèle (02_Modelisation.ipynb)
* analysé le data drift du modèle mis en production (data_drift_report.html)

Le repository principal comprend également l'interface de test Streamlit (frontend.py) de l'API.

Fonctionnement de l'interface :
- l'utilisateur choisit (dans une liste déroulante) le numéro du client dont il souhaite connaitre les résultats
- l'utilisateur clique sur le bouton "Prédiction" pour générer :
    - des informations générales sur le client en question (sexe, revenue, occupation...)
    - la probabilité de défault du client ainsi que sa classe (accepté ou refusé)
    - la visualisation du score du client sur une jauge
    - des informations concernant les principales features responsables du score et le positionnement du client par rapport au reste de la population





