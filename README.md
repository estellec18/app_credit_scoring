# app_credit_scoring

*Projet 7 du parcours Data Science OC*

Ce repository recense les fichiers nécessaires au déploiement de l'API :
- main.py : fichier python API
- test_main.py : fichier de test de l'API
- requirements.txt, Procfile, runtime.txt : fichiers indispensables pour le déploiement via Heroku

Dans le dossier data se trouvent les fichiers necessaires au backend de l'API:
- un subset des données clients qui sont soumis au modèle (sub_test.csv)
- une définition des différentes features du modèle
- le modèle entrainé
- l'explainer shap du modèle


Pour rappel, dans le [repository principal](https://github.com/estellec18/modele_de_scoring) nous avons :
- pris connaissance des données (01_EDA.ipynb),
- entrainé un modèle de classification et analysé les features qui contribuent le plus à ce modèle (02_modelisation.ipynb)
- analysé le data drift du modèle mis en production (data_drift_report.html)

Le repository principal comprend également l'interface de test Streamlit (frontend.py) de l'API.




