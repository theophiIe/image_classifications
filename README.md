# Calculs statistiques 'privacy-by-design' pour le Cloud Personnel

[![Generic badge](https://img.shields.io/badge/python-3.8.10-9cf.svg)](https://shields.io/) 
[![Generic badge](https://img.shields.io/badge/librairie-Dash-9cf.svg)](https://dash.plotly.com/) 
[![Generic badge](https://img.shields.io/badge/web-TensorFlowHub-9cf.svg)](https://tfhub.dev/s?fine-tunable=yes&module-type=image-feature-vector&network-architecture=mobilenet-v3&subtype=module,placeholder)

--------

## Pré-requis 📂
La version de Python recommandé est la `3.8.10`.

* Installez virtualenv si vous ne l'avez pas :

  ```
  pip3 install virtualenv
  ```
* Allez dans le fichier du projet

  ```
  cd chemin_vers_votre_dossier
  ```
* Créez un environnement virtuel

  ```
  python3 -m venv .venv 
  ```
* Activez l'environnement virtuel

  ```
  # Windows
  .\.venv\Scripts\activate

  # Linux / Mac
  source .venv/bin/activate
  ```
* Installez dans l'environnement les librairies requises

  ```
  pip3 install -r requirements.txt
  ```

* Créer un dossié `/img` contenant les JPEG
-------

## Exécution ⚙️

* Pour lancer les calculs de chaque modèle il faut exécuter les commandes suivantes :

  ```
  python3 vectorize_img.py
  python3 models_save.py
  python3 analyze_functions.py
  ```
* Démarrer le dashboard

  ```
  cd dashboard
  python3 app.py
  ```

Puis se rendre à l’adresse suivante (donnée dans le terminal à l'exécution): [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

## Contributeurs 👥

**Théophile Molinatti** 

**Léo Lamoureux**

**Lola Pires Pinto**

**Raphaël Lin**
