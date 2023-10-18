![COVID](image.png)
# Application Web de Classification COVID  https://covid-detection.streamlit.app/

Il s'agit d'une application web permettant de classer les images radiographiques COVID-19 à l'aide d'un modèle d'apprentissage profond. L'application est développée en Python et Streamlit.

## Table des matières

- [Aperçu](#apercu)
- [Fonctionnalités](#fonctionnalites)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Démo](#demo)
- [Contributions](#contributions)
- [Licence](#licence)

## Aperçu

L'Application Web de Classification COVID permet aux utilisateurs de télécharger des images radiographiques et d'obtenir des prédictions quant à la présence éventuelle de COVID-19. Le modèle d'apprentissage profond utilisé pour la classification a été entraîné sur un ensemble de données d'images radiographiques.

## Fonctionnalités

- Téléchargez des images radiographiques pour la classification.
- Consultez le résultat de la classification.
- Interface conviviale.
- Détails et informations sur le modèle.

## Installation

1. Clonez le dépôt :

   ```bash
   git clone [https://github.com/LaurianeMD/COVID_Classification]
   cd Covid-classification-web-app-python-streamlit


## Créez un environnement virtuel (recommandé) :
python -m venv venv
source venv/bin/activate  # Sur Windows, utilisez `venv\Scripts\activate`

## Installez les paquets requis :
pip install -r requirements.txt

## Structure du Projet
Xray_data/ : Répertoire contenant les données.<br>
cnn.py : Script de formation du modèle.<br>
main..py : Script pour faire des prédictions avec le modèle entraîné.<br>
util.py: Fonction utilitaires


## Modèle
Un classificateur COVID a été utilisé pour classer les images de radiographie X-RAY (COVID, NORMAL). Le modèle est entraîné en utilisant une base de données de radiographies sur COVID-19 disponible sur Kaggle. Un modèle de réseau de neurones à convolution (CNN) a été utilisé.

## Données
Les données sont disponibles sur Kaggle à partir du lien : https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

## Utilisation
Pour démarrer l'application web, exécutez la commande suivante :
streamlit run main.py

Cela lancera l'application dans votre navigateur web, et vous pourrez commencer à l'utiliser.

## Démo
Vous pouvez accéder à une démo en direct de l'application web via ce lien :  https://covid-detection.streamlit.app/


## Contributions
Les contributions sont les bienvenues ! Si vous souhaitez contribuer au projet, veuillez suivre ces étapes :

-Forkez le dépôt.<br>
-Créez une nouvelle branche pour votre fonctionnalité ou correction de bogue.<br>
-Effectuez vos modifications et commitez-les.<br>
-Poussez vos modifications vers votre fork.<br>
-Créez une demande d'extraction (pull request).<br>
-Veuillez vous assurer de suivre le code de conduite du projet et de contribuer de manière respectueuse et collaborative.<br>

## Licence
Ce projet est sous licence BSD 3 - voir le fichier LICENSE pour plus de détails.


## Auteurs: 
Ali Moussa MAIGA <br>
Lauriane MBAGDJE DORENAN <br>
Carel Brian Koudous Jesuton MOUSSE <br>
Ghislain MWENEMBOKA BYAMONI <br>
Milse William NZINGOU MOUHEMBE <br>



