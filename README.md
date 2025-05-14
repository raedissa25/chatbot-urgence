Ce chatbot a été développé dans le cadre de mon Projet de Fin d'Études pour assister le personnel médical dans l’analyse rapide et automatisée des signaux ECG à 2 dérivations, un format couramment utilisé dans les contextes d’urgence (ambulance, soins intensifs, etc.).

⚙️ Fonctionnalités :

📂 Upload de signaux ECG au format CSV

🧠 Analyse intelligente du signal à l’aide d’un modèle CNN 1D pré-entraîné pour détecter des classes telles que Normal, VEB/PVC, SVEB, Fusion Beat, Unknown Beat

📊 Visualisation du signal ECG directement dans l’interface

🧾 Génération automatique d’un rapport d’interprétation structuré

💬 Intégration d’un assistant IA médical (via Gemini) pour répondre à des questions cliniques ou d’analyse ECG approfondie

📄 Téléchargement du rapport d’analyse ECG

📁 Fichiers utilisés dans ce chatbot :

app.py : Script principal de l’application Streamlit

CNN1D model.h5 : Modèle CNN 1D entraîné pour la classification de signaux ECG 2 leads

.env : À créer manuellement. Ce fichier doit être placé dans le même dossier que app.py et CNN 1D model.h5, et doit contenir votre clé API Gemini :

GEMINI_API_KEY="ton api GEMINI"

requirements.txt : Liste complète des bibliothèques nécessaires à l'exécution du projet

⚠️ Ce chatbot est destiné exclusivement au personnel médical formé, afin de garantir une interprétation fiable des résultats fournis.

Réalisé par : Raed Ben Aissa

Encadré par : LT COL Mohamed Hachemi Jeridi et DR Mouna Azaiz
