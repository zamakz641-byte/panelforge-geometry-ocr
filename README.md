# PanelForge Geometry + OCR

Outil local d'extraction de cases de webtoon avec détection géométrique et protection optionnelle des zones de texte par OCR.

## Fonctionnalités

- import d'une archive CBZ ou d'un dossier d'images ;
- détection rapide des limites de cases ;
- séparation des régions verticales fusionnées ;
- préservation des bulles et légendes proches ;
- interface locale PyWebView ;
- mode géométrique utilisable sans OCR.

## Installation et lancement

Sous Windows, double-cliquez sur `run.bat`. Le script prépare automatiquement l'environnement virtuel.

Installation manuelle :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

## Architecture

- `detector.py` : détection géométrique ;
- `text_detector.py` : détection des zones textuelles ;
- `engine.py` : orchestration du découpage ;
- `image_io.py` : lecture et écriture des images ;
- `ui` : interface web locale.

## Modèle OCR

Le détecteur de texte PP-OCRv4 mobile est téléchargé au premier usage et mis en cache dans `models/`. Aucun modèle IA n'est stocké dans ce dépôt.
