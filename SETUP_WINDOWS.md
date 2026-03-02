# Setup Windows — Video Translation Pipeline

Guide d'installation des dépendances **externes** (hors Python) nécessaires
à l'exécution complète du pipeline sur Windows 10/11.

---

## Prérequis

| Outil | Version min. | Rôle |
|-------|-------------|------|
| **PowerShell** | 5.1+ | Shell utilisé par VS Code et les tasks |
| **Python** | 3.9+ | Runtime du pipeline |
| **FFmpeg** | 6.0+ | Extraction et segmentation audio |
| **whisper.cpp** | — | Transcription audio → SRT |
| **llama.cpp** (optionnel) | — | Serveur LLM local pour traduction |

> **Shell :** toutes les commandes de ce guide et les tâches VS Code
> (`tasks.json`) utilisent **PowerShell 5.1+** (`powershell.exe`).
> Windows 10/11 l'inclut nativement (Windows PowerShell 5.1).
> Si vous avez installé **PowerShell 7+** (`pwsh.exe`), il fonctionne
> également — les cmdlets utilisés sont compatibles avec les deux versions.
>
> ```powershell
> # Vérifier la version installée
> $PSVersionTable.PSVersion
> ```

---

## 1. Python & environnement virtuel

```powershell
# Vérifier la version installée
python --version   # doit afficher >= 3.9

# Créer le venv à la racine du projet
python -m venv .venv

# Activer le venv
.\.venv\Scripts\Activate.ps1

# Installer le projet avec les dépendances dev
pip install -e ".[dev]"
```

> **Note :** VS Code détecte automatiquement `.venv/` grâce à
> `python.defaultInterpreterPath` dans `.vscode/settings.json`.

---

## 2. FFmpeg

FFmpeg est utilisé par `AudioExtractor` pour diviser les vidéos en segments audio.

### Option A — winget (recommandé)

```powershell
winget install Gyan.FFmpeg
```

### Option B — Chocolatey

```powershell
choco install ffmpeg
```

### Option C — Installation manuelle

1. Télécharger depuis <https://www.gyan.dev/ffmpeg/builds/> (build `release full`)
2. Extraire dans un dossier, par exemple `C:\Tools\ffmpeg`
3. Ajouter au `PATH` :

```powershell
# Ajouter au PATH utilisateur (permanent)
[Environment]::SetEnvironmentVariable(
    "PATH",
    "$([Environment]::GetEnvironmentVariable('PATH', 'User'));C:\Tools\ffmpeg\bin",
    "User"
)
```

### Vérification

```powershell
ffmpeg -version
# Doit afficher la version sans erreur
```

---

## 3. whisper.cpp

Le `WhisperTranscriber` appelle le binaire `main.exe` de whisper.cpp en subprocess.

### Installation

1. Télécharger un build pré-compilé depuis
   <https://github.com/ggerganov/whisper.cpp/releases>
   (choisir `whisper-*-bin-x64.zip` pour Windows 64-bit)
2. Extraire dans un dossier, par exemple `C:\Tools\whisper.cpp\`
3. Télécharger un modèle GGML :

```powershell
# Depuis le dossier whisper.cpp
cd C:\Tools\whisper.cpp
# Modèle medium (bon compromis qualité/vitesse)
Invoke-WebRequest `
    -Uri "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin" `
    -OutFile "models\ggml-medium.bin"
```

### Configuration

Dans `configs/settings.json`, renseigner les chemins :

```json
{
  "whisper": {
    "bin_path": "C:/Tools/whisper.cpp/main.exe",
    "model_path": "C:/Tools/whisper.cpp/models/ggml-medium.bin",
    "lang": "auto"
  }
}
```

> **Astuce :** Utiliser des slashs `/` dans les chemins JSON — Python et
> whisper.cpp les acceptent sur Windows.

### Vérification

```powershell
C:\Tools\whisper.cpp\main.exe --help
# Doit afficher l'aide de whisper.cpp
```

---

## 4. llama.cpp (optionnel — moteur LLM local)

Le `LlamaCPPProvider` communique avec un serveur llama.cpp local via l'API
OpenAI-compatible (`/v1/chat/completions`).

### Installation

1. Télécharger depuis <https://github.com/ggerganov/llama.cpp/releases>
   (choisir `llama-*-bin-win-*.zip`)
2. Extraire dans `C:\Tools\llama.cpp\`
3. Télécharger un modèle GGUF (exemple avec Mistral 7B) :

```powershell
Invoke-WebRequest `
    -Uri "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf" `
    -OutFile "C:\Tools\llama.cpp\models\mistral-7b.gguf"
```

### Lancement du serveur

```powershell
C:\Tools\llama.cpp\llama-server.exe `
    -m "C:\Tools\llama.cpp\models\mistral-7b.gguf" `
    -c 4096 `
    --host 127.0.0.1 `
    --port 8080
```

Le pipeline se connecte par défaut à `http://127.0.0.1:8080`.

### Vérification

```powershell
# Dans un autre terminal
Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/models"
# Doit retourner la liste des modèles chargés
```

---

## 5. Configuration du projet

Copier le template de configuration et l'adapter :

```powershell
Copy-Item configs/settings_test.json configs/settings.json
```

Éditer `configs/settings.json` avec vos chemins locaux. Structure attendue :

```json
{
  "base_working_dir": "./project_workspace",
  "whisper": {
    "bin_path": "C:/Tools/whisper.cpp/main.exe",
    "model_path": "C:/Tools/whisper.cpp/models/ggml-medium.bin",
    "lang": "auto"
  },
  "llm_config": {
    "source_lang": "English",
    "target_lang": "French",
    "chunk_size": 25,
    "prompt_file": "configs/system_prompt.txt"
  },
  "translation": {
    "source_lang": "en",
    "target_lang": "fr",
    "max_chars_batch": 2000,
    "retry_delay": 30,
    "cache_file": "data/translation_cache.json"
  },
  "technical_dictionary": {}
}
```

> `configs/settings.json` est dans `.gitignore` — chaque machine a sa propre
> config avec ses chemins locaux.

---

## 6. Vérification globale

```powershell
# Activer le venv
.\.venv\Scripts\Activate.ps1

# Vérifier les outils externes
python -c "import shutil; print('FFmpeg:', shutil.which('ffmpeg'))"
python -c "from pathlib import Path; p = Path('C:/Tools/whisper.cpp/main.exe'); print('Whisper:', p if p.exists() else 'NOT FOUND')"

# Lancer les tests
python -m pytest tests/ -v

# Lancer le quality check complet
python -m ruff check src/ tests/
python -m pylint src/modules/ src/utils/ src/main.py --rcfile=pyproject.toml
```

---

## Résumé rapide

```powershell
# One-liner setup (après avoir installé FFmpeg et whisper.cpp)
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e ".[dev]"
```
