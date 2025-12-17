# 🚀 LightRAG FastAPI Dashboard

Dit project implementeert een Retrieval-Augmented Generation (RAG)-systeem, speciaal ontworpen voor het bevragen van grote, gechunkte regelsets (zoals Dungeons & Dragons) via een webdashboard. Het systeem maakt gebruik van FastAPI voor de web-API, Gradio voor een admin-uploadinterface, en simuleert `lightrag.llm.ollama` voor het inbedden van de data.

## 🛠️ Architectuur & Componenten

Het systeem bestaat uit twee hoofdcomponenten:

### 1. RAG Ingestion (`Ragify` Klasse)

Verantwoordelijk voor het voorbereiden en inbedden van de data.

* **Chunking:** Leest bronbestanden (`ruleset.md`) en splitst deze in kleinere, beheersbare brokken op basis van scheidingstekens (`---`).
* **Filtering:** Gebruikt de functie `is_dnd_chunk` om alleen relevante (D&D-gerelateerde) tekstbrokken te behouden.
* **Embedding:** Gebruikt een gesimuleerde `ollama_embed` (die in een echte implementatie de Ollama-API zou aanroepen) om vector-embeddings te genereren.
* **Opslag:** Slaat de embeddings op als NumPy-bestanden (`.npy`) en de bijbehorende metadata (zoals de bronlocatie) als JSON Lines-bestanden (`.jsonl`) in de `./<project>/<version>/rag_storage/` map.

### 2. RAG Query Dashboard (FastAPI & `RagQuery` Klasse)

Verantwoordelijk voor de gebruikersinterface, authenticatie en het ophalen van context.

* **Authenticatie:** Beveiligd met een eenvoudige login/sessie-middleware.
* **Vector Store Lading:** De `RagQuery` klasse laadt bij initialisatie alle `.npy` (embeddings) en `.jsonl` (metadata) bestanden van de geselecteerde versie in het geheugen.
* **Querying:**
1. De gebruikersvraag wordt ingebed via `ollama_embed`.
2. De cosinus-similariteit wordt berekend tussen de query-vector en alle opgeslagen data-vectoren.
3. De *top-k* meest vergelijkbare vectoren worden geselecteerd.
4. De bijbehorende tekst uit de bronbestanden (`source_data.txt` in de simulatie) wordt opgehaald en gecombineerd tot de RAG-context.
5. Deze context wordt weergegeven in het dashboard.


* **Admin Upload (Gradio):** Een aparte interface (`/gradio_mounted`), alleen toegankelijk voor **admins**, maakt het mogelijk om nieuwe RAG-artefacten (de `rag_storage` map) te uploaden naar een nieuwe of bestaande projectversie.

---

## ⚙️ Installatie en Opstarten

Dit project vereist Python en enkele bibliotheken.

### Vereisten

* Python 3.8+
* `uvicorn`
* `fastapi`
* `python-multipart` (voor form data)
* `uvicorn[standard]`
* `gradio`
* `numpy`
* `tqdm` (alleen in de Ragify-klasse)
* `lightrag` (gesimuleerd, maar vereist voor de echte implementatie)
* `torch` (geïmporteerd maar niet gebruikt in de gesimuleerde code)
* `scikit-learn` (voor `cosine_similarity`)

### Stap-voor-stap installatie

1. **Clone de repository** (indien van toepassing) en navigeer naar de projectmap.
2. **Maak een virtuele omgeving** (aanbevolen):
```bash
python -m venv venv
source venv/bin/activate  # Op Linux/macOS
venv\Scripts\activate     # Op Windows

```


3. **Installeer de afhankelijkheden:**
```bash
pip install fastapi uvicorn[standard] python-multipart gradio numpy scikit-learn

```


*(Opmerking: De gesimuleerde code bevat ook imports zoals `lightrag` en `_tokenizer`, die in een echte omgeving apart geïnstalleerd moeten worden.)*

### Applicatie Starten

Voer het script uit om de FastAPI/Uvicorn server te starten:

```bash
python your_script_name.py # Vervang your_script_name.py door de naam van je bestand

```

De applicatie zal standaard starten op: `http://0.0.0.0:80`.

---

## 🖥️ Gebruik

### 1. Inloggen

Open de applicatie in uw browser (bijv. `http://localhost`).

| Gebruikersnaam | Wachtwoord | Rol | Opmerkingen |
| --- | --- | --- | --- |
| **nigel** | `1504_Nigel` | `admin` | Heeft toegang tot het Gradio uploadscherm. |
| **simon** | `simon-x` | `user` | Standaard gebruiker. |

### 2. Dashboard Query (`/dashboard`)

Nadat u bent ingelogd, kunt u:

1. **Selecteer Project:** Kies een top-level map (bv. `DandD`, `test`).
2. **Selecteer Versie:** Kies de versie-submap (bv. `version1`, `versionX`) binnen het project. **Belangrijk:** Alleen mappen die een geldige `rag_storage` met data bevatten, worden getoond.
3. **Stel een Vraag:** Voer uw zoekopdracht in (bv. "Cleric spell slots").
4. **Resultaat:** Het systeem zal de meest relevante contextbrokken uit de vector store ophalen, de originele tekst uit de bronbestanden laden, en deze als de **LightRAG System Output** tonen.

### 3. Admin Upload (`/gradio`)

Alleen voor de `admin` rol:

1. Ga naar `/gradio` (of klik op de link in het dashboard).
2. **Upload Directory:** Upload een map die de **RAG-artefacten** bevat (de `.jsonl` meta-bestanden en de `vector_store` map met de `.npy` bestanden).
3. **Project Naam/Versie Folder Naam:** Voer het doelproject en de versiemap in (bv. `MyProject` en `v2`).
4. De bestanden worden gekopieerd naar `./<Project Naam>/<Versie Folder>/rag_storage/`, waardoor ze onmiddellijk beschikbaar zijn via het dashboard.

### 4. Data Structuur

Het systeem vereist een specifieke bestandsstructuur:

```
.
├── DandD/
│   ├── version1/
│   │   ├── ruleset.md (Originele bron)
│   │   ├── source_data.txt (Gesplitste bron, gebruikt door RagQuery)
│   │   └── rag_storage/
│   │       ├── meta_0.jsonl (Metadata over de chunks)
│   │       └── vector_store/
│   │           └── vec_0.npy (De vector embeddings)
└── test/
    └── version_a/
        └── ...

```