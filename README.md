# Classificazione articoli BBC — NLP

Progetto universitario di NLP (SUPSI) articolato in **due step**:

- **Step 1 — Baseline:** scraping binario da BBC (*sport* vs *non_sport*, 2000 articoli) e classificatore **rule-based** basato su keyword discriminanti.
- **Step 2 — Estensione completa:** scraping multi-classe su **9 categorie** (7049 articoli) e confronto di **7 modelli** di classificazione (da rule-based a BERT fine-tuned), con cross-validation 5-fold e demo Streamlit interattiva.

Lo Step 1 fissa il punto di partenza (lavoro semplice, un solo modello, problema binario); lo Step 2 rappresenta l'avanzamento sostanziale (più classi, più modelli, validazione robusta, interfaccia utente).

---

## Struttura del repository

```
.
├── Step1/                          # Baseline binaria (sport / non_sport)
│   ├── Task1.ipynb                 # scraper BBC binario
│   ├── Task2.ipynb                 # classificatore rule-based
│   └── bbc_dataset.csv             # dataset 2000 articoli (1000 sport + 1000 non_sport)
│
├── Step2/                          # Progetto completo 9 classi + 7 modelli
│   ├── Scraping.ipynb              # scraper BBC multi-classe
│   ├── Task2.ipynb                 # training di tutti i 7 modelli + salvataggio
│   ├── app.py                      # demo Streamlit (tutti i 7 modelli in tempo reale)
│   ├── bbc_dataset_9classi.csv     # dataset prodotto (7049 righe)
│   ├── models/                     # artefatti salvati dal notebook (gitignored)
│   ├── output/                     # confusion matrix + PCA generate
│   └── requirements.txt
│
├── .gitignore
└── README.md
```

---

## Step 1 — Baseline binaria

### Dataset
| Classe | Articoli |
|---|---|
| `sport` | 1000 |
| `non_sport` | 1000 |
| **Totale** | **2000** |

### Approccio
Un solo classificatore **rule-based**:
1. Tokenizzazione con `RegexpTokenizer(\w+)`, lowercase, rimozione stopwords NLTK.
2. Split 80/20 stratificato (`random_state=42`).
3. Calcolo delle frequenze di classe con `nltk.FreqDist` **solo sul train**.
4. Selezione delle keyword: parola `sport` se ≥85% delle occorrenze sono in articoli sportivi, `non_sport` se ≤15%, con frequenza minima totale = 30.
5. Predizione = classe con il maggior numero di keyword presenti nel testo (default `non_sport` in caso di pareggio).

### Risultato
| Modello | F1 |
|---|---|
| Rule-Based (271 kw sport / 1241 kw non_sport) | **0.96** |

Asimmetria: `non_sport` recall = 0.99, `sport` recall = 0.93 (effetto del tie-break che favorisce la classe di default).

### Come eseguire
```bash
cd Step1
jupyter notebook Task1.ipynb   # (opzionale) rigenera il CSV — ~1 ora
jupyter notebook Task2.ipynb   # allena e valuta il rule-based
```

---

## Step 2 — Multi-classe con 7 modelli

### Dataset — 9 classi

| Classe | Articoli |
|---|---|
| `football` | 1000 |
| `cricket` | 1000 |
| `rugby` | 1000 |
| `tennis` | 1000 |
| `other_sport` | 1000 |
| `non_sport` | 1000 |
| `formula1` | 462 |
| `american_football` | 333 |
| `golf` | 254 |
| **Totale** | **7049** |

Le ultime tre classi sono sotto-rappresentate perché BBC pubblica meno articoli in quelle sezioni (il crawler si ferma quando la coda si esaurisce).

### Come eseguire

#### 1. Installazione
```bash
cd Step2
pip install -r requirements.txt
```

I dataset di NLTK (`stopwords`) vengono scaricati automaticamente dai notebook.
Gli embeddings GloVe (300d, ~1 GB) vengono scaricati da `gensim` alla prima esecuzione.

#### 2. Scraping (opzionale — il CSV è già incluso)
Il run completo impiega ~2-3 ore per rispetto del rate limit educato verso BBC.

```bash
jupyter notebook Scraping.ipynb
```

#### 3. Classificazione
```bash
jupyter notebook Task2.ipynb
```

#### 4. Demo interattiva (Streamlit)

Dopo aver eseguito `Task2.ipynb` end-to-end, l'ultima cella («Export modelli per Streamlit») salva tutti i 7 modelli in `Step2/models/`. A quel punto:

```bash
streamlit run app.py
```

L'app carica i modelli salvati e mostra, per ogni testo incollato, la predizione di tutti e 7 i modelli, la confidenza, la distribuzione di probabilità completa sulle 9 classi e un indicatore di consenso tra modelli.

### Risultati

#### Single split 80/20

| Modello | Accuracy | F1 macro |
|---|---|---|
| Baseline Rule-Based | 0.9241 | 0.9275 |
| Logistic Regression (TF-IDF) | 0.9801 | 0.9784 |
| NN Feedforward — SGD (TF-IDF) | 0.7908 | 0.5729 |
| **NN Feedforward — Adam (TF-IDF)** | **0.9887** | **0.9883** |
| Word2Vec (GloVe 300d, IDF-weighted) + FFN | 0.9745 | 0.9751 |
| Sentence Transformer (chunked) + FFN | 0.9688 | 0.9702 |
| BERT (DistilBERT fine-tuned, 256 tok, 3 ep) | 0.9794 | 0.9797 |

#### Cross-Validation 5-fold (Stratified)

| Modello | Accuracy | F1 macro |
|---|---|---|
| Logistic Regression (TF-IDF) | 0.9823 ± 0.0022 | 0.9809 ± 0.0032 |
| **NN Feedforward + Adam (TF-IDF)** | **0.9898 ± 0.0017** | **0.9905 ± 0.0028** |

Lo std basso (<0.005) conferma che il ranking dei modelli è stabile e non artefatto della singola split. BERT, W2V e ST sono esclusi dalla CV per costo computazionale (re-encoding / re-fine-tuning su ogni fold).

### Metodologia

- **Split** 80/20 con `random_state=42` e `stratify=label` (stesso split per tutti i modelli per confronto equo)
- **Tokenizzazione**: `RegexpTokenizer(\w+)` + lowercase + rimozione stopwords NLTK + filtro token non alfabetici / len ≤ 2
- **Feature extraction**: le frequenze per classe e le keyword sono calcolate **solo sul train set** per evitare data leakage
- **Metriche**: `accuracy` e `F1 macro` (non weighted) per penalizzare gli errori sulle classi minori

### Note tecniche

- **GloVe IDF-weighted**: il peso IDF (calcolato sul train) viene usato per fare la media pesata dei vettori GloVe. Le parole rare e discriminanti (es. nomi propri: *Raducanu*, *Benfica*) contano più di quelle generiche.
- **Sentence Transformer chunked**: `all-MiniLM-L6-v2` tronca a 256 token (~200 parole). Per articoli più lunghi viene fatto il chunking in pezzi da 200 parole, encoding di ciascuno, poi media degli embedding.
- **BERT fine-tuned**: `distilbert-base-uncased` (66M parametri) con classification head a 9 classi. Input troncato a 256 token, training 3 epoche, AdamW lr=2e-5, weight_decay=0.01. Su Apple Silicon (MPS) l'addestramento richiede ~5 minuti.
- **Early stopping**: per le NN, training interrotto se la loss non migliora per 7 epoche consecutive (con restore dei best weights).
- **Reproducibility**: `torch.manual_seed(42)` e `random_state=42` su tutti gli split per risultati consistenti tra run.

---

## Evoluzione Step 1 → Step 2

| Aspetto | Step 1 | Step 2 |
|---|---|---|
| Problema | binario (2 classi) | multi-classe (9 classi) |
| Dataset | 2 000 articoli | 7 049 articoli |
| Modelli | 1 (rule-based) | 7 (rule-based, LogReg, 2×NN, GloVe, ST, BERT) |
| Validazione | single split 80/20 | single split + CV 5-fold |
| Demo | — | app Streamlit interattiva |
| F1 migliore | 0.96 (rule-based) | 0.9905 (NN Adam, CV 5-fold) |
