# Classificazione articoli BBC — 9 classi

Progetto universitario di NLP: scraping di articoli dal sito BBC e classificazione testuale multiclasse con sette approcci (rule-based, regressione logistica, reti neurali, word embeddings, sentence transformer, BERT fine-tuned) più cross-validation 5-fold sui modelli principali.

## Struttura

```
.
├── Scraping.ipynb                # scraper BBC multi-classe
├── Task2.ipynb                   # baseline + LR + NN + W2V + ST + BERT + CV
├── bbc_dataset_9classi.csv       # dataset prodotto (7049 righe)
├── output/                       # confusion matrix + PCA generate
├── requirements.txt
└── README.md
```

## Dataset — 9 classi

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

## Come eseguire

### 1. Installazione
```bash
pip install -r requirements.txt
```

I dataset di NLTK (`stopwords`) vengono scaricati automaticamente dai notebook.  
Gli embeddings GloVe (300d, ~1 GB) vengono scaricati da `gensim` alla prima esecuzione.

### 2. Scraping (opzionale — il CSV è già incluso)
Il run completo impiega ~2-3 ore per rispetto del rate limit educato verso BBC.

```bash
jupyter notebook Scraping.ipynb
```

### 3. Classificazione
```bash
jupyter notebook Task2.ipynb
```

## Risultati

### Single split 80/20

| Modello | Accuracy | F1 macro |
|---|---|---|
| Baseline Rule-Based | 0.9241 | 0.9275 |
| Logistic Regression (TF-IDF) | 0.9801 | 0.9784 |
| NN Feedforward — SGD (TF-IDF) | 0.7908 | 0.5729 |
| **NN Feedforward — Adam (TF-IDF)** | **0.9887** | **0.9883** |
| Word2Vec (GloVe 300d, IDF-weighted) + FFN | 0.9745 | 0.9751 |
| Sentence Transformer (chunked) + FFN | 0.9688 | 0.9702 |
| BERT (DistilBERT fine-tuned, 256 tok, 3 ep) | 0.9794 | 0.9797 |

### Cross-Validation 5-fold (Stratified)

| Modello | Accuracy | F1 macro |
|---|---|---|
| Logistic Regression (TF-IDF) | 0.9823 ± 0.0022 | 0.9809 ± 0.0032 |
| **NN Feedforward + Adam (TF-IDF)** | **0.9898 ± 0.0017** | **0.9905 ± 0.0028** |

Lo std basso (<0.005) conferma che il ranking dei modelli è stabile e non artefatto della singola split. BERT, W2V e ST sono esclusi dalla CV per costo computazionale (re-encoding / re-fine-tuning su ogni fold).

## Metodologia

- **Split** 80/20 con `random_state=42` e `stratify=label` (stesso split per tutti i modelli per confronto equo)
- **Tokenizzazione**: `RegexpTokenizer(\w+)` + lowercase + rimozione stopwords NLTK + filtro token non alfabetici / len ≤ 2
- **Feature extraction**: le frequenze per classe e le keyword sono calcolate **solo sul train set** per evitare data leakage
- **Metriche**: `accuracy` e `F1 macro` (non weighted) per penalizzare gli errori sulle classi minori

## Note tecniche

- **GloVe IDF-weighted**: il peso IDF (calcolato sul train) viene usato per fare la media pesata dei vettori GloVe. Le parole rare e discriminanti (es. nomi propri: *Raducanu*, *Benfica*) contano più di quelle generiche.
- **Sentence Transformer chunked**: `all-MiniLM-L6-v2` tronca a 256 token (~200 parole). Per articoli più lunghi viene fatto il chunking in pezzi da 200 parole, encoding di ciascuno, poi media degli embedding.
- **BERT fine-tuned**: `distilbert-base-uncased` (66M parametri) con classification head a 9 classi. Input troncato a 256 token, training 3 epoche, AdamW lr=2e-5, weight_decay=0.01. Su Apple Silicon (MPS) l'addestramento richiede ~5 minuti.
- **Early stopping**: per le NN, training interrotto se la loss non migliora per 7 epoche consecutive (con restore dei best weights).
- **Reproducibility**: `torch.manual_seed(42)` e `random_state=42` su tutti gli split per risultati consistenti tra run.
