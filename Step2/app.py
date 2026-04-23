import os
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import streamlit as st

import nltk
nltk.download("stopwords", quiet=True)
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


MODELS_DIR = "models"
DEVICE = torch.device("cpu")


#  Preprocessing
STOP_WORDS = set(stopwords.words("english"))
_tokenizer = RegexpTokenizer(r"\w+")


def tokenizza_e_pulisci(testo: str) -> list[str]:
    tokens = _tokenizer.tokenize(testo.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 2]


CLASSI = [
    "non_sport", "football", "tennis", "rugby", "cricket",
    "other_sport", "formula1", "american_football", "golf",
]


#  Architetture NN
class FeedforwardNN(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, n_classi: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, n_classi)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.output(self.act(self.hidden(x)))


class FeedforwardNNDrop(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, n_classi: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classi),
        )

    def forward(self, x):
        return self.net(x)


#  TF-IDF manuale
def tfidf_transform(text: str, vocab: dict, idf: np.ndarray) -> np.ndarray:
    tokens = tokenizza_e_pulisci(text)
    counts = Counter(tokens)
    vec = np.zeros(len(idf), dtype=np.float64)
    for tok, cnt in counts.items():
        idx = vocab.get(tok)
        if idx is not None:
            vec[idx] = (np.log(cnt) + 1.0) * idf[idx]  
    norm = np.sqrt((vec ** 2).sum())
    if norm > 0:
        vec /= norm
    return vec.astype(np.float32)


#  Streamlit config + sanity check modelli presenti
st.set_page_config(
    page_title="BBC News Classifier",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

required = [
    "label_encoder.pkl", "tfidf_nn.npz", "rule_based_keywords.pkl", "lr.npz",
    "w2v_idf.pkl", "nn_sgd/state_dict.pt", "nn_adam/state_dict.pt",
    "w2v/state_dict.pt", "st/state_dict.pt", "bert/config.json",
]
missing = [p for p in required if not os.path.exists(os.path.join(MODELS_DIR, p))]
if missing:
    st.error(
        "**Modelli non trovati.** Esegui Task2.ipynb fino in fondo "
        "(incluse le celle finali «Export modelli per Streamlit») per generare "
        "la cartella `./models/`.\n\n"
        f"File mancanti: `{', '.join(missing[:3])}`{' ...' if len(missing) > 3 else ''}"
    )
    st.stop()


#  Loader cachati (ogni modello caricato una sola volta)
@st.cache_resource
def load_label_encoder():
    return joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))


@st.cache_resource
def load_tfidf_nn():
    d = np.load(os.path.join(MODELS_DIR, "tfidf_nn.npz"), allow_pickle=True)
    vocab = dict(zip(d["vocab_keys"].tolist(), d["vocab_values"].astype(int).tolist()))
    return vocab, d["idf"].astype(np.float64)


@st.cache_resource
def load_rule_based():
    return joblib.load(os.path.join(MODELS_DIR, "rule_based_keywords.pkl"))


@st.cache_resource
def load_lr():
    d = np.load(os.path.join(MODELS_DIR, "lr.npz"), allow_pickle=True)
    vocab = dict(zip(d["vocab_keys"].tolist(), d["vocab_values"].astype(int).tolist()))
    return (
        vocab, d["idf"].astype(np.float64),
        d["coef"].astype(np.float64), d["intercept"].astype(np.float64),
        d["classes"],
    )


def _load_nn(subdir: str, Cls):
    with open(os.path.join(MODELS_DIR, subdir, "config.json")) as f:
        cfg = json.load(f)
    kwargs = dict(input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], n_classi=cfg["n_classi"])
    if Cls is FeedforwardNNDrop:
        kwargs["dropout"] = cfg.get("dropout", 0.3)
    model = Cls(**kwargs)
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, subdir, "state_dict.pt"), map_location=DEVICE
    ))
    model.eval()
    return model, cfg


@st.cache_resource
def load_nn_sgd():  return _load_nn("nn_sgd",  FeedforwardNN)

@st.cache_resource
def load_nn_adam(): return _load_nn("nn_adam", FeedforwardNN)

@st.cache_resource
def load_w2v_ffn(): return _load_nn("w2v",     FeedforwardNNDrop)

@st.cache_resource
def load_st_ffn():  return _load_nn("st",      FeedforwardNNDrop)


@st.cache_resource
def load_glove():
    import gensim.downloader as api
    return api.load("glove-wiki-gigaword-300")


@st.cache_resource
def load_w2v_idf():
    d = joblib.load(os.path.join(MODELS_DIR, "w2v_idf.pkl"))
    return d["idf_map"], d["idf_default"]


@st.cache_resource
def load_st_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_bert():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = os.path.join(MODELS_DIR, "bert")
    tok = AutoTokenizer.from_pretrained(path)
    mdl = AutoModelForSequenceClassification.from_pretrained(path).to(DEVICE)
    mdl.eval()
    with open(os.path.join(path, "inference_config.json")) as f:
        cfg = json.load(f)
    return tok, mdl, cfg


#  Funzioni di inferenza (una per modello)
def predict_rule_based(text, keyword_per_classe):
    tokens = tokenizza_e_pulisci(text)
    punteggi = {cls: sum(1 for t in tokens if t in keyword_per_classe[cls]) for cls in CLASSI}
    totale = sum(punteggi.values())
    if totale == 0:
        pred = "non_sport"
        probs = {cls: (1.0 if cls == "non_sport" else 0.0) for cls in CLASSI}
    else:
        pred = max(punteggi, key=punteggi.get)
        probs = {cls: punteggi[cls] / totale for cls in CLASSI}
    return pred, probs


def predict_lr(text, vocab, idf, coef, intercept, classes):
    vec = tfidf_transform(text, vocab, idf)
    logits = vec.astype(np.float64) @ coef.T + intercept
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    pred = classes[probs.argmax()]
    return pred, dict(zip(classes, probs))


def predict_nn_tfidf(text, model, vocab, idf, le):
    vec = tfidf_transform(text, vocab, idf)
    with torch.no_grad():
        logits = model(torch.tensor(vec).unsqueeze(0))
        probs = F.softmax(logits, dim=-1).numpy()[0]
    pred = le.classes_[probs.argmax()]
    return pred, dict(zip(le.classes_, probs))


def _text_to_w2v_embedding(text, glove, idf_map, idf_default, dim=300):
    tokens = tokenizza_e_pulisci(text)
    vettori, pesi = [], []
    for t in tokens:
        if t in glove:
            vettori.append(glove[t])
            pesi.append(idf_map.get(t, idf_default))
    if not vettori:
        return np.zeros(dim, dtype=np.float32)
    return np.average(np.asarray(vettori), axis=0, weights=np.asarray(pesi)).astype(np.float32)


def predict_w2v(text, model, glove, idf_map, idf_default, le):
    emb = _text_to_w2v_embedding(text, glove, idf_map, idf_default)
    with torch.no_grad():
        logits = model(torch.tensor(emb).unsqueeze(0))
        probs = F.softmax(logits, dim=-1).numpy()[0]
    pred = le.classes_[probs.argmax()]
    return pred, dict(zip(le.classes_, probs))


def _text_to_st_embedding(text, st_enc, chunk_words=200):
    parole = text.split()
    chunks = [""] if not parole else [
        " ".join(parole[i:i + chunk_words])
        for i in range(0, len(parole), chunk_words)
    ]
    embs = st_enc.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return embs.mean(axis=0).astype(np.float32)


def predict_st(text, model, st_enc, le, chunk_words=200):
    emb = _text_to_st_embedding(text, st_enc, chunk_words)
    with torch.no_grad():
        logits = model(torch.tensor(emb).unsqueeze(0))
        probs = F.softmax(logits, dim=-1).numpy()[0]
    pred = le.classes_[probs.argmax()]
    return pred, dict(zip(le.classes_, probs))


def predict_bert(text, bert_tok, bert_model, le, max_len=256):
    enc = bert_tok(
        text,
        truncation=True, padding="max_length", max_length=max_len,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        logits = bert_model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    pred = le.classes_[probs.argmax()]
    return pred, dict(zip(le.classes_, probs))

#  UI
EXAMPLES = {
    "Football": (
        "Manchester United beat Arsenal 2-1 at Old Trafford last night, with "
        "Marcus Rashford scoring the winning goal in the 89th minute after a "
        "dramatic match that saw both teams reduced to 10 men."
    ),
    "Formula 1": (
        "Max Verstappen dominated the Monaco Grand Prix, leading every lap "
        "from pole position to claim his fourth victory of the season. "
        "Ferrari's Charles Leclerc finished second ahead of Lando Norris."
    ),
    "Cricket": (
        "England won the third Test against India by 7 wickets at Lord's, "
        "with Joe Root scoring a magnificent century in the second innings "
        "to set up a series-clinching victory for the hosts."
    ),
    "Non-sport": (
        "The government announced new measures to combat inflation, including "
        "tax cuts for small businesses and increased funding for public transport. "
        "The Chancellor presented the plan to parliament this morning."
    ),
}

DEFAULT_TEXT = (
    "Rafael Nadal secured his 14th French Open title at Roland Garros, defeating "
    "Casper Ruud in straight sets on a warm Sunday afternoon in Paris."
)


# Inizializzazione session state
if "input_text" not in st.session_state:
    st.session_state["input_text"] = DEFAULT_TEXT


def _clear_text():
    st.session_state["input_text"] = ""


def _set_example(label: str):
    st.session_state["input_text"] = EXAMPLES[label]


st.title("BBC News Classifier")
st.markdown(
    "Classifica un testo in una delle **9 classi** "
    "(`football`, `cricket`, `rugby`, `tennis`, `other_sport`, `non_sport`, "
    "`formula1`, `american_football`, `golf`) usando **7 modelli NLP** addestrati "
    "sul dataset BBC (7 049 articoli). Ogni modello mostra la classe predetta "
    "e la distribuzione di probabilità sulle 9 classi."
)

# Bottoni-esempio compatti sopra la text area
ex_cols = st.columns(len(EXAMPLES))
for col, label in zip(ex_cols, EXAMPLES):
    col.button(label, use_container_width=True, on_click=_set_example, args=(label,))

st.text_area(
    "Inserisci un testo da classificare:",
    height=180,
    key="input_text",
    help="Meglio se in inglese (i modelli sono addestrati sul corpus BBC inglese).",
)

col1, col2, _ = st.columns([1, 1, 4])
classifica = col1.button("Classifica", type="primary", use_container_width=True)
col2.button("Pulisci", use_container_width=True, on_click=_clear_text)

user_text = st.session_state["input_text"]

if classifica and user_text.strip():
    le = load_label_encoder()
    vocab_nn, idf_nn = load_tfidf_nn()

    results = {}
    progress = st.progress(0.0, text="Caricamento modelli...")

    progress.progress(0.10, text="Rule-based...")
    kw = load_rule_based()
    results["Rule-based"] = predict_rule_based(user_text, kw)

    progress.progress(0.20, text="Logistic Regression...")
    vocab_lr, idf_lr, coef, intercept, classes = load_lr()
    results["Logistic Regression"] = predict_lr(user_text, vocab_lr, idf_lr, coef, intercept, classes)

    progress.progress(0.30, text="NN Feedforward (SGD)...")
    m_sgd, _ = load_nn_sgd()
    results["NN FF (SGD)"] = predict_nn_tfidf(user_text, m_sgd, vocab_nn, idf_nn, le)

    progress.progress(0.45, text="NN Feedforward (Adam)...")
    m_adam, _ = load_nn_adam()
    results["NN FF (Adam)"] = predict_nn_tfidf(user_text, m_adam, vocab_nn, idf_nn, le)

    progress.progress(0.60, text="Word2Vec (GloVe) + FFN...")
    m_w2v, _ = load_w2v_ffn()
    glove = load_glove()
    idf_map, idf_default = load_w2v_idf()
    results["Word2Vec + FFN"] = predict_w2v(user_text, m_w2v, glove, idf_map, idf_default, le)

    progress.progress(0.75, text="Sentence Transformer...")
    m_st, st_cfg = load_st_ffn()
    st_enc = load_st_encoder()
    results["Sentence Transformer"] = predict_st(user_text, m_st, st_enc, le, st_cfg.get("chunk_words", 200))

    progress.progress(0.90, text="BERT (DistilBERT)...")
    bert_tok, bert_model, bert_cfg = load_bert()
    results["BERT (DistilBERT)"] = predict_bert(user_text, bert_tok, bert_model, le, bert_cfg["max_len"])

    progress.progress(1.0, text="Completato")
    progress.empty()

    preds = [r[0] for r in results.values()]
    winner, count = Counter(preds).most_common(1)[0]
    if count == len(preds):
        st.success(f"**Consenso unanime** — tutti i 7 modelli predicono: **`{winner}`**")
    elif count >= 5:
        st.info(f"**Maggioranza ({count}/7)** → **`{winner}`**")
    else:
        st.warning(f"**Modelli in disaccordo** — {count}/7 per **`{winner}`** (guarda i dettagli sotto)")

    st.subheader("Riepilogo predizioni")
    summary = pd.DataFrame([
        {
            "Modello": name,
            "Predizione": pred,
            "Confidenza": f"{probs[pred]:.1%}",
            "Top-2": sorted(probs.items(), key=lambda x: -x[1])[1][0],
        }
        for name, (pred, probs) in results.items()
    ])
    st.dataframe(summary, hide_index=True, use_container_width=True)

    st.subheader("Distribuzione di probabilità per modello")

    items = list(results.items())
    for i in range(0, len(items), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(items):
                break
            name, (pred, probs) = items[i + j]
            with col:
                st.markdown(f"**{name}** → `{pred}`  ·  {probs[pred]:.1%}")
                df_prob = (
                    pd.DataFrame(probs.items(), columns=["classe", "probabilità"])
                    .sort_values("probabilità", ascending=False)
                    .set_index("classe")
                )
                st.bar_chart(df_prob, height=280)

    with st.expander("Performance dei modelli sul test set (80/20)"):
        st.dataframe(
            pd.DataFrame([
                ("Rule-based",           "0.9241", "0.9275"),
                ("Logistic Regression",  "0.9801", "0.9784"),
                ("NN FF (SGD)",          "0.7908", "0.5729"),
                ("NN FF (Adam)",         "0.9887", "0.9883"),
                ("Word2Vec + FFN",       "0.9745", "0.9751"),
                ("Sentence Transformer", "0.9688", "0.9702"),
                ("BERT (DistilBERT)",    "0.9794", "0.9797"),
            ], columns=["Modello", "Accuracy", "F1 macro"]),
            hide_index=True, use_container_width=True,
        )

elif classifica and not user_text.strip():
    st.warning("Inserisci prima un testo da classificare.")
