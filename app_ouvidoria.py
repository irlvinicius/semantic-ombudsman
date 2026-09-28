import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Ouvidoria Inteligente", page_icon="🏛️", layout="wide")

DATA_PATH = Path(__file__).resolve().parent / "data" / "manifestacoes.json"
MODELOS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/distiluse-base-multilingual-cased-v2",
]


# ---------- Funções auxiliares (cache conforme dica do enunciado) ----------
@st.cache_resource
def carregar_modelo(nome):
    return SentenceTransformer(nome)


@st.cache_data
def carregar_dados():
    with open(DATA_PATH, encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


@st.cache_data
def gerar_embeddings(nome_modelo, textos):
    modelo = carregar_modelo(nome_modelo)
    return modelo.encode(list(textos), normalize_embeddings=True)


def reduzir_2d(embeddings, metodo):
    if metodo == "PCA":
        return PCA(n_components=2).fit_transform(embeddings)
    perplexity = max(2, min(10, len(embeddings) - 1))
    return TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)


def mostrar_resultado(score, texto):
    """Destaque por cor: verde > 0.7, amarelo > 0.5, vermelho nos demais."""
    if score > 0.7:
        st.success(f"🟢 **{score:.3f}** — {texto}")
    elif score > 0.5:
        st.warning(f"🟡 **{score:.3f}** — {texto}")
    else:
        st.error(f"🔴 **{score:.3f}** — {texto}")


# ---------- Sidebar ----------
st.sidebar.title("⚙️ Configurações")
nome_modelo = st.sidebar.selectbox("Modelo de embedding", MODELOS)
top_k = st.sidebar.slider("Top-k resultados", min_value=1, max_value=10, value=5)

# ---------- Dados e embeddings ----------
df = carregar_dados()
textos = df["texto"].tolist()
ids = df["id"].tolist()

with st.spinner("Carregando modelo e gerando embeddings..."):
    modelo = carregar_modelo(nome_modelo)
    embeddings = gerar_embeddings(nome_modelo, tuple(textos))

st.title("🏛️ Ouvidoria Inteligente")
aba_busca, aba_base, aba_espaco, aba_chunking = st.tabs(
    ["🔍 Busca Semântica", "📋 Base Completa", "🔴 Espaço Vetorial", "🔴 Chunking"]
)

# ---------- Aba 1: Busca semântica ----------
with aba_busca:
    consulta = st.text_area("Descreva o problema:", placeholder="Ex.: a rua está cheia de buracos")
    if st.button("Buscar", key="buscar") and consulta.strip():
        vetor = modelo.encode([consulta], normalize_embeddings=True)
        similaridades = cosine_similarity(vetor, embeddings)[0]
        for i in np.argsort(similaridades)[::-1][:top_k]:
            st.caption(f"{ids[i]} · {df.loc[i, 'categoria_oficial']}")
            mostrar_resultado(similaridades[i], textos[i])

# ---------- Aba 2: Base completa ----------
with aba_base:
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("Gerar matriz de similaridade", key="matriz"):
        matriz = cosine_similarity(embeddings)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(matriz, xticklabels=ids, yticklabels=ids, cmap="viridis", vmin=0, vmax=1, square=True, ax=ax)
        ax.set_title("Matriz de similaridade de cosseno")
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(fontsize=7)
        st.pyplot(fig)

# ---------- Aba 3: Espaço vetorial ----------
with aba_espaco:
    metodo = st.radio("Método de redução", ["PCA", "t-SNE"], horizontal=True)
    pontos = reduzir_2d(embeddings, metodo)

    plot_df = df.assign(x=pontos[:, 0], y=pontos[:, 1])
    fig = px.scatter(
        plot_df, x="x", y="y", color="categoria_oficial",
        hover_data=["id", "texto"], title=f"Manifestações no espaço vetorial ({metodo})",
    )
    fig.update_traces(marker=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)

    st.info("💬 Comentário: *(escreva aqui, depois de ver o gráfico, se os clusters semânticos "
            "coincidem com as categorias oficiais e quais categorias se misturam)*")

# ---------- Aba 4: Chunking ----------
with aba_chunking:
    texto_longo = st.text_area("Cole uma manifestação longa:", height=200, key="texto_longo")

    col1, col2, col3 = st.columns(3)
    estrategia = col1.selectbox("Estratégia", ["RecursiveCharacterTextSplitter", "CharacterTextSplitter"])
    chunk_size = col2.number_input("chunk_size", min_value=50, max_value=1000, value=300, step=10)
    chunk_overlap = col3.number_input("chunk_overlap", min_value=0, max_value=500, value=50, step=10)

    if st.button("Gerar chunks", key="chunks") and texto_longo.strip():
        if chunk_overlap >= chunk_size:
            st.error("O overlap precisa ser menor que o chunk_size.")
        else:
            if estrategia == "RecursiveCharacterTextSplitter":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""], keep_separator="end",
                )
            else:
                splitter = CharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=". ", keep_separator="end",
                )

            chunks = splitter.split_text(texto_longo)
            emb_chunks = modelo.encode(chunks, normalize_embeddings=True)

            st.subheader(f"{len(chunks)} chunks gerados")
            tabela = pd.DataFrame({
                "chunk": range(len(chunks)),
                "caracteres": [len(c) for c in chunks],
                "texto": chunks,
                "embedding (5 primeiras dims)": [np.round(v[:5], 3).tolist() for v in emb_chunks],
            })
            st.dataframe(tabela, use_container_width=True, hide_index=True)

            with st.expander("Ver embeddings completos"):
                st.dataframe(pd.DataFrame(emb_chunks), use_container_width=True)

            if len(chunks) >= 3:
                pts = PCA(n_components=2).fit_transform(emb_chunks)
                fig = px.scatter(
                    x=pts[:, 0], y=pts[:, 1], text=[f"chunk {i}" for i in range(len(chunks))],
                    title="Chunks no espaço vetorial (PCA)", labels={"x": "PC1", "y": "PC2"},
                )
                fig.update_traces(textposition="top center", marker=dict(size=11))
                st.plotly_chart(fig, use_container_width=True)