import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"




# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import asyncio
import nest_asyncio
import tempfile
import os

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# TensorFlow + Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# LangChain + FAISS + HuggingFace
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# HuggingFace transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# PDF utilities
import pypdf
import PyPDF2

# Fix asyncio loop issues 
nest_asyncio.apply()

# Streamlit page config
st.set_page_config(page_title="ML & AI Explorer", layout="wide")

# Sidebar navigation
st.sidebar.title("🔍 ML & AI Explorer")
section = st.sidebar.selectbox("Choose a Task", ["Regression", "Neural Network", "Clustering", "LLM Q&A"])


# ---------------------- REGRESSION MODULE ---------------------- #
if section == "Regression":
    st.title("📈 Regression Analysis")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("🔍 Dataset Preview")
        st.dataframe(df.head())

        target = st.selectbox("Select target column", df.columns)

        if target:
            X = df.drop(columns=[target])
            y = df[target]

            if st.checkbox("Scale features"):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X.values

            model = LinearRegression()
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)

            st.subheader("📊 Model Performance")
            st.write(f"**MAE:** {mean_absolute_error(y, y_pred):.4f}")
            st.write(f"**R² Score:** {r2_score(y, y_pred):.4f}")

            st.subheader("🔁 Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y, y_pred)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            st.subheader("🔮 Custom Prediction")
            input_data = {}
            for col in X.columns:
                val = st.number_input(f"Enter value for {col}", value=float(df[col].mean()))
                input_data[col] = val

            input_df = pd.DataFrame([input_data])
            if st.checkbox("Use scaled input"):
                input_df = scaler.transform(input_df)
            pred = model.predict(input_df)
            st.success(f"Predicted {target}: {pred[0]:.2f}")

# ---------------------- NEURAL NETWORK MODULE ---------------------- #
elif section == "Neural Network":
    st.title("🧠 Neural Network Classifier")
    nn_file = st.file_uploader("Upload classification CSV file", type=["csv"], key="nn_upload")

    if nn_file:
        df = pd.read_csv(nn_file)
        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head())

        target_col = st.selectbox("Select target column", df.columns)

        if target_col:
            X = df.drop(columns=[target_col])
            y = pd.get_dummies(df[target_col])

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            st.subheader("⚙️ Set Hyperparameters")
            epochs = st.slider("Epochs", 1, 100, 10)
            lr = st.number_input("Learning rate", value=0.001, format="%f")

            model = Sequential([
                Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
                Dense(32, activation='relu'),
                Dense(y.shape[1], activation='softmax')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            history = model.fit(X_scaled, y, epochs=epochs, verbose=0, validation_split=0.2)

            st.subheader("📉 Training Progress")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            ax[0].plot(history.history['loss'], label='Loss')
            ax[0].plot(history.history['val_loss'], label='Val Loss')
            ax[0].legend()
            ax[0].set_title("Loss")
            ax[1].plot(history.history['accuracy'], label='Accuracy')
            ax[1].plot(history.history['val_accuracy'], label='Val Accuracy')
            ax[1].legend()
            ax[1].set_title("Accuracy")
            st.pyplot(fig)

            st.subheader("🔮 Upload Test Sample for Prediction")
            test_file = st.file_uploader("Upload test sample CSV (same features as training)", key="nn_test")

            if test_file:
                test_df = pd.read_csv(test_file)
                test_scaled = scaler.transform(test_df)
                preds = model.predict(test_scaled)
                predicted_labels = y.columns[np.argmax(preds, axis=1)]
                test_df['Prediction'] = predicted_labels
                st.write(test_df)
                st.download_button("📥 Download Predictions", data=test_df.to_csv(index=False), file_name="predictions.csv")

# ---------------------- CLUSTERING MODULE ---------------------- #
elif section == "Clustering":
    st.title("🔘 K-Means Clustering")
    clus_file = st.file_uploader("Upload CSV for Clustering", type=["csv"])

    if clus_file:
        df = pd.read_csv(clus_file)
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("⚙️ Preprocessing")
        features = st.multiselect("Select features for clustering", df.columns.tolist(), default=df.columns.tolist())
        X = df[features]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = st.slider("Select number of clusters (k)", 2, 10, 3)
        kmeans = KMeans(n_clusters=k, random_state=0)
        clusters = kmeans.fit_predict(X_scaled)

        df['Cluster'] = clusters
        st.subheader("📌 Clustered Data")
        st.dataframe(df)

        if X.shape[1] == 2:
            st.subheader("📈 2D Cluster Visualization")
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
            centers = kmeans.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            st.pyplot(fig)

        st.download_button("📥 Download Clustered Data", data=df.to_csv(index=False), file_name="clustered_data.csv")

# ---------------------- LLM Q&A MODULE ---------------------- #
elif section == "LLM Q&A":
    st.title("📖 2025 Ghana Budget Q&A")

    st.markdown("""
    **Architecture (RAG):**
    - Load and split PDF
    - Embed and store in FAISS
    - Retrieve relevant chunks
    - Ask Falcon-RW-1B LLM to generate answers via Hugging Face
    """)

    pdf_url = "https://mofep.gov.gh/sites/default/files/budget-statements/2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    st.markdown(f"[📥 Download the 2025 Budget PDF]({pdf_url})")

    pdf_file = st.file_uploader("📄 Upload the 2025 Budget PDF", type="pdf")

    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        st.success("✅ PDF uploaded successfully!")

        # Load and split PDF
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)

        st.write(f"✅ Document split into {len(chunks)} chunks")

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectordb = FAISS.from_documents(chunks, embeddings)
        retriever = vectordb.as_retriever()

        # Load Falcon-RW-1B via Hugging Face
        tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")
        model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=1024)
        llm = HuggingFacePipeline(pipeline=pipe)

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # User query input
        query = st.text_input("Ask a question about the 2025 Ghana Budget")

        if query:
            with st.spinner("⏳ Generating answer..."):
                try:
                    response = qa_chain.run(query)
                    st.success("✅ Answer:")
                    st.write(response)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
