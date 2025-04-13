import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

# -------------------- Regression Section --------------------
def regression_page():
    st.header("📈 Regression Model")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="regression")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target = st.text_input("Enter target column name", key="reg_target")
        if target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]

            if st.checkbox("Handle missing values", key="reg_missing"):
                X = X.fillna(X.mean())

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
            col2.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")

            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            st.subheader("Custom Prediction")
            inputs = {}
            for col in X.columns:
                inputs[col] = st.number_input(f"{col}", key=f"reg_{col}")
            if st.button("Predict", key="reg_predict"):
                custom_data = pd.DataFrame([inputs])
                prediction = model.predict(custom_data)
                st.success(f"Predicted {target}: {prediction[0]:.2f}")

# -------------------- Clustering Section --------------------
def clustering_page():
    st.header("🧩 Clustering Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="clustering")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        features = st.multiselect("Select features for clustering", df.columns, key="cluster_features")
        if len(features) >= 2:
            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            k = st.slider("Number of clusters", 2, 10, 3, key="n_clusters")
            kmeans = KMeans(n_clusters=k)
            clusters = kmeans.fit_predict(X_scaled)
            df["Cluster"] = clusters

            st.subheader("Cluster Visualization")
            if len(features) == 2:
                fig = px.scatter(df, x=features[0], y=features[1], color="Cluster")
            elif len(features) == 3:
                fig = px.scatter_3d(df, x=features[0], y=features[1], z=features[2], color="Cluster")
            else:
                st.warning("Select 2 or 3 features for visualization")
                return
            st.plotly_chart(fig)

            st.download_button(
                label="Download Clustered Data",
                data=df.to_csv(index=False),
                file_name="clustered_data.csv",
                mime="text/csv"
            )

# -------------------- Neural Network Section --------------------
def nn_page():
    st.header("🧠 Neural Network Classifier")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="nn")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        target = st.text_input("Enter target column name", key="nn_target")
        if target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]

            # Preprocessing
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            le = LabelEncoder()
            y = le.fit_transform(y)

            # Hyperparameters
            st.subheader("Model Configuration")
            col1, col2, col3 = st.columns(3)
            with col1:
                epochs = st.slider("Epochs", 10, 100, 50)
            with col2:
                lr = st.slider("Learning Rate", 0.001, 0.1, 0.01, format="%f")
            with col3:
                batch_size = st.selectbox("Batch Size", [16, 32, 64])

            # Model architecture
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(len(np.unique(y)), activation='softmax')
            ])
            
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            # Training
            if st.button("Train Model", key="nn_train"):
                history = model.fit(X, y, 
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_split=0.2,
                                  verbose=0)
                
                st.subheader("Training Progress")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(history.history['loss'], label='Training Loss')
                ax1.plot(history.history['val_loss'], label='Validation Loss')
                ax1.set_title('Loss Evolution')
                ax1.legend()
                
                ax2.plot(history.history['accuracy'], label='Training Accuracy')
                ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
                ax2.set_title('Accuracy Evolution')
                ax2.legend()
                
                st.pyplot(fig)

# -------------------- LLM Q&A Section --------------------
def llm_page():
    st.header("🤖 LLM Q&A with RAG")
    
    # Document processing
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], key="llm")
    if uploaded_file:
        with st.spinner("Processing document..."):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader("temp.pdf")
            pages = loader.load_and_split()
            
            text_splitter = CharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(pages)
            
            embeddings = HuggingFaceEmbeddings()
            db = FAISS.from_documents(docs, embeddings)
        
        # Q&A interface
        question = st.text_input("Ask about the document:", key="llm_question")
        if question:
            with st.spinner("Searching for answers..."):
                relevant_docs = db.similarity_search(question, k=3)
                context = "\n\n".join([doc.page_content for doc in relevant_docs])
                
                llm = pipeline(
                    "text-generation",
                    model="mistralai/Mistral-7B-Instruct-v0.1",
                    device_map="auto"
                )
                
                prompt = f"""<s>[INST] Answer the question based on the context below:
                Context: {context}
                Question: {question} [/INST]"""
                
                response = llm(
                    prompt,
                    max_new_tokens=500,
                    do_sample=True,
                    temperature=0.7
                )[0]['generated_text']
                
            st.subheader("Answer")
            st.write(response.split("[/INST]")[-1].strip())

# -------------------- Main App --------------------
def main():
    st.set_page_config(page_title="ML Suite", layout="wide")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a module:", [
        "Regression",
        "Clustering",
        "Neural Network",
        "LLM Q&A"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **ML Suite** provides multiple machine learning capabilities:
    - 📈 Regression modeling
    - 🧩 K-Means clustering
    - 🧠 Neural network training
    - 🤖 LLM-based Q&A with RAG
    """)
    
    if page == "Regression":
        regression_page()
    elif page == "Clustering":
        clustering_page()
    elif page == "Neural Network":
        nn_page()
    elif page == "LLM Q&A":
        llm_page()

if __name__ == "__main__":
    main()