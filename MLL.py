import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from transformers import pipeline

# Streamlit App
st.title("Integrated Machine Learning and NLP App")

# Sidebar Navigation
app_mode = st.sidebar.selectbox("Choose the functionality", ["Regression", "Clustering", "Neural Network", "LLM"])

# Regression Functionality
if app_mode == "Regression":
    st.header("Regression Model")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
        target_column = st.text_input("Specify target column name")
        if target_column:
            try:
                X = data.drop(columns=[target_column])
                y = data[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"Mean Absolute Error: {mae}")
                st.write(f"R2 Score: {r2}")
                plt.scatter(y_test, y_pred)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Actual vs Predicted")
                st.pyplot(plt)
                custom_input = st.text_input("Enter custom data (comma-separated values)")
                if custom_input:
                    custom_data = np.array([float(i) for i in custom_input.split(",")]).reshape(1, -1)
                    prediction = model.predict(custom_data)
                    st.write(f"Prediction: {prediction[0]}")
            except Exception as e:
                st.error(f"Error: {e}")

# Clustering Functionality
elif app_mode == "Clustering":
    st.header("Clustering with K-Means")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
        features = st.multiselect("Select features for clustering", data.columns)
        if features:
            X = data[features]
            n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                data['Cluster'] = kmeans.fit_predict(X)
                st.write("Clustered Dataset:")
                st.write(data)
                csv = data.to_csv(index=False)
                st.download_button("Download Clustered Dataset", data=csv, mime="text/csv")
                if len(features) == 2:
                    plt.scatter(data[features[0]], data[features[1]], c=data['Cluster'], cmap='viridis')
                    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red')
                    plt.xlabel(features[0])
                    plt.ylabel(features[1])
                    plt.title("Clusters")
                    st.pyplot(plt)
                elif len(features) == 3:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(data[features[0]], data[features[1]], data[features[2]], c=data['Cluster'], cmap='viridis')
                    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=200, c='red')
                    ax.set_xlabel(features[0])
                    ax.set_ylabel(features[1])
                    ax.set_zlabel(features[2])
                    st.pyplot(fig)
            except Exception as e:
                st.error(f"Error: {e}")

# Neural Network Functionality
elif app_mode == "Neural Network":
    st.header("Neural Network for Classification")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
        target_column = st.text_input("Specify target column name")
        if target_column:
            try:
                X = data.drop(columns=[target_column])
                y = pd.get_dummies(data[target_column]).values
                model = Sequential([
                    Dense(128, activation='relu', input_shape=(X.shape[1],)),
                    Dense(64, activation='relu'),
                    Dense(y.shape[1], activation='softmax')
                ])
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                history = model.fit(X, y, validation_split=0.2, epochs=10, batch_size=32)
                plt.plot(history.history['loss'], label='Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.legend()
                st.pyplot(plt)
                custom_input = st.text_input("Enter custom data (comma-separated values)")
                if custom_input:
                    custom_data = np.array([float(i) for i in custom_input.split(",")]).reshape(1, -1)
                    prediction = model.predict(custom_data)
                    st.write(f"Prediction: {np.argmax(prediction)}")
            except Exception as e:
                st.error(f"Error: {e}")

# LLM Functionality
elif app_mode == "LLM":
    st.header("Large Language Model (LLM) - Q&A")
    model_name = st.selectbox("Select LLM Model", ["mistralai/Mistral-7B-Instruct-v0.1"])
    qa_pipeline = pipeline("question-answering", model=model_name)
    uploaded_file = st.file_uploader("Upload your dataset (PDF or CSV)", type=["csv", "pdf"])
    if uploaded_file:
        st.write("Dataset Uploaded Successfully")
        question = st.text_input("Ask a Question")
        if question:
            try:
                context = "Provide context from the uploaded dataset here."
                response = qa_pipeline(question=question, context=context)
                st.write(f"Answer: {response['answer']}")
            except Exception as e:
                st.error(f"Error: {e}")
