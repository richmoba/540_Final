import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scripts.config import Config
import streamlit as st
from sklearn.preprocessing import LabelEncoder

def load_data():
    try:
        falls_data = pd.read_csv(Config.FALLS_CSV_PATH)
        adls_data = pd.read_csv(Config.ADLS_CSV_PATH)
        
        data = pd.concat([falls_data, adls_data], axis=0, ignore_index=True)
        
        st.write(f"Data loaded successfully. Shape: {data.shape}")
        st.write(f"Columns: {data.columns.tolist()}")
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def preprocess_data(data):
    if data is None or data.empty:
        st.error("No data to preprocess.")
        return None, None, None, None

    try:
        st.write(f"Preprocessing data. Original shape: {data.shape}")
        
        X = data[Config.FEATURES]
        y = data['label']
        
        # Convert labels to binary (assuming 'fall' is the positive class)
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        st.write(f"Features shape: {X.shape}")
        st.write(f"Labels shape: {y.shape}")
        st.write(f"Unique labels: {np.unique(y)}")
        
        # Remove rows with NaN or infinite values
        mask = np.isfinite(X).all(axis=1)
        X = X[mask]
        y = y[mask]
        
        st.write(f"Shape after removing NaN/inf values: {X.shape}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        
        st.write(f"Train set shape: {X_train.shape}")
        st.write(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error preprocessing data: {str(e)}")
        return None, None, None, None