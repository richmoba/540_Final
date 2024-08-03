import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
from scripts.config import Config

def load_data():
    if not os.path.exists(Config.FALLS_CSV_PATH) or not os.path.exists(Config.ADLS_CSV_PATH):
        st.error(f"Error: CSV files not found")
        st.stop()
    
    data_falls = pd.read_csv(Config.FALLS_CSV_PATH)
    data_adls = pd.read_csv(Config.ADLS_CSV_PATH)
    return pd.concat([data_falls, data_adls], axis=0)

def preprocess_data(data):
    X = data[Config.FEATURES]
    y = data['label']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)