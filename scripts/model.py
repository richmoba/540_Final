import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import plotly.express as px
from scripts.config import Config

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def build_improved_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_deep_learning(X_train, y_train):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy',  # Use binary crossentropy for binary classification
                  metrics=['accuracy'])
    
    # Ensure y_train is the correct shape
    y_train = y_train.astype(int).reshape(-1, 1)
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    if model_name == "Deep Learning":
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    else:
        y_pred = model.predict(X_test)
    
    # Ensure y_test is also flattened and converted to int
    y_test = y_test.astype(int).flatten()
    
    # Remove any NaN or infinite values
    mask = np.isfinite(y_pred) & np.isfinite(y_test)
    y_pred = y_pred[mask]
    y_test = y_test[mask]
    
    if len(y_pred) == 0 or len(y_test) == 0:
        st.error("No valid predictions or test labels after removing NaN/inf values.")
        return
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    st.write(f"{model_name} Model Performance:")
    st.write(f"Accuracy: {accuracy}")
    st.write("Classification Report:")
    st.text(report)
    
    fig_cm = px.imshow(cm, text_auto=True, title=f'{model_name} Confusion Matrix')
    st.plotly_chart(fig_cm)

    # Additional information for debugging
    st.write(f"y_test shape: {y_test.shape}, unique values: {np.unique(y_test)}")
    st.write(f"y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")