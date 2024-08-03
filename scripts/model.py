from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import streamlit as st
import plotly.express as px
from scripts.config import Config

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_deep_learning(X_train, y_train):
    dl_model = Sequential([
        Dense(128, input_dim=len(Config.FEATURES), activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    dl_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stopping])
    return dl_model

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    if model_name == 'Deep Learning':
        y_pred = (y_pred > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    st.write(f"{model_name} Model Performance:")
    st.write(f"Accuracy: {accuracy}")
    st.write("Classification Report:")
    st.text(report)
    
    fig_cm = px.imshow(cm, text_auto=True, title=f'{model_name} Confusion Matrix')
    st.plotly_chart(fig_cm)