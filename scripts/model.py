import numpy as np  # Importing the necessary libraries
from sklearn.ensemble import RandomForestClassifier # Importing the necessary libraries
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # Importing the necessary libraries
from tensorflow.keras.models import Sequential  # Importing the necessary libraries
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # Importing the necessary libraries
from tensorflow.keras.optimizers import Adam    # Importing the necessary libraries
from tensorflow.keras.callbacks import EarlyStopping    # Importing the necessary libraries
import streamlit as st  # Importing the necessary libraries
import plotly.express as px # Importing the necessary libraries
from scripts.config import Config   # Importing the necessary libraries

def train_random_forest(X_train, y_train):  # Function to train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)    # Create a Random Forest model
    rf_model.fit(X_train, y_train)  # Fit the model on the training data
    return rf_model   # Return the trained model

def build_improved_model(input_shape):  # Function to build an improved Deep Learning model
    model = Sequential([    # Create a Sequential model
        Dense(128, activation='relu', input_shape=input_shape), # Add a Dense layer with ReLU activation
        BatchNormalization(),   # Add a BatchNormalization layer
        Dropout(0.3),   # Add a Dropout layer
        Dense(256, activation='relu'),  # Add another Dense layer with ReLU activation
        BatchNormalization(),   # Add another BatchNormalization layer
        Dropout(0.3),   # Add another Dropout layer
        Dense(128, activation='relu'),  # Add another Dense layer with ReLU activation
        BatchNormalization(),   # Add another BatchNormalization layer
        Dropout(0.3),   # Add another Dropout layer
        Dense(64, activation='relu'),   # Add another Dense layer with ReLU activation
        BatchNormalization(),   # Add another BatchNormalization layer
        Dropout(0.3),   # Add another Dropout layer
        Dense(1, activation='sigmoid')  # Add a Dense layer with Sigmoid activation
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),  # Compile the model with Adam optimizer
                  loss='binary_crossentropy',   # Use binary crossentropy loss
                  metrics=['accuracy'])   # Use accuracy as the metric
    
    return model    # Return the model

def train_deep_learning(X_train, y_train):  # Function to train the Deep Learning model
    model = Sequential([    # Create a Sequential model
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Add a Dense layer with ReLU activation
        Dropout(0.2),       # Add a Dropout layer
        Dense(32, activation='relu'),   # Add another Dense layer with ReLU activation
        Dropout(0.2),    # Add another Dropout layer
        Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),  # Compile the model with Adam optimizer
                  loss='binary_crossentropy',  # Use binary crossentropy for binary classification
                  metrics=['accuracy']) # Use accuracy as the metric
    
    # Ensure y_train is the correct shape
    y_train = y_train.astype(int).reshape(-1, 1)    # Reshape y_train to (-1, 1)
    
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)  # Fit the model on the training data
    return model    # Return the trained model

def evaluate_model(model, X_test, y_test, model_name):  # Function to evaluate the model
    if model_name == "Deep Learning":   # If the model is Deep Learning
        y_pred_proba = model.predict(X_test)    # Predict probabilities
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()   # Convert probabilities to binary predictions
    else:   # If the model is not Deep Learning
        y_pred = model.predict(X_test)  # Predict using the model
    
    # Ensure y_test is also flattened and converted to int
    y_test = y_test.astype(int).flatten()   # Flatten and convert y_test to int

    # Remove any NaN or infinite values
    mask = np.isfinite(y_pred) & np.isfinite(y_test)    # Create a mask to remove NaN or infinite values
    y_pred = y_pred[mask]           # Apply the mask to y_pred
    y_test = y_test[mask]        # Apply the mask to y_test
    
    if len(y_pred) == 0 or len(y_test) == 0:    # If there are no valid predictions or test labels
        st.error("No valid predictions or test labels after removing NaN/inf values.")  # Display an error message
        return  # Return
    
    accuracy = accuracy_score(y_test, y_pred)   # Calculate accuracy
    report = classification_report(y_test, y_pred)  # Generate classification report
    cm = confusion_matrix(y_test, y_pred)   # Generate confusion matrix
    
    st.write(f"{model_name} Model Performance:")    # Display the model performance
    st.write(f"Accuracy: {accuracy}")   # Display the accuracy
    st.write("Classification Report:")  # Display the classification report
    st.text(report) # Display the classification report as text
    
    fig_cm = px.imshow(cm, text_auto=True, title=f'{model_name} Confusion Matrix')  # Create a confusion matrix plot
    st.plotly_chart(fig_cm) # Display the confusion matrix plot

    # Additional information for debugging
    st.write(f"y_test shape: {y_test.shape}, unique values: {np.unique(y_test)}")   # Display the shape and unique values of y_test
    st.write(f"y_pred shape: {y_pred.shape}, unique values: {np.unique(y_pred)}")   # Display the shape and unique values of y_pred