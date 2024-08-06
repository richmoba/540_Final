import pandas as pd     # Importing the necessary libraries              
import numpy as np      # Importing the necessary libraries            
from sklearn.model_selection import train_test_split    # Importing the necessary libraries       
from sklearn.preprocessing import StandardScaler        # Importing the necessary libraries
from scripts.config import Config           # Importing the necessary libraries
import streamlit as st      # Importing the necessary libraries
from sklearn.preprocessing import LabelEncoder      # Importing the necessary libraries    

def load_data():    # Function to load the data        
    try:    # Try block to handle exceptions        
        falls_data = pd.read_csv(Config.FALLS_CSV_PATH)   # Load the falls data from the CSV file     
        adls_data = pd.read_csv(Config.ADLS_CSV_PATH)   # Load the ADLS data from the CSV file
        
        data = pd.concat([falls_data, adls_data], axis=0, ignore_index=True)    # Concatenate the falls and ADLS data
        
        st.write(f"Data loaded successfully. Shape: {data.shape}")  # Display a success message
        st.write(f"Columns: {data.columns.tolist()}")   # Display the columns of the data
        
        return data    # Return the loaded data
    except Exception as e:  # Except block to handle exceptions
        st.error(f"Error loading data: {str(e)}")   # Display an error message
        return None    # Return None

def preprocess_data(data):  # Function to preprocess the data
    if data is None or data.empty:  # If the data is None or empty
        st.error("No data to preprocess.")  # Display an error message
        return None, None, None, None   # Return None

    try:    # Try block to handle exceptions
        st.write(f"Preprocessing data. Original shape: {data.shape}")   # Display a message
        
        X = data[Config.FEATURES]   # Extract the features
        y = data['label']   # Extract the labels
        
        # Convert labels to binary (assuming 'fall' is the positive class)
        le = LabelEncoder()   # Create a LabelEncoder object
        y = le.fit_transform(y)  # Fit and transform the labels
        
        st.write(f"Features shape: {X.shape}")  # Display the shape of the features
        st.write(f"Labels shape: {y.shape}")    # Display the shape of the labels
        st.write(f"Unique labels: {np.unique(y)}")  # Display the unique labels
        
        # Remove rows with NaN or infinite values
        mask = np.isfinite(X).all(axis=1)   # Create a mask to remove NaN or infinite values
        X = X[mask] # Apply the mask to the features
        y = y[mask]     # Apply the mask to the labels
        
        st.write(f"Shape after removing NaN/inf values: {X.shape}")  # Display the shape after removing NaN or infinite values
        
        scaler = StandardScaler()   # Create a StandardScaler object
        X_scaled = scaler.fit_transform(X)  # Fit and transform the features
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)    # Split the data into training and testing sets
        
        st.write(f"Train set shape: {X_train.shape}")   # Display the shape of the training set
        st.write(f"Test set shape: {X_test.shape}")    # Display the shape of the testing set
        
        return X_train, X_test, y_train, y_test   # Return the preprocessed data
    except Exception as e:  # Except block to handle exceptions
        st.error(f"Error preprocessing data: {str(e)}")  # Display an error message
        return None, None, None, None   # Return None