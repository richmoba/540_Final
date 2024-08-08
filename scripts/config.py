# class Config:   # Class to store configuration variables       
#     FALLS_CSV_PATH = r'C:\Users\richmoba\OneDrive - Duke University\AI\Final_Project_stuff\data\ur_fall_detection_dataset\extracted_features\urfall-cam0-falls.csv'   # Path to the CSV file containing fall data   
#     ADLS_CSV_PATH = r'C:\Users\richmoba\OneDrive - Duke University\AI\Final_Project_stuff\data\ur_fall_detection_dataset\extracted_features\urfall-cam0-adls.csv'   # Path to the CSV file containing ADL data       
#     FEATURES = [    # List of features to use for training the model
#         'HeightWidthRatio',     
#         'MajorMinorRatio',
#         'BoundingBoxOccupancy',
#         'MaxStdXZ',
#         'HHmaxRatio',
#         'H',
#         'D',
#         'P40'   ]
#     RF_MODEL_PATH = 'models/random_forest_model.pkl'    # Path to save the Random Forest model
#     DL_MODEL_PATH = 'models/deep_learning_model.h5'     # Path to save the Deep Learning model
#     DL_MODEL_PATH = 'models/deep_learning_model.keras'      # Path to save the Deep Learning model
import os

class Config:
    # Get the current directory of this script
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths relative to the base directory
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'ur_fall_detection_dataset', 'extracted_features')
    
    FALLS_CSV_PATH = os.path.join(DATA_DIR, 'urfall-cam0-falls.csv')
    ADLS_CSV_PATH = os.path.join(DATA_DIR, 'urfall-cam0-adls.csv')
    
    FEATURES = [
        'HeightWidthRatio',     
        'MajorMinorRatio',
        'BoundingBoxOccupancy',
        'MaxStdXZ',
        'HHmaxRatio',
        'H',
        'D',
        'P40'
    ]
    
    RF_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
    DL_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'deep_learning_model.keras')