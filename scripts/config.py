class Config:   # Class to store configuration variables       
    FALLS_CSV_PATH = r'C:\Users\richmoba\OneDrive - Duke University\AI\Final_Project_stuff\data\ur_fall_detection_dataset\extracted_features\urfall-cam0-falls.csv'   # Path to the CSV file containing fall data   
    ADLS_CSV_PATH = r'C:\Users\richmoba\OneDrive - Duke University\AI\Final_Project_stuff\data\ur_fall_detection_dataset\extracted_features\urfall-cam0-adls.csv'   # Path to the CSV file containing ADL data       
    FEATURES = [    # List of features to use for training the model
        'HeightWidthRatio',     
        'MajorMinorRatio',
        'BoundingBoxOccupancy',
        'MaxStdXZ',
        'HHmaxRatio',
        'H',
        'D',
        'P40'   ]
    RF_MODEL_PATH = 'models/random_forest_model.pkl'    # Path to save the Random Forest model
    DL_MODEL_PATH = 'models/deep_learning_model.h5'     # Path to save the Deep Learning model
    DL_MODEL_PATH = 'models/deep_learning_model.keras'      # Path to save the Deep Learning model