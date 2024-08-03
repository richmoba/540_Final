class Config:
    FALLS_CSV_PATH = r'C:\Users\richmoba\OneDrive - Duke University\AI\Final_Project_stuff\data\ur_fall_detection_dataset\extracted_features\urfall-cam0-falls.csv'
    ADLS_CSV_PATH = r'C:\Users\richmoba\OneDrive - Duke University\AI\Final_Project_stuff\data\ur_fall_detection_dataset\extracted_features\urfall-cam0-adls.csv'
    FEATURES = ['HeightWidthRatio', 'MajorMinorRatio', 'BoundingBoxOccupancy', 'MaxStdXZ', 'HHmaxRatio', 'H', 'D', 'P40']
    RF_MODEL_PATH = 'models/random_forest_model.pkl'
    DL_MODEL_PATH = 'models/deep_learning_model.h5'