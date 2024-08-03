import streamlit as st
from scripts.data_loader import load_data, preprocess_data
from scripts.model import train_random_forest, train_deep_learning, evaluate_model
from scripts.real_time_monitoring import real_time_monitoring
from scripts.config import Config
from scripts.utils import check_gpu
import joblib

def main():
    st.title("Fall Detection with Orbbec 3D Depth Camera")

    check_gpu()
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    rf_model = train_random_forest(X_train, y_train)
    dl_model = train_deep_learning(X_train, y_train)

    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    evaluate_model(dl_model, X_test, y_test, "Deep Learning")

    joblib.dump(rf_model, Config.RF_MODEL_PATH)
    dl_model.save(Config.DL_MODEL_PATH)

    if st.button('Start Real-time Monitoring'):
        real_time_monitoring(rf_model, dl_model)

if __name__ == "__main__":
    main()