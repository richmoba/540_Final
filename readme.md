# Fall Detection with Orbbec 3D Depth Camera

This project implements a fall detection system using an Orbbec 3D Depth Camera. It utilizes both Random Forest and Deep Learning models to classify falls based on extracted features from depth frames.

## Project Structure

project_root/
│
├── data/
│   ├── ur_fall_detection_dataset/
│   │   └── extracted_features/
│   │       ├── urfall-cam0-falls.csv
│   │       └── urfall-cam0-adls.csv
│   └── processed/
│
├── models/
│   ├── random_forest_model.pkl
│   └── deep_learning_model.h5
│
├── scripts/
│   ├── init.py
│   ├── config.py
│   ├── utils.py
│   ├── data_loader.py
│   ├── model.py
│   └── real_time_monitoring.py
│
├── main.py
├── setup.py
├── requirements.txt
└── README.md

## Installation

1. Clone this repository:
git clone https://github.com/your-username/fall-detection.git
cd fall-detection
2. Install the required packages:
pip install -r requirements.txt


## Usage

Run the main script:
streamlit run main.py

This will start the Streamlit app. Follow the on-screen instructions to train models and start real-time monitoring.

## Features

- Data loading and preprocessing
- Training of Random Forest and Deep Learning models
- Model evaluation with performance metrics and confusion matrices
- Real-time fall detection using the Orbbec 3D Depth Camera

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
