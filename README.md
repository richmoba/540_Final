# 540_Final Richmond Baker
Richmond.Baker@duke.edu
404-309-3296
540 Final project Python program that uses a camera to monitor human movement  
# Fall Detection with Camera's

This project implements a fall detection system using an  regular 2D cameras. It utilizes both Random Forest and Deep Learning models to classify falls based on extracted features from depth frames.

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
git clone https://github.com/richmoba/540_Final.git
cd fall-detection
2. Install the required packages:
pip install -r requirements.txt


## Usage

Run the main script:
Streamlit run main.py

This will start the Streamlit app. will show you how the models are performing and then will give you a chance to drop a recording and have it tell you if the video showes you someone walking or falling The real-time monitoring portion is not currently working but it will review video that has been saved and tell you whats going on.

## Features

- Data loading and preprocessing
- Training of Random Forest and Deep Learning models
- Model evaluation with performance metrics and confusion matrices
- Real-time fall detection using 2d Camera

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
