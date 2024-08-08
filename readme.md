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

to run it from the web use this link:
https://540final-syx32dxwbxuhsogvrgukgp.streamlit.app/ 
Streamlit (540final-syx32dxwbxuhsogvrgukgp.streamlit.app)


Run the main script:
streamlit run main.py

This will start the Streamlit app. Follow the on-screen instructions to train models and start real-time monitoring.
there are sample files of walling in falling in the github folder called "Sample Videos"
I noteced that If I run it locally with the video files on the local hard drive it properly detects falls, but if I run it with the video files on a external hard drive it does not.
and when I run it from the web it detects walking as falling... still need to work on that.  

## Features

- Data loading and preprocessing
- Training of Random Forest and Deep Learning models
- Model evaluation with performance metrics and confusion matrices
- Real-time fall detection using the Orbbec 3D Depth Camera


##Bugs
Noticed a few bugs with it depending on the machine that I run it on:
When I run it on my lower powered machine and with the sample video files on external  hard drive it detects falls when there are none.

But if I moved the sample files to the C: drive it worked. 

Noticing the same thing on the web hosted version.. 
I will update the readme to reflect this as well. 

Streamlit (540final-syx32dxwbxuhsogvrgukgp.streamlit.app)
## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
