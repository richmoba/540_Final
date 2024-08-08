from setuptools import setup, find_packages # Always prefer setuptools over distutils

setup(  # Setup the
    name="fall_detection",  # Package name
    version="0.1",  # Package version
    packages=find_packages(),   # Find all packages
    install_requires=[  # List of dependencies
        'pandas',   # Pandas library
        'numpy',    # NumPy library
        'scikit-learn', # Scikit-learn library
        'tensorflow',   # TensorFlow library
        'joblib',   # Joblib library
        'plotly',   # Plotly library
        'streamlit',    # Streamlit library
        'opencv-python',    # OpenCV library
        'opencv-python-headless' #  OpenCV library headless
       ],  # End of dependencies
)   # End of setup