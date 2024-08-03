from setuptools import setup, find_packages

setup(
    name="fall_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow',
        'joblib',
        'plotly',
        'streamlit',
        'opencv-python',
    ],
)