Repository for my current work on the protein scaffold filling problem

Main files: 
- autoencoder.py - implements and tests my denoising convolutional autoencoder
- lstm.py - implements and tests my LSTM-based models for the protein gap filling problem


Dependencies: 
    python 3.9
    tensorflow < 2.11
    MS Visual Studio 2019
    CUDA v.1.1X
    cuDNN?
    miniconda

    https://www.tensorflow.org/install/pip

    Installing tensorflow<2.11 with conda, which is the last version of tensorflow that can run on Windows Native
    As such, requires older versions of MS Visual Studio (2019), CUDA Toolkit 11.2, and cuDNN SDK 8.1.0

conda install:
    Install miniconda from the internet

    conda create --name tf python=3.9
    conda activate tf
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    pip install --upgrade pip
    pip install "tensorflow<2.11" 
