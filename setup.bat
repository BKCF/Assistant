@echo off

conda create -n assistant0 python=3.10.9
conda activate assistant0

pip3 uninstall torch torchvision torchaudio -y
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

set ASSISTANT_HOME=%cd%

Rem call setup-llama-cpp.bat
Rem call setup-whisper-openvino.bat
Rem call setup-fast-tortoise.bat

pip install TTS
pip install numpy==1.24
pip install py-espeak-ng
python cudatest.py

IF ERRORLEVEL NEQ 0(
    pip3 uninstall torch torchvision torchaudio -y
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
)