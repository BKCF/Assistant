set LLAMA_OPENBLAS=1
Rem python cudatest.py
set CMAKE_ARGS="-DLLAMA_CUBLAS=on" 
set FORCE_CMAKE=1 
pip3 uninstall llama-cpp-python -y
pip3 install llama-cpp-python --no-cache-dir