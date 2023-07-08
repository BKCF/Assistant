git clone https://github.com/152334H/tortoise-tts-fast
cd tortoise-tts-fast
pip3 uninstall tortoise -y

Rem Remove the following line from requirements
Rem bigvgan @ git+https://github.com/152334H/BigVGAN.git@HEAD ; python_version >= "3.8" and python_full_version != "3.9.7" and python_version < "3.12"
Rem pip install scipy==1.0.0
pip3 install -r ../my-requirements.txt
python -m pip install -e .
pip3 install git+https://github.com/152334H/BigVGAN.git