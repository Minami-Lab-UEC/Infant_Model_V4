apt update
pip install -U pip
apt install -y cmake ffmpeg
pip install scikit_video opencv-python
cd opencv-3.4.16/build/
make -j4
make install
ldconfig
cd ../../densetrack/
pip install .
cd ../