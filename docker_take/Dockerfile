FROM tensorflow/tensorflow:latest-gpu

ADD requirements.txt requirements.txt
# あらかじめ requirements.txt を ADD or COPY などで
# docker 内部に送っておくことが必要
# Infant_Model_V4にいることが前提

# RUN chmod 777 requirements.txt

# RUN pip install -U pip &&\
#   pip install --no-cache-dir IPython matplotlib pandas scipy sklearn tensorflow_addons
RUN pip install --upgrade pip
RUN apt update
RUN apt install -y cmake ffmpeg
RUN pip install -U pip &&\
  pip install --no-cache-dir -r requirements.txt
# opencvのbuildとdensetrackのsetup
RUN cd opencv-3.4.16
RUN cd build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE -D OPENCV_ENABLE_NONFREE=ON -D WITH_GTK=ON ..
RUN cmake --build .
RUN make -j4
RUN make install
RUN ldconfig
RUN cd ../../densetrack
RUN pip install .


