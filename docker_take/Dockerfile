FROM tensorflow/tensorflow:latest-gpu

ADD requirements.txt requirements.txt
# あらかじめ requirements.txt を ADD or COPY などで
# docker 内部に送っておくことが必要

# RUN chmod 777 requirements.txt

# RUN pip install -U pip &&\
#   pip install --no-cache-dir IPython matplotlib pandas scipy sklearn tensorflow_addons
RUN pip install -U pip &&\
  pip install --no-cache-dir -r requirements.txt