#!/bin/sh
# takeshita_testでもよい dense_track_docker.tarからimageを作成する
# cat dense_track_docker.tar | docker import - takeshita_test:latestで.tarからイメージ作成

docker exec -it densetrack /bin/bash
