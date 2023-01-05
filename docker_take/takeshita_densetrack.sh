docker run --gpus all -d \
	--volume /home/takeshita/Infant_Model_V4:/home/takeshita/Infant_Model_V4 \
	-w /home/takeshita/Infant_Model_V4 \
	takeshita \
	bash -c './takeshita_docker_init.sh && python densetrack_test_takeshita.py'
