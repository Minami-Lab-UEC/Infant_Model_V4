docker run --gpus all -it \
	--volume /home/takeshita/Infant_Model_V4:/home/takeshita/Infant_Model_V4 \
	-w /home/takeshita/Infant_Model_V4 \
	-p 6006:6006 \
	takeshita \
	bash
