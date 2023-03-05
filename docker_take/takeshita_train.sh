docker run --gpus all -d \
	--volume /home/takeshita/Infant_Model_V4:/home/takeshita/Infant_Model_V4 \
	-w /home/takeshita/Infant_Model_V4 \
	takeshita \
	python proposed_er_except.py
