import os
import numpy as np
import skvideo.io
import densetrack

video_file_name_list = [
        'Moments_in_Time_Mini/training/crying/getty-asian-baby-cry-video-id805355932_4.mp4',
        'Moments_in_Time_Mini/training/crying/flickr-5-3-2-9-5-1-4-2-6153295142_19.mp4',
        # 'Moments_in_Time_Mini/training/crying/getty-sad-real-man-crying-video-id688113760_10.mp4',
        'Moments_in_Time_Mini/training/crying/yt-tvvMDShjkc8_44.mp4',
        'Moments_in_Time_Mini/training/walking/flickr-7-7-6-0-1-0-9-1-4877601091_1.mp4',
        #'Moments_in_Time_Mini/training/walking/flickr-1-6-7-3-3-8-3-6-18716733836_40.mp4',
        'Moments_in_Time_Mini/training/walking/yt-X4tzjjxeJVY_379.mp4',
        'Moments_in_Time_Mini/training/walking/4-4-5-1-5-0-0-7-3244515007_30.mp4',
        'Moments_in_Time_Mini/training/running/flickr-4-3-2-7-4-8-8-6-16943274886_9.mp4',
        'Moments_in_Time_Mini/training/running/flickr-3-7-5-6078422375_7.mp4',
        'Moments_in_Time_Mini/training/running/flickr-0-7-2-6-2-1-1-1-20107262111_3.mp4'
]

data_dir = 'test_densetrack'
for video_file_name in video_file_name_list:
        video_frames = skvideo.io.vreader(fname=video_file_name, as_grey=True)
        video_gray = np.stack([np.reshape(x, x.shape[1:3])
                        for x in video_frames]).astype(np.uint8, copy=False)
        tracks = densetrack.densetrack(video_gray, adjust_camera=True)
        head, tail = os.path.split(video_file_name)
        name = os.path.splitext(tail)[0]
        np.save(os.path.join(data_dir, name + '-traj'), tracks)