import os
import numpy as np
import skvideo.io
import densetrack

# eat * 6, throw * 3, read * 3, stack * 3
video_file_name_list = [
        'Moments_in_Time_Mini/training/eating/flickr-4-6-4-1-3-7-8-7-6246413787_14.mp4',
        # 'Moments_in_Time_Mini/training/eating/getty-boy-eating-strawberries-video-id472619821_6.mp4',
        'Moments_in_Time_Mini/training/eating/getty-dinner-under-the-open-sky-video-id695443296_1.mp4',
        'Moments_in_Time_Mini/training/eating/getty-boy-eating-a-red-apple-video-id473087719_3.mp4',
        'Moments_in_Time_Mini/training/eating/flickr-5-7-6-1-2-6-0-1-5357612601_17.mp4',
        'Moments_in_Time_Mini/training/eating/flickr-7-4-6-0-1-1-2-9-19874601129_3.mp4',
        'Moments_in_Time_Mini/training/eating/getty-asia-girls-eating-video-id482673157_9.mp4',
        'Moments_in_Time_Mini/training/throwing/flickr-0-1-4-4-3-2-0-2-2601443202_4.mp4',
        'Moments_in_Time_Mini/training/throwing/flickr-0-6-1-1-3-6-2-6-7006113626_2.mp4',
        'Moments_in_Time_Mini/training/throwing/flickr-6-0-2-7670037602_30.mp4',
        'Moments_in_Time_Mini/training/reading/flickr-0-1-6-8-7-5-2-2-4701687522_28.mp4',
        'Moments_in_Time_Mini/training/reading/flickr-4-0-7-0-0-0-5-3-14240700053_15.mp4',
        'Moments_in_Time_Mini/training/reading/flickr-4-0-8-1-0-0-0-4-15240810004_32.mp4',
        'Moments_in_Time_Mini/training/stacking/yt-ZAvMdV-CFS0_19.mp4',
        'Moments_in_Time_Mini/training/stacking/yt-VpjLW-SxvuM_30.mp4',
        'Moments_in_Time_Mini/training/stacking/yt-SwrVHlveS0A_24.mp4',
]

DATA_DIR = 'track_sample_100over_select'
# MOVIE_DIR = 'Moments_in_Time_Mini/training'
# VERB_LIST = ['eating', 'biting', 'playing+sports', 'studying', 'assembling', 'stacking', 'throwing', 'reading']
# VERB_LIST = ['eating']

# for verb in VERB_LIST:
        # video_file_name_list = os.listdir(os.path.join(MOVIE_DIR, verb))
        # os.makedirs(os.path.join(DATA_DIR, verb), exist_ok=True)
for video_file_name in video_file_name_list:
        video_frames = skvideo.io.vreader(fname=video_file_name, as_grey=True)
        video_gray = np.stack([np.reshape(x, x.shape[1:3])
                        for x in video_frames]).astype(np.uint8, copy=False)
        tracks = densetrack.densetrack(video_gray, track_length=15
        , adjust_camera=True)
        if len(tracks) < 100:
                print('track_sample_100under! : ', video_file_name, ', ', len(tracks))
                continue
        
        head, tail = os.path.split(video_file_name)
        name = os.path.splitext(tail)[0]
        # np.save(os.path.join(DATA_DIR, verb, name + '-traj'), tracks)
        np.save(os.path.join(DATA_DIR, name + '-traj'), tracks)