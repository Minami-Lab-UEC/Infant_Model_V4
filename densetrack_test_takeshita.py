import os
import numpy as np
import skvideo.io
import densetrack

# eat * 6, throw * 3, read * 3, stack * 3
video_file_name_list = [
        'Moments_in_Time_Mini/training/biting/getty-baby-teething-problem-video-id692394240_1.mp4',
	'Moments_in_Time_Mini/training/biting/getty-blond-haired-woman-eating-an-apple-in-her-kitchen-video-id73246632_5.mp4',
	'Moments_in_Time_Mini/training/biting/getty-blond-haired-woman-laughing-and-smiling-whilst-eating-an-apple-video-id73246634_21.mp4',
	'Moments_in_Time_Mini/training/biting/getty-boy-eats-a-hamburger-video-id101300656_2.mp4',
	'Moments_in_Time_Mini/training/biting/getty-boy-eats-apple-video-id473102205_11.mp4',
	'Moments_in_Time_Mini/training/biting/getty-brother-and-sister-eating-healthy-snacks-and-reading-book-at-kitchen-video-id1002-26_12.mp4',
        'Moments_in_Time_Mini/training/playing+sports/getty-father-teaching-his-sons-how-to-play-american-football-slow-motion-video-id467201725_24.mp4',
	'Moments_in_Time_Mini/training/playing+sports/getty-man-throwing-football-in-park-provo-utah-usa-video-id97374053_4.mp4',
	'Moments_in_Time_Mini/training/playing+sports/giphy-JV93Bi5lbWnle_2.mp4',
	'Moments_in_Time_Mini/training/studying/getty-boy-reading-book-on-floor-in-school-library-video-id454427828_17.mp4',
	'Moments_in_Time_Mini/training/studying/getty-college-student-with-books-researching-in-library-video-id595456004_4.mp4',
	'Moments_in_Time_Mini/training/studying/getty-children-studying-video-id156870300_1.mp4',
	'Moments_in_Time_Mini/training/stacking/getty-workers-stacking-crates-of-tnt-atop-wooden-platform-for-the-trinity-video-idmr_00092377_18.mp4',
	'Moments_in_Time_Mini/training/stacking/getty-workers-moving-stacks-of-50-dollar-bills-on-table-kansas-city-kansas-video-id125128656_10.mp4',
        'Moments_in_Time_Mini/training/stacking/getty-little-girl-playing-with-blocks-video-id488280395_13.mp4'
]

DATA_DIR = 'test_densetrack'
MOVIE_DIR = 'Moments_in_Time_Mini/training'
# VERB_LIST = ['eating', 'biting', 'playing+sports', 'studying', 'assembling', 'stacking', 'throwing', 'reading/']
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
        if len(tracks) == 0:
                continue
        # print('track_length_90over! : ', video_file_name)
        head, tail = os.path.split(video_file_name)
        name = os.path.splitext(tail)[0]
        # np.save(os.path.join(DATA_DIR, verb, name + '-traj'), tracks)
        np.save(os.path.join(DATA_DIR, name + '-traj'), tracks)