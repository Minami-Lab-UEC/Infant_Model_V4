import os
import numpy as np
import skvideo.io
import densetrack

# eat * 6, throw * 3, read * 3, stack * 3
video_file_name_list = [
        'Moments_in_Time_Mini/training/biting/getty-beautiful-blonde-eating-a-green-apple-video-id598747074_5.mp4',
        'Moments_in_Time_Mini/training/biting/getty-baby-boy-fruit-video-id146171604_2.mp4',
        'Moments_in_Time_Mini/training/biting/getty-baby-teething-problem-video-id692394240_1.mp4',
        'Moments_in_Time_Mini/training/biting/getty-angry-cat-biting-and-hissing-video-id489683554_2.mp4',
        'Moments_in_Time_Mini/training/biting/getty-babies-ruining-smartphones-video-id686040830_19.mp4',
        'Moments_in_Time_Mini/training/biting/getty-babies-ruining-smartphones-video-id682718098_5.mp4',
        'Moments_in_Time_Mini/training/playing+sports/getty-arizona-wranglers-quarterback-alan-risher-throwing-pass-to-teammate-video-id1377-34_2.mp4',
        'Moments_in_Time_Mini/training/playing+sports/getty-behind-view-of-boy-playing-catch-video-id647330573_30.mp4',
        'Moments_in_Time_Mini/training/playing+sports/getty-father-teaching-his-sons-how-to-play-american-football-slow-motion-video-id467201707_10.mp4',
        'Moments_in_Time_Mini/training/studying/flickr-0-7-3-6-4-6-3-2-6307364632_59.mp4',
        'Moments_in_Time_Mini/training/studying/getty-boy-doing-homework-american-fork-utah-usa-video-id93232127_4.mp4',
        'Moments_in_Time_Mini/training/studying/2jyueyJmiEQ_119.mp4',
        'Moments_in_Time_Mini/training/assembling/flickr-5-9-9-1-5-3-6-5-15559915365_32.mp4',
        'Moments_in_Time_Mini/training/stacking/getty-arranging-orange-and-pepper-slices-video-id162682612_26.mp4',
        'Moments_in_Time_Mini/training/stacking/getty-afar-men-build-stone-enclosure-on-august-16-2011-on-road-from-makele-video-id139111660_9.mp4',
]

data_dir = 'test_densetrack'
for video_file_name in video_file_name_list:
        video_frames = skvideo.io.vreader(fname=video_file_name, as_grey=True)
        video_gray = np.stack([np.reshape(x, x.shape[1:3])
                        for x in video_frames]).astype(np.uint8, copy=False)
        tracks = densetrack.densetrack(video_gray, track_length=90
        , adjust_camera=True)
        if len(tracks) != 0:
                print(video_file_name)
        head, tail = os.path.split(video_file_name)
        name = os.path.splitext(tail)[0]
        np.save(os.path.join(data_dir, name + '-traj'), tracks)