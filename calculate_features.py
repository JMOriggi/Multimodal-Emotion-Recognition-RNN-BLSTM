import wave
import numpy as np
import math
import os
import cf


def calculate_features(frames, freq, options):
    n = len(frames)
    window_sec = 0.2
    window_n = int(freq * window_sec)
    use_derivatives = False

    st_f = cf.stFeatureExtraction(frames, freq, window_n, window_n / 2)
    
    print(st_f.shape)
    
    return st_f


























