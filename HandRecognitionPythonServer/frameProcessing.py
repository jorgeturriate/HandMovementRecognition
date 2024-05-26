import cv2 as cv
import numpy as np
from PIL import Image


def OpticalFlow(prev_gray, frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask = np.zeros_like(frame)
    mask[..., 1] = 255
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    return gray, rgb, magnitude

def calculate_frame_similarity(frame1, frame2):
    gray_frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray_frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    similarity = cv.matchTemplate(gray_frame1, gray_frame2, cv.TM_CCOEFF_NORMED)
    return similarity[0][0]

def extract_keyframes(frames, num_keyframes=5):
    num_frames = len(frames)
    keyframes = [frames[0]]
    interval = num_frames // (num_keyframes - 1)
    for i in range(1, num_keyframes - 1):
        start_index = i * interval
        end_index = min(start_index + interval, num_frames)
        selected_frame = None
        max_similarity = -1
        for j in range(start_index, end_index):
            similarity = calculate_frame_similarity(keyframes[-1], frames[j])
            if similarity > max_similarity:
                max_similarity = similarity
                selected_frame = frames[j]
        keyframes.append(selected_frame)
    keyframes.append(frames[-1])
    return keyframes

def resize_and_concatenate_keyframes(keyframes, target_size=(180, 180)):
    resized_keyframes = [cv.resize(frame, target_size) for frame in keyframes]
    concatenated_image = cv.hconcat(resized_keyframes)
    return concatenated_image


def process_frames(frames):
    real_frames = []
    optical_frames = []
    movilities=[]

    first_frame = cv.flip(frames[0], 1)  # Flip the frame horizontally
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

    movement_threshold = 1.0  # Adjust this threshold based on your requirements

    for frame in frames:
        
        frame = cv.flip(frame, 1)  # Flip the frame horizontally
        real_frames.append(frame)
        prev_gray, optical_frame, magnitude = OpticalFlow(prev_gray, frame)
        optical_frames.append(optical_frame)
        movilities.append(np.mean(magnitude))

    average_magnitude = np.mean(movilities)

    if average_magnitude > movement_threshold:
        real_keyframes = extract_keyframes(real_frames[-30:], num_keyframes=5)
        optical_keyframes = extract_keyframes(optical_frames[-30:], num_keyframes=5)

        real_image = resize_and_concatenate_keyframes(real_keyframes)
        optical_image = resize_and_concatenate_keyframes(optical_keyframes)

        return real_image, optical_image
    else:
        return False, False