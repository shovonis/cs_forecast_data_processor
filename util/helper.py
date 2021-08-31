import datetime
import os
import uuid
from os import walk
import numpy as np
import pandas as pd


def create_dir_if_not_exists(data_save_path):
    # Make Sure to create the data save directory
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    # Create data save dirs
    os.makedirs(data_save_path + '/eye', exist_ok=True)
    os.makedirs(data_save_path + '/head', exist_ok=True)


def read_file(file_name, time_format, date_time_required=True):
    data = pd.read_csv(file_name)
    if date_time_required:
        data['Time'] = pd.to_datetime(data['Time'], format=time_format)
    else:
        data['Time'] = pd.to_datetime(data['Time'], format=time_format).dt.strftime('%H-%M-%S')

    return data


def get_image_file_names(image_dir):
    _, _, filenames = next(walk(image_dir))
    return filenames


def search_a_frame(frames, frame_number):
    matching = [match for match in frames if "Frame-" + str(int(frame_number)) in match]
    return matching[0]


def get_frame_and_time_of_interest(frames, frame_number, matched_frames, window_size=15):
    # Pre Processing
    time_of_frame = matched_frames.replace("Frame-" + str(int(frame_number)) + "-", '')
    time_of_frame = time_of_frame.replace(".png", '')
    time_of_frame = datetime.datetime.strptime(time_of_frame, '%H-%M-%S-%f').strftime('%H-%M-%S')
    time_of_frame = datetime.datetime.strptime(time_of_frame, "%H-%M-%S")

    # Get time + window and time - window
    frames_at_time_plus_window = (time_of_frame + datetime.timedelta(0, 1))
    frames_at_time_minus_window = (time_of_frame + datetime.timedelta(0, -window_size))
    time_of_interests = []

    end_time = time_of_frame
    while frames_at_time_minus_window < end_time:
        time_of_interests.append(frames_at_time_minus_window.time().strftime('%H-%M-%S'))
        frames_at_time_minus_window += datetime.timedelta(0, 1)

    start_time = time_of_frame
    time_of_interests.append(start_time.time().strftime('%H-%M-%S'))
    while start_time < frames_at_time_plus_window:
        start_time += datetime.timedelta(0, 1)
        # print("Start Time + t", current_frame_time.time().strftime('%H-%M-%S'))
        time_of_interests.append(start_time.time().strftime('%H-%M-%S'))

    frame_of_interests = []
    for time in time_of_interests:
        for frame in frames:
            if time in frame:
                frame_of_interests.append(frame)

    return frame_of_interests, time_of_interests

