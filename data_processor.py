import datetime
from os import walk
import pandas as pd
import cv2
import shutil, os
from util.disparity import disparity
from util.optical import optical_flow
import numpy as np
import uuid


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
    frames_at_time_plus_window = (time_of_frame + datetime.timedelta(0, 3))
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


def create_and_save_video(video_name, frames, image_directory):
    video = cv2.VideoWriter(video_name, 0, 22, frame_size)
    for frame in frames:
        current_im_dir = image_directory + '/' + frame
        video.write(cv2.imread(current_im_dir))

    cv2.destroyAllWindows()
    video.release()


def create_disparity_video(video_path, stereo_images, image_directory):
    video = cv2.VideoWriter(video_path, 0, fps, (256, 256))
    for frame in stereo_images:
        current_im_dir = image_directory + '/' + frame
        # print("Image Dir: " + current_im_dir)

        img = cv2.imread(current_im_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        disp_img = disparity.generate_disparity_sgm(img)

        # plt.imshow(disp_img, 'gray')
        # plt.show()

        disp_img = np.uint8(255 * disp_img)
        video.write(disp_img)

    cv2.destroyAllWindows()
    video.release()


def save_frames(dest, frames, image_directory):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for frame in frames:
        current_im_dir = image_directory + '/' + frame
        shutil.copy(current_im_dir, dest)


def process_data_per_class(simulation, individual, data_src_path, data_save_path, meta_data):
    # Get all Frames
    image_dir = data_src_path + 'Frames'
    frame_list = get_image_file_names(image_dir)

    # Get Verbal Feedback
    verbal_feedback_file = data_src_path + '/' + 'verbal_global.csv'
    verbal_feedbacks = read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')

    eye_tracking_data = read_file(data_src_path + '/' + 'eye_tracking.csv', time_format='%H-%M-%S-%f',
                                  date_time_required=False)
    head_tracking_data = read_file(data_src_path + '/' + 'head_tracking.csv', time_format='%H-%M-%S-%f',
                                   date_time_required=False)

    # Make Sure to create the data save directory
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    os.makedirs(data_save_path + '/clips', exist_ok=True)
    os.makedirs(data_save_path + '/disp', exist_ok=True)
    os.makedirs(data_save_path + '/optic', exist_ok=True)
    os.makedirs(data_save_path + '/eye', exist_ok=True)
    os.makedirs(data_save_path + '/head', exist_ok=True)
    os.makedirs(data_save_path + '/frames', exist_ok=True)

    for index, verbal_feedback_frames in verbal_feedbacks.iterrows():
        frame_at_time_t = verbal_feedback_frames['Frame']
        verbal_feedback = verbal_feedback_frames['CSG']
        fms = verbal_feedback_frames['CS']

        matched_frame = search_a_frame(frame_list, frame_at_time_t)
        frame_of_interest, time_of_interest = get_frame_and_time_of_interest(frame_list, frame_at_time_t, matched_frame,
                                                                             window_size=window)
        class_directory = ''
        if float(verbal_feedback) == class_rule["low"]:
            class_directory = '0'
        if class_rule['low'] < float(verbal_feedback) <= class_rule['medium']:
            class_directory = '1'
        if class_rule['medium'] < float(verbal_feedback) <= class_rule['high']:
            class_directory = '2'

        # Generate Unique File Identifier
        unique_id = str(uuid.uuid4())[:16]

        # Save the video clips
        clips_dir = data_save_path + '/clips/'
        if not os.path.exists(clips_dir):
            os.makedirs(clips_dir)
        clip_name = '/clip-' + unique_id + '.mp4'
        video_name = clips_dir + clip_name
        create_and_save_video(video_name, frame_of_interest, image_directory=image_dir)

        # Save Frames
        frame_name = '/frames/' + '/' + unique_id + '/'
        frame_save_dir = data_save_path + frame_name
        save_frames(frame_save_dir, frame_of_interest, image_directory=image_dir)

        # Save Optical  Flow
        optic_dir = data_save_path + '/optic/'
        if not os.path.exists(optic_dir):
            os.makedirs(optic_dir)
        flow_name = '/optic-' + unique_id + '.mp4'
        optical_flow_save_dir = optic_dir + flow_name
        optical_flow.get_optical_flow(video_name, optical_flow_save_dir)

        # Save Disparity Map
        disp_dir = data_save_path + '/disp/'
        if not os.path.exists(disp_dir):
            os.makedirs(disp_dir)
        disp_name = '/disp-' + unique_id + '.mp4'
        d_video_name = disp_dir + disp_name
        create_disparity_video(d_video_name, frame_of_interest, image_directory=image_dir)

        # Save Eye_tracking
        eye_dir = data_save_path + '/eye/'
        if not os.path.exists(eye_dir):
            os.makedirs(eye_dir)
        eye_name = '/eye-' + unique_id + '.csv'
        e_file = eye_dir + eye_name
        data = eye_tracking_data[eye_tracking_data['Time'].isin(time_of_interest)]
        data.to_csv(e_file, index=False, columns=['#Frame', 'Convergence_distance', 'LeftPupilDiameter',
                                                  'RightPupilDiameter', 'NrmSRLeftEyeGazeDirX', 'NrmSRLeftEyeGazeDirY',
                                                  'NrmSRLeftEyeGazeDirZ', 'NrmSRRightEyeGazeDirX',
                                                  'NrmSRRightEyeGazeDirY', 'NrmSRRightEyeGazeDirZ'])

        # # # Save Head Tracking
        head_dir = data_save_path + '/head/'
        if not os.path.exists(head_dir):
            os.makedirs(head_dir)
        head_name = '/head-' + unique_id + '.csv'
        h_file = head_dir + head_name
        data = head_tracking_data[head_tracking_data['Time'].isin(time_of_interest)]
        data.to_csv(h_file, index=False, columns=['#Frame', 'HeadQRotationX', 'HeadQRotationY', 'HeadQRotationZ',
                                                  'HeadQRotationW'])

        meta_data = meta_data.append({'uid': unique_id, 'individual': individual, 'simulation': simulation,
                                      'frame': frame_name, 'video_clip': clip_name,
                                      'optical': flow_name, 'disparity': disp_name,
                                      'eye': eye_name, 'head': head_name,
                                      'cs_class': class_directory, 'fms': fms},
                                     ignore_index=True)
    return meta_data


def start_data_processing(data_path, data_save_directory, make_class=False):
    simulations = os.listdir(data_path)
    print("Simulation List: ", simulations)
    meta_data = pd.DataFrame(
        columns=['uid', 'individual', 'simulation', 'frame', 'video_clip', 'optical', 'disparity', 'eye',
                 'head', 'cs_class', 'fms'])
    meta_file = data_save_directory + 'meta_data.csv'

    for simulation in simulations:
        simulation_path = os.path.join(data_path, simulation + '/')
        individual_list = os.listdir(simulation_path)
        for individual in individual_list:
            print(f"Processing Individual {individual} in simulation {simulation}")
            indiv_data_save_dir = os.path.join(data_save_directory, simulation + '/' + individual)

            # Creating data Save Directories
            if not make_class:
                if not os.path.exists(indiv_data_save_dir):
                    os.makedirs(indiv_data_save_dir)

            # Individual Raw Data Path
            individual_raw_data_path = os.path.join(simulation_path, individual + '/')

            meta_data = process_data_per_class(simulation, individual, individual_raw_data_path, data_save_directory,
                                               meta_data)
            meta_data.to_csv(meta_file)


def process_verbal_feedback(path):
    simulations = os.listdir(path)
    print("Simulation List: ", simulations)
    fms = 0
    for simulation in simulations:
        simulation_path = os.path.join(path, simulation + '/')
        individual_list = os.listdir(simulation_path)
        print(simulation)
        total_indiv = len(individual_list)
        for individual in individual_list:
            individual_raw_data_path = os.path.join(simulation_path, individual + '/')
            verbal_feedback_file = individual_raw_data_path + 'verbal_feedback.csv'
            verbal_feedbacks = read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')
            cs = verbal_feedbacks['CS']
            if cs.shape[0] < 13:
                i = cs.shape[0]
                while i < 13:
                    cs._set_value(i, 0)
                    i += 1
            fms = fms + cs.to_numpy()

        print("FMS: ", fms)

        for individual in individual_list:
            individual_raw_data_path = os.path.join(simulation_path, individual + '/')
            verbal_feedback_file = individual_raw_data_path + 'verbal_feedback.csv'
            verbal_feedbacks = read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')

            CGS = pd.DataFrame({'CSG': fms / total_indiv})
            CGS = pd.concat([verbal_feedbacks, CGS], axis=1)
            CGS['Time'] = pd.to_datetime(CGS['Time'], format='%Y.%m.%d %H:%M:%S:%f')
            CGS = CGS.dropna()
            print(CGS)
            save_file = individual_raw_data_path + 'verbal_global.csv'
            # CGS.to_csv(save_file, date_format='%Y.%m.%d %H:%M:%S:%f', index=False)

        fms = 0


def get_class_rule(path):
    simulations = os.listdir(path)
    tmp = []
    for simulation in simulations:
        simulation_path = os.path.join(path, simulation + '/')
        individual_list = os.listdir(simulation_path)
        print(simulation)

        for individual in individual_list:
            individual_raw_data_path = os.path.join(simulation_path, individual + '/')
            verbal_feedback_file = individual_raw_data_path + 'verbal_feedback.csv'
            verbal_feedbacks = read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')
            cs = verbal_feedbacks['CS']  # TODO: Convert to GCS
            tmp.extend(cs.to_numpy())

    all_cs = np.array(tmp)
    print(np.percentile(all_cs, 25))
    print(np.percentile(all_cs, 50))
    print(np.percentile(all_cs, 75))
    # print("Shape: ", tmp.shape)


# ................................................. SETUP CONFIGURATIONS ..............................................
if __name__ == "__main__":
    # Video Save Config
    frame_size = (512, 256)
    window = 10
    fps = 20
    path = 'data/raw/'
    data_save_dir = 'data/processed/'
    class_rule = {'low': 0.0, 'medium': 1.0, 'high': 2.0}

    # process_verbal_feedback(path)
    get_class_rule(path)
    # start_data_processing(path, data_save_dir, make_class=True)
# ................................................. SETUP CONFIGURATIONS END ...........................................
