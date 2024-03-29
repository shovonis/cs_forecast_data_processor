import os
import uuid
import pandas as pd
import util.helper as helper
import datetime
import numpy as np


def init_data_files(data_src_path):
    image_dir = data_src_path + 'Frames'
    frame_list = helper.get_image_file_names(image_dir)
    verbal_feedback_file = data_src_path + '/' + 'verbal_global.csv'
    verbal_feedbacks = helper.read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')
    eye_tracking_data = helper.read_file(data_src_path + '/' + 'eye_tracking.csv', time_format='%H-%M-%S-%f',
                                         date_time_required=False)
    head_tracking_data = helper.read_file(data_src_path + '/' + 'head_tracking.csv', time_format='%H-%M-%S-%f',
                                          date_time_required=False)
    return eye_tracking_data, frame_list, head_tracking_data, verbal_feedbacks


def define_class_rule(verbal_feedback):
    class_directory = ''
    if min_fms <= float(verbal_feedback) <= class_rule["low"]:
        class_directory = '0'  # No Sickness (None)
    if class_rule['low'] < float(verbal_feedback) <= class_rule['medium']:
        class_directory = '1'  # Low sickness
    if class_rule['medium'] < float(verbal_feedback) <= class_rule['high']:
        class_directory = '2'  # Moderate sickness
    if class_rule['high'] < float(verbal_feedback) <= max_fms:
        class_directory = '3'  # High sickness

    return class_directory


def save_head_tracking_data(data_save_path, head_tracking_data, time_of_interest, unique_id, cs_severity, fms):
    head_dir = data_save_path + '/head/'
    if not os.path.exists(head_dir):
        os.makedirs(head_dir)
    head_data_file = '/head-' + unique_id + '.csv'
    h_file = head_dir + head_data_file
    data = head_tracking_data[head_tracking_data['Time'].isin(time_of_interest)]

    data = data.groupby('Time').mean()
    data.insert(1, "fms", fms)
    data.insert(2, "cs_severity_class", cs_severity)

    data.to_csv(h_file, index=True,
                columns=['fms', "cs_severity_class", 'HeadQRotationX', 'HeadQRotationY', 'HeadQRotationZ',
                         'HeadQRotationW', 'HeadEulX', 'HeadEulY', 'HeadEulZ'], float_format='%.3f')
    return head_data_file


def save_eye_tracking_data(data_save_path, eye_tracking_data, time_of_interest, unique_id, cs_severity, fms):
    eye_dir = data_save_path + '/eye/'
    if not os.path.exists(eye_dir):
        os.makedirs(eye_dir)
    eye_data_file = '/eye-' + unique_id + '.csv'
    e_file = eye_dir + eye_data_file
    data = eye_tracking_data[eye_tracking_data['Time'].isin(time_of_interest)]

    data["ConvergenceValid"] = data["ConvergenceValid"].astype(bool)
    data = data[data["ConvergenceValid"] == True]
    data.drop(["#Frame", "ConvergenceValid", "Left_Eye_Closed", "Right_Eye_Closed", "LocalGazeValid",
               "WorldGazeValid"], axis=1, inplace=True)

    # Group By second to get second only value.
    data = data.groupby('Time').mean()
    data.insert(1, "fms", fms)
    data.insert(2, "cs_severity_class", cs_severity)
    data.to_csv(e_file, index=True, float_format='%.3f')

    return eye_data_file


def prepare_hr_data(physio_data_path, toi):
    # Process HR data
    hr_file = physio_data_path + "/HR.csv"
    hr_data = pd.read_csv(hr_file, header=None)
    date_time = pd.Timestamp(float(hr_data.iloc[0]), unit='s', tz='US/Central')  # Reset timezone to US/Central
    hr_data = hr_data.drop([0, 1])
    hr_data.columns = ["HR"]
    hr_time = []
    for i in range(len(hr_data)):
        next_time = date_time + datetime.timedelta(0, i)
        hr_time.append(next_time.strftime('%I-%M-%S'))
    hr_data["Time"] = hr_time
    filtered_hr_data = hr_data[hr_data["Time"].isin(toi)]

    return filtered_hr_data


def prepare_eda_data(physio_data_path, toi):
    # Process HR data
    eda_file = physio_data_path + "/EDA.csv"
    eda_data = pd.read_csv(eda_file, header=None)
    date_time = pd.Timestamp(float(eda_data.iloc[0]), unit='s', tz='US/Central')  # Reset timezone to US/Central
    eda_data = eda_data.drop([0, 1])
    eda_data.columns = ["EDA"]
    eda_data = eda_data.groupby(np.arange(len(eda_data)) // 4).mean()
    eda_time = []
    for i in range(len(eda_data)):
        next_time = date_time + datetime.timedelta(0, i)
        eda_time.append(next_time.strftime('%I-%M-%S'))
    eda_data["Time"] = eda_time
    filtered_eda_data = eda_data[eda_data["Time"].isin(toi)]
    print(filtered_eda_data.shape)
    return filtered_eda_data


def save_physio_data(data_save_path, time_of_interest, individual, unique_id, cs, fms):
    physio_data_path = physiological_data_path + "/" + individual
    hr_data_save_path = data_save_path + "/hr"
    eda_data_save_path = data_save_path + "/eda"
    if not os.path.exists(hr_data_save_path):
        os.makedirs(hr_data_save_path)
        os.makedirs(eda_data_save_path)
    hr_file_name = '/hr-' + unique_id + '.csv'
    eda_file_name = '/eda-' + unique_id + '.csv'
    full_path = hr_data_save_path + hr_file_name
    eda_full_path = eda_data_save_path + eda_file_name

    hr_data = prepare_hr_data(physio_data_path, time_of_interest)
    eda_data = prepare_eda_data(physio_data_path, time_of_interest)
    hr_data.insert(1, "fms", fms)
    hr_data.insert(2, "cs_severity_class", cs)

    eda_data.insert(1, "fms", fms)
    eda_data.insert(2, "cs_severity_class", cs)
    hr_data[["Time", "fms", "cs_severity_class", "HR"]].to_csv(full_path, index=False, float_format='%.3f')
    eda_data[["Time", "fms", "cs_severity_class", "EDA"]].to_csv(eda_full_path, index=False, float_format='%.3f')

    return hr_file_name, eda_file_name


def process_data(simulation, individual, data_src_path, data_save_path, meta_data):
    eye_tracking_data, frame_list, head_tracking_data, verbal_feedbacks = init_data_files(data_src_path)
    helper.create_dir_if_not_exists(data_save_path)

    for index, verbal_feedback_frames in verbal_feedbacks.iterrows():
        frame_at_time_t = verbal_feedback_frames['Frame']
        verbal_feedback = verbal_feedback_frames['CSG']
        fms = verbal_feedback_frames['CS']
        matched_frame = helper.search_a_frame(frame_list, frame_at_time_t)
        time_of_interest = helper.get_frame_and_time_of_interest(frame_list, frame_at_time_t,
                                                                 matched_frame,
                                                                 window_size=interest_window)
        unique_id = str(uuid.uuid4())[:16]
        class_directory = define_class_rule(verbal_feedback)
        eye_name = save_eye_tracking_data(data_save_path, eye_tracking_data,
                                          time_of_interest, unique_id, class_directory, fms)
        head_name = save_head_tracking_data(data_save_path, head_tracking_data, time_of_interest, unique_id,
                                            class_directory, fms)

        hr_file_name, eda_file_name = save_physio_data(data_save_path, time_of_interest, individual, unique_id,
                                                       class_directory, fms)

        meta_data = meta_data.append({'uid': unique_id, 'individual': individual, 'simulation': simulation,
                                      'eye': eye_name, 'head': head_name, 'hr': hr_file_name, 'eda': eda_file_name,
                                      'cs_severity_class': class_directory, 'fms': fms}, ignore_index=True)
    return meta_data


def start_data_processing(data_path, data_save_directory, make_class=False):
    participant_list = os.listdir(data_path)
    print("Participants List: ", participant_list)
    meta_data = pd.DataFrame(columns=['uid', 'individual', 'simulation', 'eye', 'head', 'cs_severity_class', 'fms'])
    meta_file = data_save_directory + 'meta_data.csv'

    for participant in participant_list:
        participant_path = os.path.join(data_path, participant + '/')
        simulations = os.listdir(participant_path)
        for sim in simulations:
            print(f"Processing individual- {participant} in {sim}")
            ind_data_save_dir = os.path.join(data_save_directory, participant + '/' + sim)
            # Creating data Save Directories
            if not os.path.exists(ind_data_save_dir):
                os.makedirs(ind_data_save_dir)

            # Individual Raw Data Path
            individual_raw_data_path = os.path.join(participant_path, sim + '/')
            print("Individual Raw Data directory: ", individual_raw_data_path)
            print("Participants Data Save Dir: ", ind_data_save_dir)
            meta_data = process_data(sim, participant, individual_raw_data_path, ind_data_save_dir,
                                     meta_data)

            meta_data.to_csv(meta_file)


# ................................................. Main File ..............................................
if __name__ == "__main__":
    interest_window = 30
    min_fms = 0.00
    max_fms = 10.00
    fps = 20
    path = '/media/save-lab/Data/data/data_raw/hmd_data'
    physiological_data_path = "/media/save-lab/Data/data/data_raw/physiological_data"

    data_save_dir = '../../processed_data/forecast_data/'
    class_rule = {'low': 0.66, 'medium': 1.0, 'high': 2.0}  # See analysis of verbal feedback file
    start_data_processing(path, data_save_dir, make_class=False)
