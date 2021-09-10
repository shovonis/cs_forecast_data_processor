import os
import uuid
import pandas as pd
import util.helper as helper


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


def save_head_tracking_data(data_save_path, head_tracking_data, time_of_interest, unique_id):
    head_dir = data_save_path + '/head/'
    if not os.path.exists(head_dir):
        os.makedirs(head_dir)
    head_data_file = '/head-' + unique_id + '.csv'
    h_file = head_dir + head_data_file
    data = head_tracking_data[head_tracking_data['Time'].isin(time_of_interest)]
    data.to_csv(h_file, index=False, columns=['#Frame', 'HeadQRotationX', 'HeadQRotationY', 'HeadQRotationZ',
                                              'HeadQRotationW'])
    return head_data_file


def save_eye_tracking_data(data_save_path, eye_tracking_data, time_of_interest, unique_id):
    eye_dir = data_save_path + '/eye/'
    if not os.path.exists(eye_dir):
        os.makedirs(eye_dir)
    eye_data_file = '/eye-' + unique_id + '.csv'
    e_file = eye_dir + eye_data_file
    data = eye_tracking_data[eye_tracking_data['Time'].isin(time_of_interest)]
    data.to_csv(e_file, index=False, columns=['#Frame', 'Convergence_distance', 'LeftPupilDiameter',
                                              'RightPupilDiameter', 'NrmSRLeftEyeGazeDirX', 'NrmSRLeftEyeGazeDirY',
                                              'NrmSRLeftEyeGazeDirZ', 'NrmSRRightEyeGazeDirX',
                                              'NrmSRRightEyeGazeDirY', 'NrmSRRightEyeGazeDirZ'])

    return eye_data_file


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
                                          time_of_interest, unique_id)
        head_name = save_head_tracking_data(data_save_path, head_tracking_data, time_of_interest, unique_id)

        meta_data = meta_data.append({'uid': unique_id, 'individual': individual, 'simulation': simulation,
                                      'eye': eye_name, 'head': head_name,
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
    path = '/media/save-lab/Data/data/data_raw'
    data_save_dir = '../../processed_data/forecast_data/'
    class_rule = {'low': 0.66, 'medium': 1.0, 'high': 2.0}  # See analysis of verbal feedback file
    start_data_processing(path, data_save_dir, make_class=False)
