import os

import numpy as np
import pandas as pd
import util.helper as helper


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
            verbal_feedbacks = helper.read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')
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
            verbal_feedbacks = helper.read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')

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
            verbal_feedback_file = individual_raw_data_path + 'verbal_global.csv'
            verbal_feedbacks = helper.read_file(verbal_feedback_file, time_format='%Y.%m.%d %H:%M:%S:%f')
            cs = verbal_feedbacks['CSG']  # TODO: Convert to GCS
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
    window = 30
    min_fms = 0.00
    max_fms = 10.00
    fps = 20
    path = 'data/raw/'
    data_save_dir = 'data/forecast_data/'
    class_rule = {'low': 0.66, 'medium': 1.0, 'high': 2.0}

    process_verbal_feedback(path)
    get_class_rule(path)
# ................................................. SETUP CONFIGURATIONS END ...........................................
