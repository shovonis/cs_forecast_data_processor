import data_processor as dp

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Video Save Config
    frame_size = (512, 256)
    window = 10
    fps = 20
    path = '../../data/raw/'
    data_save_dir = '../../data/processed/'
    class_rule = {'low': 1, 'medium': 4, 'high': 10}
    dp.start_data_processing(path, data_save_dir, make_class=True)
