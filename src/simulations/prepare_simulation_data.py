import os
from utils.mnist_reader import get_and_save_train_test_dataset

if __name__ == '__main__':
    # See README.md for parameter descriptons.
    data_params = {'dataset': 'mnist_784',
                   'trains_per_class': 5000,
                   'tests_per_class': 1000,
                   'trains_in_test_set': False,
                   'training_labels': None,
                   'testing_labels': None,
                   'precision_required': 7,
                   'shuffle': False,
                   'use_edge_detection': False}

    data_root_dir = '..' + os.sep + '..' + os.sep + 'datasets'
    data_sub_dir = ('split1_' + str(data_params['dataset']) + '_' + str(data_params['trains_per_class']) +
                    '_' + str(data_params['tests_per_class']))

    x_train, y_train, x_test, y_test = get_and_save_train_test_dataset(data_params=data_params,
                                                                       data_root_dir = data_root_dir,
                                                                       data_sub_dir = data_sub_dir)