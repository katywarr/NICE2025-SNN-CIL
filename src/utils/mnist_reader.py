import os
import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from scipy import ndimage
from pathlib import Path


def get_and_save_train_test_dataset(data_root_dir: str, data_sub_dir: str, data_params: dict):
    x_train, y_train, x_test, y_test = get_train_test_dataset(data_root_dir=data_root_dir,
                                                              data_params=data_params)
    full_save_dir = os.path.join(data_root_dir, data_sub_dir)
    save_train_test_dataset(to_directory=full_save_dir,
                            x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                            data_params=data_params)
    return x_train, y_train, x_test, y_test

def save_train_test_dataset(to_directory: str,
                            x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                            data_params: dict):
    print('Saving train and test examples to directory {}. Current directory is {}.'.format(to_directory, os.getcwd()))
    if Path(os.path.join(to_directory, 'x_train.npy')).is_file():
        raise ValueError('Data already exists in the specified directory {}. '
                         'Not saving the data (to prevent overwriting). '
                         'Specify a different location or remove the data then re-run.'
                         .format(to_directory))
    if not Path(to_directory).is_dir():
        print('Directory {} does not exist. Creating it.'.format(to_directory))
        os.mkdir(to_directory)
    np.save(os.path.join(to_directory, 'x_train.npy'), x_train)
    np.save(os.path.join(to_directory, 'y_train.npy'), y_train)
    np.save(os.path.join(to_directory, 'x_test.npy'), x_test)
    np.save(os.path.join(to_directory, 'y_test.npy'), y_test)
    with open(os.path.join(to_directory, 'data_params.pkl'), 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump(data_params, f)
    print('Data examples and data description saved in directory {}.'.format(to_directory))

def load_train_test_dataset(from_directory: str):
    print('Loading dataset from directory {}. Current directory is {}.'.format(from_directory, os.getcwd()))
    if not Path(from_directory).is_dir():
        raise ValueError('Directory {} does not exist. Unable to load data.\n'
                         'Run simulation_data.py to create it.'.format(from_directory))
    x_train = np.load(os.path.join(from_directory, 'x_train.npy'))
    y_train = np.load(os.path.join(from_directory, 'y_train.npy'))
    x_test = np.load(os.path.join(from_directory, 'x_test.npy'))
    y_test = np.load(os.path.join(from_directory, 'y_test.npy'))
    with open(os.path.join(from_directory, 'data_params.pkl'), 'rb') as f:
        # noinspection PyTypeChecker
        data_params = pickle.load(f)
    return x_train, y_train, x_test, y_test, data_params


def get_train_test_dataset(data_root_dir: str, data_params: dict) -> tuple:
    try:
        dataset = data_params['dataset']
        trains_per_class = data_params['trains_per_class']
        tests_per_class = data_params['tests_per_class']
        training_labels = data_params['training_labels']
        testing_labels = data_params['testing_labels']
        shuffle_data = data_params['shuffle']
        precision_required = data_params['precision_required']
        trains_in_test = data_params['trains_in_test_set']
        edge_detection = data_params['use_edge_detection']
    except KeyError as e:
        print('Error: The following key was missing from the data parameters (data_params) required '
              'to generate the training and test data: ', e)
        raise e

    x, y = read_openml_data(data_dir=data_root_dir,
                            dataset=dataset,
                            precision_required=precision_required,
                            edge_features=edge_detection)

    print('{} data has been loaded. Preparing the training and test examples. '.format(dataset), end='')
    x_train, y_train, x_test, y_test = select_and_prepare_data(x, y, trains_per_class, tests_per_class,
                                                               trains_in_test=trains_in_test,
                                                               training_labels=training_labels,
                                                               testing_labels=testing_labels,
                                                               shuffle_data=shuffle_data)
    print('...test and train data has been prepared.')
    return x_train, y_train, x_test, y_test


def select_and_prepare_data(x: np.ndarray, y:np.ndarray, trains_per_class: int, tests_per_class: int, trains_in_test: bool = False,
                            training_labels: list = None, testing_labels: list = None, shuffle_data: bool = False) -> tuple:
    """
    Given a set of data examples and their associated labels, select and prepare training and test data according
    to the given parameters.
    :param x: (np.ndarray) the data examples.
    :param y: (np.ndarray) the labels associated with the data examples.
    :param trains_per_class: (int) the number of training examples required per class.
    :param tests_per_class: (int) the number of test examples required per class.
    :param trains_in_test: (bool) whether to include training examples in the test set. Including training
        examples in the test set can be useful to check for overfitting.
    :param training_labels: (list) the sub-set of training classes required. Set to None to include all the training
        classes. Specify a list to restrict to training examples to a subset of the classes. Defaults to None.
    :param testing_labels: (list) the subset of test classes required. Set to None to include all the test classes.
         Specify a list to restrict to training examples to a subset of the classes. Defaults to None.
    :param shuffle_data: (bool) whether to shuffle the data before splitting into training and test set. Shuffling
         will result in data that is Independent and Identically Distributed (IID). No shuffling results
         in data for Class Incremental Learning. Defaults to False (no shuffle).
    """
    # Check that there are sufficient data examples to satisfy the tests_per_class and trains_per_class
    required_examples = tests_per_class + trains_per_class
    max_examples_per_label, labels = examples_per_label(y)
    if not trains_in_test and required_examples > max_examples_per_label:
        error_string = ("Insufficent data examples for required trains {} and tests {} per label. "
                        "Maximum number of examples per label is {}."
                        .format(trains_per_class, tests_per_class, required_examples))
        raise (ValueError(error_string))
    if trains_in_test and max(trains_per_class, tests_per_class) > max_examples_per_label:
        error_string = ("Insufficent data examples for required trains {} and tests {} per label. "
                        "Maximum number of examples per label is {}. Training examples re-used for test."
                        .format(trains_per_class, tests_per_class, required_examples))
        raise (ValueError(error_string))

    idx = np.argsort(y)
    x_sorted = x[idx]
    y_sorted = y[idx]

    # If the training set is not specified, train across all the labels
    training_labels = labels if training_labels is None else training_labels
    # If the testing set is not specified, test across all the training labels
    testing_labels = labels if testing_labels is None else testing_labels

    # Prepare the numpy structures
    num_trains = len(training_labels) * trains_per_class
    num_tests = len(testing_labels) * tests_per_class
    x_train = np.empty(shape=(num_trains, x.shape[1]), dtype=int)
    y_train = np.empty(shape=(num_trains,), dtype=str)
    x_test = np.empty(shape=(num_tests, x.shape[1]), dtype=int)
    y_test = np.empty(shape=(num_tests,), dtype=str)

    train_start_idx = 0
    test_start_idx = 0
    for label in labels:
        if label in training_labels or label in testing_labels:
            # Get all the examples for this label
            examples = x_sorted[y_sorted==label]
            examples_idx = 0
            if label in training_labels:
                x_train[train_start_idx:train_start_idx + trains_per_class] \
                    = examples[examples_idx:examples_idx+trains_per_class]
                y_train[train_start_idx:train_start_idx + trains_per_class] = np.repeat(label, trains_per_class)
                train_start_idx += trains_per_class
                if not trains_in_test:
                    examples_idx += trains_per_class
            if label in testing_labels:
                x_test[test_start_idx:test_start_idx + tests_per_class] = examples[examples_idx:examples_idx+tests_per_class]
                y_test[test_start_idx:test_start_idx + tests_per_class] = np.repeat(label, tests_per_class)
                test_start_idx += tests_per_class

    if shuffle_data:
        print('Shuffling data...')
        x_train, y_train = shuffle(x_train, y_train, random_state=0)
        x_test, y_test = shuffle(x_test, y_test, random_state=0)

    return x_train, y_train, x_test, y_test

def read_openml_data(data_dir: str = '..' + os.sep + 'datasets',
                     dataset :str = 'mnist_784',
                     precision_required: int = None, edge_features: bool = False):
    """
    Read a dataset from openml.org and cache it for future calls to this function.
    :param data_dir: directory to cache read data in. The function will check if the data exists in this directory
        prior to attempting to fetch from the remote site.
    :param dataset: name of the dataset to read. For example, `mnist_784` or `fashion-MNIST`.
    :param precision_required: the precision required for the data (must be lower than the original precision).
    :return: a tuple `(examples, labels)`
    """
    if not Path(data_dir).is_dir():
        raise ValueError('Error: The root directory specified for the data does not exist: {} '
              'Check that this is correctly defined or create it.'
              ' (Current working dir is {})'.format(data_dir, os.getcwd()))

    full_data_dir = data_dir + os.sep + dataset
    x_file = os.path.join(full_data_dir, 'x.npy')
    y_file = os.path.join(full_data_dir, 'y.npy')
    print('Looking for previously acquired {} dataset in the folder {}'
          ' (Current working dir is {})'.format(dataset, full_data_dir, os.getcwd()))

    if os.path.isfile(x_file) and os.path.isfile(y_file):
        print('Reading MNIST data from {} and {}  (Current working dir is {})'
              .format(x_file, y_file, os.getcwd()))
        x = np.load(x_file, allow_pickle=True)
        y = np.load(y_file, allow_pickle=True)
    else:
        print('Fetching {} data from openml.org site'.format(dataset))
        data = fetch_openml(dataset, parser='auto', as_frame=False)
        x = data.data
        y = data.target
        print('Caching data in {}'.format(full_data_dir), end='')
        if not Path(full_data_dir).is_dir():
            print(' Directory {} does not exist. Creating it.'.format(full_data_dir), end='')
            os.mkdir(full_data_dir)
        np.save(x_file, x)
        np.save(y_file, y)
        print('... data cached')

    if edge_features:
        print('Extracting edge features...')
        x = calc_prewitt_all_images(x)
        print('... features extracted')

    if precision_required is not None:
        print('Reducing data precision to {} ...'.format(precision_required), end='')
        x = set_data_precision(data=x, required_precision=precision_required)
        print('... precision reduced. ')

    return x, y

def examples_per_label(y: np.ndarray):
    """
    Return a list of labels for a dataset and the maximum number of examples per label to ensure a balanced
    dataset. This maximum will be the number of examples in for the label with the least data.
    :param y: a numpy array of all the labels
    :return: a tuple `(minimum number of examples per label, unique labels)`
    """
    unique_labels = np.unique(y)
    min_number_per_label = -1
    for label in unique_labels:
        label_count = y[y == label].shape[0]
        min_number_per_label = label_count if (label_count < min_number_per_label or min_number_per_label == -1) \
            else min_number_per_label
    return min_number_per_label, unique_labels

def set_data_precision(data: np.ndarray, required_precision: int) -> np.ndarray:
    """
    Reduces the precision of data in a numpy array. Zero values are not affected.
    original_precision = 5, [0,1,2,3,4,5], required_precision = 1 gives output [0,0,0,1,1,1]
    :param data: numpy array of data
    :param required_precision: precision required where precision is the number of possible non-zero values.
    :return data: numpy array of data with reduced precision
    """
    original_precision = int(np.max(data))

    if required_precision is None or required_precision == original_precision:
        return data
    if required_precision > original_precision:
        error_str = ('Required precision {} is greater than original precision {}. Unable to generate data.'
                     .format(required_precision, original_precision))
        raise(ValueError(error_str))

    increment = original_precision / required_precision
    data_out = data.copy()

    for new_precision in range(1, required_precision + 1):
        start_val = (new_precision-1) * increment
        end_val = start_val + increment
        data_out[(data_out > start_val) & (data_out <= end_val)] = new_precision

    return data_out

def print_data_sparsity_per_label(x: np.ndarray, y: np.ndarray) -> int:
    unique_labels = np.unique(y)
    sum_average = 0
    for label in unique_labels:
        x_label = x[y == label]
        average = int(np.sum(x_label > 0) / x_label.shape[0])
        print('Label: {} Average number of non-zero values: {}'.format(label, average))
        sum_average += average
    average_all = int(sum_average / len(unique_labels))
    print('Average overall: {}'.format(average_all))
    return average_all

def calc_prewitt(image: np.ndarray):
    prewitt = ndimage.prewitt(image)
    # Convert negatives to positives
    prewitt = np.sqrt(prewitt ** 2)
    return prewitt

def calc_prewitt_all_images(images:np.ndarray):
    prewitt_images = np.empty(images.shape)
    for image_num in range(images.shape[0]):
        prewitt = calc_prewitt(images[image_num])
        prewitt_images[image_num] = prewitt
    return prewitt_images


