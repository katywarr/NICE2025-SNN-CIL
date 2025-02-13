import os
import glob
from typing import Tuple
import pickle
import numpy as np
import pprint
import pandas as pd
from pathlib import Path
from utils.excel_handler import ExcelHandler
from subnetwork.classifier_subnet import ClassifierSubnet

class Classifier:

    def __init__(self, learning_params: dict, from_directory: str = None, verbose: bool = True):
        """
        Instantiate a Classifier object from scratch.
        To instantiate a classifier from a dictionary of learning parameters, specify the learning parameters only.
        To instantiate a Classifier from a previously learnt network, use the Classifier.from_saved() method.
        :param learning_params: (dict)
            The learning parameters associated with the simulation.
        :param from_directory: (str)
            Read a pre-saved network from this directory. The learning parameters will be ignored.
        :param verbose: (bool)
            Set to True for verbose output.
        """
        self.verbose = verbose

        self.learning_params = learning_params
        self.subnetworks = np.empty(shape=(0,), dtype = ClassifierSubnet)
        self.learnt_labels = np.empty(shape=(0,), dtype = str)
        self.f = None
        self.data_precision = 0
        self.last_predict_results = None

        if from_directory is not None:
            self.__load_network(from_directory)

        pp = pprint.PrettyPrinter(indent=4)
        print('==============================================================')
        print('Classifier network defined as follows:')
        print('\nLearning params summary: ')
        pp.pprint(self.learning_params)
        print('==============================================================')


    @classmethod
    def from_saved(cls, from_directory: str, verbose: bool = True):
        """
        Instantiate a Classifier from a set of saved files at the given path.
        :param from_directory: (str)
            Read a pre-saved network from this directory. The learning parameters will be ignored.
        :param verbose: (bool)
            Set to True for verbose output.
        """
        classifier = Classifier(learning_params={}, from_directory=from_directory, verbose=verbose)
        return classifier, classifier.learnt_labels

    def learn(self, x_train: np.ndarray, y_train: np.ndarray):
        """
        Perform training.
        :param x_train: (np.ndarray)
            Numpy array containing the training data examples.
        :param y_train: (np.ndarray)
            Numpy array containing the training labels.
        :return: (np.ndarray)
            Numpy array containing the learnt labels. The length of this array corresponds to the number of subnetworks.
        """
        if len(self.subnetworks) == 0:          # First time. Initialise f.
            self.f = x_train.shape[1]
            self.data_precision = np.max(x_train)
        # -----------------------------------------------------------
        # Validate input
        # -----------------------------------------------------------
        if len(x_train.shape) != 2:
            raise ValueError('Unable to run simulation. x_train shape {} is invalid - should have 2 dims: '
                             .format(x_train.shape))
        if x_train.shape[1] != self.f:
            raise ValueError('Unable to run simulation as the length of the data samples do not match the expected '
                             'number of features {}. x_train samples are each length {}'
                             .format(self.f, x_train.shape[1]))
        if y_train.shape[0] != x_train.shape[0]:
            raise ValueError('Unable to run simulation as the number of training labels {} did '
                             'not match the number of training samples {}'
                             .format(y_train.shape[0], x_train.shape[0]))

        # -----------------------------------------------------------
        # Learn
        # -----------------------------------------------------------
        y_train = y_train.astype(int)
        x_train = x_train.astype(int)
        num_trains = x_train.shape[0]

        print('Learning:')
        print('      Length of each signal:      {}'.format(self.f))
        print('      Number of training signals: {}'.format(num_trains))
        for signal, classification, train_id in zip(x_train, y_train, range(num_trains)):
            #print('\n>> Learning signal {} classification {}.'.format(train_id, classification), end='')
            subnet_index = np.where(self.learnt_labels == str(classification))[0]
            if len(subnet_index) == 0:
                # Haven't seen this classification previously so it requires a new subnet
                subnet = ClassifierSubnet(num_features=self.f,
                                          data_precision=self.data_precision,
                                          classification_label=classification,
                                          learning_params=self.learning_params,
                                          verbose=self.verbose)
                self.subnetworks = np.append(self.subnetworks, subnet)
                self.learnt_labels = np.append(self.learnt_labels, str(classification))
            else:   # There will only be one.
                subnet = self.subnetworks[subnet_index[0]]
            if not self.verbose: print(' {}'.format(train_id), end='', flush=True) # Instead of verbose subnet output
            subnet.learn(signal=signal)
        print('\nLearning complete.\n')
        return self.learnt_labels


    def predict(self, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Perform tests.
        :param x_test: (np.ndarray)
            Numpy array containing the test data examples.
        :param y_test: (np.ndarray)
            Numpy array containing the test labels.
        :return: (Tuple)
            Three dataframes containing the test diagnostics: Full scores for each of the tests (one row per test),
            Average scores per each label (one row per label). Average correct and incorrect
            firing activity per label (one row per label).
        """
        if len(self.subnetworks) == 0:
            raise ValueError('Unable to run test. No subnetworks have been trained.')

        # -----------------------------------------------------------
        # Validate input
        # -----------------------------------------------------------
        if len(x_test.shape) != 2:
            raise ValueError('Unable to run simulation. x_test shape {} is invalid - should have 2 dims: '
                             .format(x_test.shape))
        if x_test.shape[1] != self.f:
            raise ValueError('Unable to run simulation as the length of the data samples do not match the expected '
                             'number of features {}. x_test samples are each length {}.'
                             .format(self.f, x_test.shape[1]))
        if y_test.shape[0] != x_test.shape[0]:
            raise ValueError('Unable to run simulation as the number of test labels {} did '
                             'not match the number of test samples {}'
                             .format(y_test.shape[0], x_test.shape[0]))

        # -----------------------------------------------------------
        # Perform inference
        # -----------------------------------------------------------
        predictions = np.full(shape=y_test.shape, fill_value='x', dtype=str)
        test_scores_per_subnet = np.full(shape=(y_test.shape[0], len(self.subnetworks)), fill_value=0, dtype=float)
        spiking_correct_per_test = np.zeros(shape=y_test.shape, dtype=int)
        spiking_incorrect_per_test = np.zeros(shape=y_test.shape, dtype=int)

        if not self.verbose:
            print('Simulation results: + = correct / - = incorrect / x = no result / * = subnetwork draw')
            print('Run prediction: ', end='')

        for signal, test_number in zip(x_test, range(y_test.shape[0])):

            best_score = 0
            draw = False
            for subnet, label, subnet_num in zip(self.subnetworks, self.learnt_labels, range(len(self.learnt_labels))):
                score, spikes = subnet.predict(signal=signal, learning=False)
                # Take as a proportion of total permanent
                #score = round(score/np.count_nonzero(subnet.h_permanent),2)
                test_scores_per_subnet[test_number][subnet_num] = score
                if y_test[subnet_num] == label:
                    spiking_correct_per_test[test_number] += spikes
                else:
                    spiking_incorrect_per_test[test_number] += spikes

                if score > best_score:
                    best_score = score
                    # Set the prediction to the label associated with this subnet
                    predictions[test_number] = label
                    draw = False
                else:
                    if predictions[test_number] != 'x' and score == best_score:
                        # It's a draw. Just take the prediction we had (it may be wrong)
                        if self.verbose:
                            print('\n****** Prediction draw between labels {} ({}) and {} ({})'
                              .format(predictions[test_number], best_score, label, score))
                        draw = True
            if self.verbose:
                print('\n>> Signal {} classification {} prediction {}.'
                      .format(test_number, y_test[test_number], predictions[test_number]))
            else:
                print(' {}'.format(test_number), end='')
                print('+', end='') if predictions[test_number] == y_test[test_number] else print('-', end='')
                if draw: print('*', end='')
                if predictions[test_number] == 'x': print('x', end='')

        # -----------------------------------------------------------
        # Collate diagnostic information
        # -----------------------------------------------------------
        # It is possible for there to be more test labels than learnt labels if the network has been restricted to
        # only learn a sub-set of labels. This is useful for testing and optimising individual subnetworks.
        all_results = pd.DataFrame(test_scores_per_subnet)
        # Rename the columns so they identify the classification label rather than the subnet number
        for subnet_num, label in zip(range(len(self.subnetworks)), self.learnt_labels):
            all_results = all_results.rename(columns={subnet_num: label})

        all_results['actual'] = y_test
        all_results['predictions'] = predictions
        all_results['spiking_correct_per_test'] = spiking_correct_per_test
        all_results['spiking_incorrect_per_test'] = spiking_incorrect_per_test
        all_results['spiking_per_test'] = np.add(spiking_correct_per_test, spiking_incorrect_per_test)

        self.last_predict_results = all_results

        print('===>all results\n{}'.format(all_results))

        return all_results

    def get_last_predict_results(self):
        return self.last_predict_results

    def get_predict_diagnostics(self, all_results: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Get diagnostic averages for each of the subnetworks for a set of tests.
        :param all_results: (pd.DataFrame)
            DataFrame containing the full results for all the tests. This is returned from the predict method.
        :return: (Tuple[pd.DataFrame, pd.DataFrame])
            Two dataframes. One containing the averages scores per test per label and the expected label result. The
            second contains the average firing activity (correct and incorrect) per test per subnetwork.
        """
        test_labels = np.unique(all_results['actual'])
        average_scores_per_label = np.full(shape=(len(test_labels), len(self.subnetworks)), fill_value=0, dtype=float)
        # Establish the average score for each subnet for each ground truth label.
        print('\nEstablishing averages')
        for actual, actual_num in zip(test_labels, range(len(test_labels))):
            # Select the rows for the where the actual result is the label
            test_scores_for_actual = all_results.loc[all_results['actual'] == actual]

            for label, label_num in zip(test_labels, range(len(test_labels))):
                # Establish the average score for this label for this actual classification
                scores_per_label_np = test_scores_for_actual.loc[:, label].values
                average_scores_per_label[actual_num][label_num] = round(np.average(scores_per_label_np), 2) \
                    if len(average_scores_per_label) > 0 else 0

        average_scores = pd.DataFrame(average_scores_per_label)
        print('... done establishing averages')
        # Rename the columns so they identify the classification label rather than the subnet number
        for subnet_num, label in zip(range(len(test_labels)), self.learnt_labels):
            average_scores = average_scores.rename(columns={subnet_num: label})
        average_scores['actual'] = test_labels

        """
        # For each subnet create a single row containing the average hidden firing correctly per test and the
        # average firing incorrectly per test.
        print('Establishing firing activity')
        firing_correct_per_subnet = np.full(shape=(len(self.subnetworks),), fill_value=0, dtype=float)
        firing_incorrect_per_subnet = np.full(shape=(len(self.subnetworks),), fill_value=0, dtype=float)

        for subnet_num, label in zip(range(len(self.subnetworks)), self.learnt_labels):
            correct_label_index = np.where(test_labels == label)[0]
            incorrect_label_indices = np.where(test_labels != label)[0]
            firing_correct_per_subnet[subnet_num] = average_scores_per_label[correct_label_index, subnet_num]
            firing_incorrect_per_subnet[subnet_num] = (
                round(np.average(average_scores_per_label[incorrect_label_indices, subnet_num]), 2)) \
                if len(average_scores_per_label[incorrect_label_indices, subnet_num]) > 0 else 0

        firing_summary = pd.DataFrame(self.learnt_labels)
        firing_summary['correct'] = firing_correct_per_subnet
        firing_summary['incorrect'] = firing_incorrect_per_subnet
        print('... done establishing firing activity')
        """
        return average_scores, int(np.average(all_results['spiking_per_test']))

    def subnet_data(self) -> pd.DataFrame:
        """
        Return a dataframe containing diagnostic data for each of the trained subnetworks. This includes information
        such as the number of neurons that matured during training and the number of training samples that
        contributed to learning.
        :return: (pd.DataFrame)
            Dataframe describing the trained subnetworks.
        """

        num_subnets = len(self.subnetworks)
        sn_classifications = np.empty(shape=(num_subnets,), dtype=str)
        sn_permanent = np.empty(shape=(num_subnets,), dtype=int)
        sn_refreshed = np.empty(shape=(num_subnets,), dtype=int)
        sn_insufficient_conns = np.empty(shape=(num_subnets,), dtype=int)

        sn_training_samples = np.empty(shape=(num_subnets,), dtype=int)
        sn_training_samples_contrib= np.empty(shape=(num_subnets,), dtype=int)
        sn_training_samples_not_novel= np.empty(shape=(num_subnets,), dtype=int)

        sn_average_active_training = np.empty(shape=(num_subnets,), dtype=float)

        sn_f_h_weight_av = np.empty(shape=(num_subnets,), dtype=float)
        sn_f_h_weight_std = np.empty(shape=(num_subnets,), dtype=float)
        sn_f_h_weight_max = np.empty(shape=(num_subnets,), dtype=float)
        sn_f_h_weight_min = np.empty(shape=(num_subnets,), dtype=float)

        sn_h_out_weight_av = np.empty(shape=(num_subnets,), dtype=float)
        sn_h_out_weight_max = np.empty(shape=(num_subnets,), dtype=float)
        sn_h_out_weight_min = np.empty(shape=(num_subnets,), dtype=float)

        for subnet, index in zip(self.subnetworks, range(len(self.subnetworks))):
            sn_classifications[index] = str(subnet.classification_label)
            sn_permanent[index] = subnet.h_matured_thresholds.shape[0]
            sn_refreshed[index] = subnet.total_refreshed

            sn_training_samples[index] = subnet.total_training_samples
            sn_training_samples_contrib[index] = subnet.contributing_training_samples
            sn_training_samples_not_novel[index] = subnet.novelty_threshold_exceeded

            sn_average_active_training[index] = round(subnet.active_training_features / sn_training_samples[index], 2)
            mature_weights = subnet.h_matured_weights.copy()
            mature_weights = mature_weights.flatten()
            # Any zeros correspond to no connection. Ignore.
            mature_weights = mature_weights[mature_weights != 0]

            sn_f_h_weight_av[index] = np.average(mature_weights) if len(mature_weights) > 0 else 0
            sn_f_h_weight_std[index] = np.std(mature_weights) if len(mature_weights) > 0 else 0
            sn_f_h_weight_max[index] = np.max(mature_weights) if len(mature_weights) > 0 else 0
            sn_f_h_weight_min[index] = np.min(mature_weights) if len(mature_weights) > 0 else 0

            mature_out_weights = subnet.h_matured_weights_out.copy()
            sn_h_out_weight_av[index] = np.average(mature_out_weights) if len(mature_out_weights) > 0 else 0
            sn_h_out_weight_max[index] = np.max(mature_out_weights) if len(mature_out_weights) > 0 else 0
            sn_h_out_weight_min[index] = np.min(mature_out_weights) if len(mature_out_weights) > 0 else 0

        df = pd.DataFrame({'classification': sn_classifications,
                           'h_permanent': sn_permanent,
                           'h_refreshed': sn_refreshed,
                           'h_insufficient_conns': sn_insufficient_conns,
                           'training samples': sn_training_samples,
                           'training samples contrib': sn_training_samples_contrib,
                           'training samples not novel': sn_training_samples_not_novel,
                           # The following proportion provides an indication of the spread of the training data.
                           'training samples contrib prob': np.divide(sn_training_samples_contrib, sn_training_samples),
                           'training sparsity': sn_average_active_training,
                           'f_h_weight_average': sn_f_h_weight_av,
                           'f_h_weight_std': sn_f_h_weight_std,
                           'f_h_weight_max': sn_f_h_weight_max,
                           'f_h_weight_min': sn_f_h_weight_min,
                           'h_out_weight_average': sn_h_out_weight_av,
                           'h_out_weight_max': sn_h_out_weight_max,
                           'h_out_weight_min': sn_h_out_weight_min,
                           })
        return df

    def h_mature_neuron_data(self) -> (np.ndarray, np.ndarray):
        num_subnets = len(self.subnetworks)
        # The mature neuron count varies depending on the subnet.
        # Establish the maximum. This will inform the size of the matrices returned from this function
        max_matured = 0
        for subnet in self.subnetworks:
            matured = subnet.h_matured_weights.shape[0]
            if matured > max_matured:
                max_matured = matured

        sn_f_h_mature_weights = np.full(shape=(num_subnets, max_matured, self.f), fill_value=0, dtype=float)
        sn_h_mature_thresholds = np.full(shape=(num_subnets, max_matured), fill_value=0, dtype=float)
        sn_h_learning_thresholds = np.full(shape=(num_subnets, int(self.learning_params['learning_pool_size'])),
                                           fill_value=0, dtype=float)
        sn_h_mature_weights_out = np.full(shape=(num_subnets, max_matured), fill_value=0, dtype=float)


        for subnet, index in zip(self.subnetworks, range(len(self.subnetworks))):
            mature_weights = subnet.h_matured_weights.copy()
            sn_f_h_mature_weights[index, 0:subnet.h_matured_weights.shape[0]] = mature_weights
            mature_thresholds = subnet.h_matured_thresholds.copy()
            sn_h_mature_thresholds[index, 0:subnet.h_matured_thresholds.shape[0]] = mature_thresholds
            learning_thresholds = subnet.h_learning_thresholds.copy()
            sn_h_learning_thresholds[index, 0:subnet.h_learning_thresholds.shape[0]] = learning_thresholds
            mature_weights_out = subnet.h_matured_weights_out.copy()
            sn_h_mature_weights_out[index, 0:subnet.h_matured_weights_out.shape[0]] = mature_weights_out
        dict_out = {
            'mature_weights': sn_f_h_mature_weights,
            'mature_thresholds': sn_h_mature_thresholds,
            'learning_thresholds': sn_h_learning_thresholds,
            'mature_weights_out': sn_h_mature_weights_out
        }
        return dict_out

    def get_thresholds_firing(self, index):
        return self.subnetworks[index].thresholds_firing

    def get_training_diagnostics_all_subnets(self):
        matured_dfs = None
        refreshed_dfs = None
        for subnet, index in zip(self.subnetworks, range(len(self.subnetworks))):
            matured_df, refreshed_df= subnet.get_training_diagnostics()
            if matured_dfs is None:
                matured_dfs = matured_df
            else:
                matured_dfs = pd.concat([matured_dfs, matured_df])
            if refreshed_dfs is None:
                refreshed_dfs = refreshed_df
            else:
                refreshed_dfs = pd.concat([refreshed_dfs, refreshed_df])
        return matured_dfs, refreshed_dfs


    def save_network(self, root_directory: str, network_name: str) -> str:
        """
        Save the trained network to the specified directory. The weights and thresholds of all the subnetworks will
        be saved to the directory along with an Excel file describing the network's hyperparameters. A classifier based
        on the saved network can then be created from these files using the from_saved function.

        :param root_directory: (str)
            The root directory in which the network will be saved.
        :param network_name: (str)
            The name of the network to be saved. A subdirectory under the root will be created.
        :return: (str)
            The full path to the directory containing the saved network files.
        """
        if not Path(root_directory).is_dir():
            raise ValueError('Unable to save network as the directory {} does not exist. (Current directory is {})'
                             .format(root_directory, os.getcwd()))
        full_dir = root_directory + os.sep + network_name
        if Path(full_dir).is_dir():
            raise ValueError('Unable to save network as the directory {} exists. '
                             'Not saving to prevent overwriting. (Current directory is {})'
                             .format(full_dir, os.getcwd()))
        os.mkdir(full_dir)
        network_description_file = ExcelHandler(full_dir, 'network_description')
        network_description_file.add_row_from_dicts(
            [{'f': int(self.f), 'precision': int(self.data_precision)}, self.learning_params],
            sheet_name='network_description')

        if self.last_predict_results is not None:
            prediction_results_filename = os.path.join(full_dir, 'prediction_results.pkl')
            with open(prediction_results_filename, 'wb') as f:
                # noinspection PyTypeChecker
                pickle.dump(self.last_predict_results, f)

        for subnet, subnet_num in zip(self.subnetworks, range(self.subnetworks.shape[0])):
            filename = os.path.join(full_dir, 'subnet_'+str(subnet_num)+'.pkl')
            with open(filename, 'wb') as f:
                # noinspection PyTypeChecker
                pickle.dump(subnet, f)

        print('Network description saved to directory {}'.format(full_dir))
        return full_dir

    def __load_network(self, saved_network_dir: str):
        """
        Load a trained network from a previously stored description.
        This is a private method used to initialise the network using the from_saved initialisation function.

        :param saved_network_dir:
            Location of the network description files to be loaded.
        """

        print('Reading network from {}'.format(saved_network_dir))

        if not Path(saved_network_dir).is_dir():
            print('Error: the given path {} does not exist. The current working directory is {}.'.format(
                saved_network_dir, os.getcwd()))
            return

        # Check that the directory contained the expected files
        network_description_file = os.path.join(saved_network_dir, 'network_description.xlsx')
        if not Path.is_file(Path(network_description_file)):
            raise ValueError('The network description file is not located at {}'.format(network_description_file))

        subnets_files = glob.glob(saved_network_dir + os.sep + 'subnet_*.pkl')
        num_subnetworks = len(subnets_files)
        if num_subnetworks == 0:
            raise ValueError('Error: the given path {} does not contain any subnetwork data. '
                  'Specifically: a file subnet*.pkl could not be located in the directory'.format(saved_network_dir))

        # Read in the network description
        network_desc_handler = ExcelHandler(saved_network_dir, 'network_description')
        description_df = network_desc_handler.read_sheet('network_description')
        self.learning_params = description_df.loc[0].to_dict()
        self.f = int(self.learning_params['f'])
        self.data_precision = int(self.learning_params['precision'])
        # Strictly speaking these are not in the learning parameter dictionary
        del self.learning_params['f']
        del self.learning_params['precision']

        # Read in the last prediction results, if they exist.
        prediction_results_filename = os.path.join(saved_network_dir, 'prediction_results.pkl')
        if Path(prediction_results_filename).is_file():
            with open(prediction_results_filename, 'rb') as f:
                self.last_predict_results = pickle.load(f)

        # Load the individual subnetworks
        for subnet_num in range(num_subnetworks):
            filename = os.path.join(saved_network_dir, 'subnet_'+str(subnet_num)+'.pkl')
            with open(filename, 'rb') as f:
                # noinspection PyTypeChecker
                subnet = pickle.load(f)
            self.subnetworks = np.append(self.subnetworks, subnet)
        self.learnt_labels = np.arange(num_subnetworks).astype(dtype=str)
