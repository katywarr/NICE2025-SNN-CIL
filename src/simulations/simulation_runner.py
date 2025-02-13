import pprint
import numpy as np
import random
import os
import seaborn as sns
import time
from classifier import Classifier
from utils.excel_handler import ExcelHandler
from utils.file_handler import generate_simulation_results_folders
from utils.mnist_reader import load_train_test_dataset
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd


class SimulationRunner:

    def __init__(self, data_directory: str, results_directory_root: str, verbose:bool = False) -> None:
        """
        :param data_directory: (str)
            Directory containing the training and test data that this SimulationRunner will use. The data remains the
            same for every simulation.
        :param results_directory_root: (str)
            Directory where the results will be stored. This includes details of the tests, plots, and the learnt
            networks.
        """

        # Load the pre-prepared data
        self.x_train, self.y_train, self.x_test, self.y_test, self.data_params = (
            load_train_test_dataset(from_directory=data_directory))
        self.data_precision = self.data_params['precision_required']
        self.f = self.x_train.shape[1]
        self.verbose = verbose

        # Prepare directories and files for the simulation results. These are all under the
        (self.sim_dir, self.sim_plots_dir, self.sim_networks_dir, self.sim_results_file) = (
            generate_simulation_results_folders(simulation_folder_root=results_directory_root))
        self.results_file_handler = ExcelHandler(self.sim_dir, self.sim_results_file)


        self.simulation_number = 0
        self.test_number = 0

        print('Data summary:'
              '\n    Training labels:                  {}'
              '\n    Testing labels:                   {}'
              '\n    Trains per classification:        {}'
              '\n    Tests per classification:         {}'
              '\n    Total training:                   {}'
              '\n    Total test:                       {}'
              '\n    Trains in test set?:              {}'
              .format(self.data_params['training_labels'],
                      self.data_params['testing_labels'],
                      self.data_params['trains_per_class'],
                      self.data_params['tests_per_class'],
                      self.y_train.shape[0],
                      self.y_test.shape[0],
                      self.data_params['trains_in_test_set']))

        self.pp = pprint.PrettyPrinter(indent=4)

    def run_simulations_for_test(self, test_id: str,
                                 learning_params: dict, num_simulations: int, generate_plots: bool = False):
        """
        Run the several simulations for the same network hyperparameters. Record the results in a spreadsheet and
        optionally generate and store plots describing the simulations.

        :param learning_params: (dict)
            Dictionary of the learning parameters describing the test.
        :param num_simulations: (int)
            Number of simulations to run.
        :param generate_plots: (bool)
            Set to True to generate and save plots.
        """
        accuracies = np.zeros(shape=(num_simulations,), dtype=float)
        spikes_per_prediction = np.zeros(shape=(num_simulations,), dtype=int)

        sn_av_perm = np.zeros(shape=(num_simulations,), dtype=float)
        sn_av_refresh = np.zeros(shape=(num_simulations,), dtype=float)
        sn_av_train = np.zeros(shape=(num_simulations,), dtype=float)
        sn_av_train_contributing = np.zeros(shape=(num_simulations,), dtype=float)

        test_summary_sheet='test_summaries'
        self.test_number += 1

        for simulation in range(num_simulations):
            sim_results = self.run_simulation(test_id=test_id,
                                              learning_params=learning_params, generate_plots=generate_plots)

            # Cache the results for this simulation along with for the test summary
            accuracies[simulation] = sim_results['accuracy']
            spikes_per_prediction[simulation] = sim_results['average_spikes_per_prediction']
            # The av_sn_* params are np arrays with one value for each subnet. Take the subnet averages for the sim.
            sn_av_perm[simulation] = sim_results['sn_av_permanent']
            sn_av_refresh[simulation] = sim_results['sn_av_refreshed']
            sn_av_train[simulation] = sim_results['sn_av_train']
            sn_av_train_contributing[simulation] = sim_results['sn_av_train_contributing']

        # Write a summary sheet for the test, giving the averages over all the simulations.
        average_accuracy = round(np.average(accuracies), 4)
        accuracy_variation = round(max(np.max(accuracies)-average_accuracy, average_accuracy-np.min(accuracies)),4)
        test_results = {'average_accuracy': average_accuracy,
                        'accuracy_variation': accuracy_variation,
                        'average_spikes_per_test': round(np.average(spikes_per_prediction), 2),
                        'sn_av_perm': round(np.average(sn_av_perm),2),
                        'sn_av_refresh': round(np.average(sn_av_refresh), 2),
                        'sn_av_train': round(np.average(sn_av_train), 2),
                        }
        # Add the test to the spreadsheet.
        self.results_file_handler.add_row_from_dicts(
            [{'test_ref': test_id, 'test': self.test_number, 'num simulations': num_simulations},
             self.data_params, learning_params, test_results
             ],
            sheet_name=test_summary_sheet)

        print('Test {} ({}) results: {}'.format(self.test_number, test_id, test_results))

    def run_simulation(self, test_id: str,
                       learning_params: dict, generate_plots: bool = False) -> dict:
        random.seed(5)
        simulation_summary_sheet = 'simulation_summaries'

        self.simulation_number += 1
        network_name = 'simulation_'+str(self.simulation_number)

        x_train = self.x_train.copy()
        y_train = self.y_train.copy()
        x_test = self.x_test.copy()
        y_test = self.y_test.copy()

        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        #                                           *** TRAINING ***
        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        classifier = Classifier(learning_params, verbose=self.verbose)
        time1=time.time()
        labels_learnt = classifier.learn(x_train, y_train)
        time2=time.time()
        learning_time = round(time2-time1, 2)

        # -------------------------------------------------------------------------------------------------------------
        # Save the networks
        # -------------------------------------------------------------------------------------------------------------
        classifier.save_network(root_directory=self.sim_networks_dir, network_name=network_name)

        # -------------------------------------------------------------------------------------------------------------
        # Plot data about the learnt networks
        # -------------------------------------------------------------------------------------------------------------
        colours = sns.color_palette("husl", len(labels_learnt))
        if generate_plots:

            print('Generating plots...(this may take a while), to prevent plot generation in future runs,'
                  'pass generate_plots=False to the simulation_runner.')

            # ---------------------------------------------------------------------------------------------------------
            # Learning: Number matured per subnet plot
            # ---------------------------------------------------------------------------------------------------------
            matured, refreshed = classifier.get_training_diagnostics_all_subnets()
            sns.lineplot(data=matured, hue='Subnetwork', x='training samples', y='number matured', legend="full",
                         palette=colours, style='Subnetwork')
            plt.savefig(self.sim_plots_dir + os.path.sep + str(self.simulation_number) + '-maturing_rates.png',
                        dpi=1000)
            plt.close()

            # ---------------------------------------------------------------------------------------------------------
            # Learning: Number refreshed per subnet plot
            # ---------------------------------------------------------------------------------------------------------
            sns.lineplot(data=refreshed, hue='Subnetwork', x='training samples', y='number refreshed', legend="full",
                         palette=colours, style='Subnetwork')
            plt.savefig(self.sim_plots_dir + os.path.sep + str(self.simulation_number) + '-refreshing_rates.png',
                        dpi=1000)
            plt.close()

            # ---------------------------------------------------------------------------------------------------------
            # Learning: Threshold density plot per subnetwork
            # ---------------------------------------------------------------------------------------------------------
            neuron_data = classifier.h_mature_neuron_data()

            all_mature_thresholds = neuron_data['mature_thresholds']

            classification = np.empty(shape=(0,), dtype=int)
            thresholds = np.empty(shape=(0,), dtype=float)
            for subnet_num in np.argsort(labels_learnt):
                subnet_thresholds = all_mature_thresholds[subnet_num][all_mature_thresholds[subnet_num] > 0]
                thresholds = np.append(thresholds, subnet_thresholds)
                classification = np.append(classification, np.repeat(int(labels_learnt[subnet_num]),
                                                                     len(subnet_thresholds)))
            df = pd.DataFrame(data={'classification': classification, 'thresholds': thresholds})
            scatter = sns.kdeplot(data=df, x="thresholds", hue="classification",
                                  palette=sns.color_palette(colours, as_cmap=True))
            full_file_name = self.sim_plots_dir + os.path.sep + str(
                self.simulation_number) + '-all_mature_thresholds.png'
            fig1 = scatter.get_figure()
            fig1.savefig(full_file_name, dpi=1000)
            plt.close()
        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        #                                           *** PREDICT ***
        # -------------------------------------------------------------------------------------------------------------
        # -------------------------------------------------------------------------------------------------------------
        time3=time.time()
        all_results = classifier.predict(x_test, y_test)
        time4 = time.time()
        inference_time = round(time4-time3, 2)
        predictions = all_results['predictions']

        # -------------------------------------------------------------------------------------------------------------
        # Save the data pertaining to this simulation in Excel
        # -------------------------------------------------------------------------------------------------------------
        results = classification_report(y_true=y_test, y_pred=predictions, zero_division=0, output_dict=True)
        subnet_summary = classifier.subnet_data()
        self.results_file_handler.add_rows(df=subnet_summary, sheet_name='subnet_' + str(self.simulation_number))
        results_summary = classification_report(y_true=y_test, y_pred=predictions, zero_division=0, output_dict=True)
        # Accuracy needs to be removed to store without error as it does not equate to a dict.
        del results_summary['accuracy']
        for key in results_summary.keys():
            self.results_file_handler.add_row_from_dicts(
                [{'item': key}, dict(results_summary[key])],
                sheet_name='summary_' + str(self.simulation_number))

        # Allocate a sheet for the complete scores and firing summaries
        average_scores, average_spikes = classifier.get_predict_diagnostics(all_results)
        self.results_file_handler.add_rows(df=all_results, sheet_name='all_results_' + str(self.simulation_number))
        self.results_file_handler.add_rows(df=average_scores, sheet_name='av_results_' + str(self.simulation_number))

        # -------------------------------------------------------------------------------------------------------------
        # Save detailed simulation results in a separate sheet (one per simulation)
        # -------------------------------------------------------------------------------------------------------------
        simulation_results = {'test_ref': test_id,
                              'accuracy': results['accuracy'],
                              'average_spikes_per_prediction': average_spikes,
                              'sn_av_permanent': np.average(subnet_summary['h_permanent']),
                              'sn_av_refreshed': np.average(subnet_summary['h_refreshed']),
                              'sn_av_train': np.average(subnet_summary['training samples']),
                              'sn_av_train_contributing': np.average(subnet_summary['training samples contrib']),
                              'learning_time': learning_time,
                              'inference_time': inference_time
                              }

        self.results_file_handler.add_row_from_dicts(
            [{'test': self.test_number, 'simulation': self.simulation_number},
             self.data_params, learning_params, simulation_results
             ],
            sheet_name=simulation_summary_sheet)

        print('Predictions for simulation {} - Accuracy: {} Permanent/subnet: {} Average spikes per prediction: {}'
              .format(str(self.simulation_number), simulation_results['accuracy'],
                      simulation_results['sn_av_permanent'],
                      simulation_results['average_spikes_per_prediction']))

        # -------------------------------------------------------------------------------------------------------------
        # Always save the confusion matrix
        # -------------------------------------------------------------------------------------------------------------
        ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=predictions,
                                                labels=np.append(np.sort(labels_learnt), 'x'),
                                                normalize='true', values_format = '.2f')
        plt.savefig(self.sim_plots_dir + os.path.sep + str(self.simulation_number) + '-confusion.png', dpi=1000)
        plt.close()

        if generate_plots:
            # ---------------------------------------------------------------------------------------------------------
            # Prediction: Thresholds firing per subnet plot
            # ---------------------------------------------------------------------------------------------------------
            classification = np.empty(shape=(0,), dtype = str)
            thresholds_firing = np.empty(shape=(0,), dtype = float)

            for subnet_num in np.argsort(labels_learnt):
                firing = classifier.get_thresholds_firing(subnet_num)
                thresholds_firing = np.append(thresholds_firing, firing)
                classification = np.append(classification, np.repeat(labels_learnt[subnet_num], len(firing)))

            df2 = pd.DataFrame(data={'Thresholds firing': thresholds_firing, 'Subnetwork (class)': classification})
            hist = sns.histplot(data=df2, x="Thresholds firing", y="Subnetwork (class)",  hue='Subnetwork (class)',
                                palette=sns.color_palette(colours, as_cmap = True),
                                legend=False)
            full_file_name = self.sim_plots_dir + os.path.sep + str(self.simulation_number) + '-all_firing.png'
            fig1 = hist.get_figure()
            fig1.savefig(full_file_name, dpi=1000)
            plt.close()

        return simulation_results
