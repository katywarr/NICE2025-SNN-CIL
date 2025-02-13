from typing import Tuple

import numpy as np
import pandas as pd

from utils.one_hot_encoding import one_hot_encode_signal_repeat
from subnetwork.neuron_utils import get_spiking


def initialise_weights(source_population_size: int, sparsity: float,
                       target_population_size: int, w_init: float,
                       proportion: bool) -> np.ndarray:
    """

    """
    if proportion:
        # Each of the hidden neurons has exactly the same number of connections. Note that this rounding to int step
        # means that the number of connections could be slightly less or more than the average.
        weights = np.zeros(shape=(target_population_size, source_population_size), dtype=float)
        connections_per_target = int(sparsity * source_population_size)
        weights[:, 0:connections_per_target] = w_init
        rng = np.random.default_rng()
        for one_weights in weights:
            rng.shuffle(one_weights)
    else:
        # Treat the connection sparsity as a PROBABILITY.
        # Therefore, the number of connections will vary for different hidden neurons.
        weights = np.random.rand(target_population_size, source_population_size)
        weights[weights > sparsity] = 0
        weights[weights > 0] = w_init

    return weights

class ClassifierSubnet:

    """
    Maintains the hidden neuron weights and permanence values for the subnet
    """
    def __init__(self,
                 classification_label: str,
                 num_features: int,
                 data_precision: int,
                 learning_params: dict,
                 verbose: bool = True):

        # The classification label is only required for info/debug messages and for reading in the data
        self.classification_label = classification_label
        self.f = num_features
        self.precision = data_precision
        self.learning_params = learning_params
        self.verbose = verbose
        # ------------------------------------------------------------------------------------------------------------
        # The parameters that defined the trained subnetwork. These can be passed or read from a file.
        # ------------------------------------------------------------------------------------------------------------
        # Pre-synaptic weights and hidden neuron thresholds
        self.h_matured_weights = np.empty(shape=(0, self.f), dtype=float)
        self.h_matured_thresholds = np.empty(shape=(0,), dtype=float)
        # Output weights
        self.h_matured_weights_out = np.empty(shape=(0,), dtype=float)

        # ------------------------------------------------------------------------------------------------------------
        # Establish all the learning parameters.
        # ------------------------------------------------------------------------------------------------------------
        # An optional parameter to enable testing of the case where the learning pool is not replenished.
        # Usually not used so defaults to True. Add this so it is not mandatory to specify.
        if 'replenish_learning_pool' not in learning_params:
            learning_params['replenish_learning_pool'] = 1
        try:
            self.learning_pool_size = int(learning_params['learning_pool_size'])
            self.f_h_sparsity = learning_params['f_h_sparsity']
            self.h_threshold_mean = learning_params['h_threshold_mean']
            self.h_threshold_sd = learning_params['h_threshold_sd']
            self.pre_synap_weights_init = learning_params['h_weights_init']
            self.learn_weight_p = learning_params['h_weights_p']
            self.learn_weight_d = learning_params['h_weights_d']
            self.novelty_threshold = learning_params['novelty_threshold']
            self.noise_tolerance_ratio = learning_params['noise_tolerance_ratio']
            self.p_init = learning_params['p_init']
            self.p_potentiate = learning_params['p_potentiate']
            self.p_mature_threshold = learning_params['p_mature_threshold']
            self.p_deprecate = learning_params['p_deprecate']
            self.replenish_learning_pool = learning_params['replenish_learning_pool']
        except KeyError as e:
            print('Error: The following key was missing from the passed learning parameters:', e)
            raise e

        # The total potential to be divided across the firing weights
        self.learn_weight_p_total = self.learn_weight_p * self.h_threshold_mean
        # Initialise the learning weights matrix
        self.equal_connections_per_h = True
        self.h_learning_weights = initialise_weights(source_population_size=self.f,
                                                     target_population_size=self.learning_pool_size,
                                                     sparsity=self.f_h_sparsity,
                                                     w_init=self.pre_synap_weights_init,
                                                     proportion=self.equal_connections_per_h)
        self.h_learning_thresholds, self.h_learning_w_pots = self.get_thresholds(number=self.learning_pool_size)
        # Initialise the permanence for each hidden neuron
        self.h_learning_permanence = np.full(shape=(self.learning_pool_size,), fill_value=self.p_init, dtype=float)

        print('\n\nInitialised subnetwork classification {} for learning. Thresholds range from min {} to maximum {}\n'
              .format(classification_label,
                      np.min(self.h_learning_thresholds) if len(self.h_learning_thresholds) > 1 else 0,
                      np.max(self.h_learning_thresholds) if len(self.h_learning_thresholds) > 1 else 0))
        # ------------------------------------------------------------------------------------------------------------
        # The learning metrics for debug and analysis.
        # ------------------------------------------------------------------------------------------------------------
        self.total_refreshed = 0
        # How many training samples have been used during learning so far?
        self.total_training_samples = 0
        # How many of the training samples contributed to the learning. Some will be ignored if they are not
        # sufficiently novel.
        self.contributing_training_samples = 0
        # Also sum up the active training features in the training data.
        # A subnet's hyperparameter optimisation will depend on the average training features in a training sample,
        # so this is useful for calculating this average later.
        self.active_training_features = 0

        self.thresholds_firing = np.empty(shape=(0,), dtype=float)
        self.matured_data = {'training samples': np.array([0], dtype=int),
                             'number matured': np.array([0], dtype=int)}
        self.refreshed_data = {'training samples': np.array([0], dtype=int),
                               'number refreshed': np.array([0], dtype=int)}
        self.novelty_threshold_exceeded = 0
        # ------------------------------------------------------------------------------------------------------------
        # Prevent the hidden neurons from being expired when they go below the permanance threshold and from being
        # replenished when they are moved to the mature state.
        # This is to simulate a no-neurogenesis network - i.e. what would happen if the neurons were simply selected
        # from the learning pool but never replaced in the pool (refer to the paper for details on this).
        # ------------------------------------------------------------------------------------------------------------
        if not self.replenish_learning_pool:
            print('** Warning: The learning pool is not being replenished.')
        # The following flag is only used when the neurons in the learning pool are not being replenished.
        # When the learning pool contains only redundant neurons, it is essentially empty so no learning will occur.
        self.h_learning_redundant = np.full(shape=(self.learning_pool_size,), fill_value=False, dtype=bool)
        # Flag to indicate the learning pool is empty and a message has been printed to indicate this.
        self.learning_pool_empty = False

    # -----------------------------------------------------------------------------------------------------------------
    # Learn
    # -----------------------------------------------------------------------------------------------------------------
    def learn(self, signal: np.ndarray) -> int:
        self.total_training_samples += 1
        if self.learning_pool_size == 0: return 0

        # Case when the learning pool has run out of neurons (only if self.replenish_learning_pool == False)
        if self.learning_pool_empty:
            return self.h_matured_thresholds.shape[0]
        # Print a message the first time round
        if np.count_nonzero(self.h_learning_redundant) == self.learning_pool_size:
            print('All the learning neurons have been used up. This network will no longer learn.')
            self.learning_pool_empty = True
            # The return value should be exactly the learning pool size.
            return self.h_matured_thresholds.shape[0]

        if self.verbose:
            print('Learning Class: {} Training example: {}'
              .format( self.classification_label, self.total_training_samples))

        # If at least one mature neuron exists, perform a prediction
        if self.h_matured_thresholds.shape[0] >= 1:
            inference_score, total_spiking = self.predict(signal, learning = True)
            if inference_score > self.novelty_threshold:
                if self.verbose:
                    print('    Inference score: {}, total spiking: {} Novelty threshold exceeded. Not learning.'
                          .format(inference_score, total_spiking))
                self.novelty_threshold_exceeded += 1
                return self.h_matured_thresholds.shape[0]
            else:
                if self.verbose:
                    print('    Inference score: {}, total spiking: {}'
                          .format(inference_score, total_spiking))

        self.contributing_training_samples += 1

        # Get the signal density for the purposes of logging. Just take the number of the highest precision
        # self.active_training_features += np.count_nonzero(signal)
        unique, counts = np.unique(signal, return_counts=True)
        self.active_training_features += counts[-1]
        one_hot_signal_repeat = one_hot_encode_signal_repeat(signal,
                                                             repeats=self.learning_pool_size,
                                                             phase_length=self.precision)
        non_perm_weights = self.h_learning_weights.copy()
        spiked = np.full(shape=(self.learning_pool_size,), fill_value=False)

        # Learn using the learning pool
        for time_index in range(self.precision - 1, -1, -1):
            spiking_this_time = get_spiking(f_h_weights=non_perm_weights,
                                            one_hot_signal_repeat=one_hot_signal_repeat,
                                            time_index=time_index,
                                            thresholds=self.h_learning_thresholds)
            spiking_this_time[spiked] = False
            spiking_indices_this_time = np.where(spiking_this_time)[0]

            for hidden in spiking_indices_this_time:
                signal_time_slice = one_hot_signal_repeat[0, :, time_index]
                self.update_weights_for_hidden(h_index=hidden, signal_time_slice=signal_time_slice)
            spiked[spiking_this_time] = True

        spiking_indices = np.where(spiked)[0]
        non_spiking_indices = np.where(~spiked)[0]

        self.update_permanence(spiking=spiking_indices, non_spiking=non_spiking_indices)

        # Refresh the hidden neurons that have expired
        expired_hidden_neurons = np.where(self.h_learning_permanence == 0)[0]
        if self.verbose:
            print('    Train complete: Spiking: {} Matured so far: {} Expiring: {}'
                  .format(len(spiking_indices), self.h_matured_thresholds.shape[0], len(expired_hidden_neurons)))

        for expired in expired_hidden_neurons:
            self.refresh_immature_hidden_neuron(expired)

        return self.h_matured_thresholds.shape[0]

    # -----------------------------------------------------------------------------------------------------------------
    # Run single prediction
    # -----------------------------------------------------------------------------------------------------------------
    def predict(self, signal: np.ndarray, learning: bool = False) -> (float, int):
        total_permanent = self.h_matured_weights.shape[0]
        if total_permanent == 0: return 0, 0

        one_hot_signal_repeat = one_hot_encode_signal_repeat(signal,
                                                             repeats=self.h_matured_weights.shape[0],
                                                             phase_length=self.precision)

        h_spiked = np.full(shape=(total_permanent,), fill_value=False)
        for time_index in range(self.precision-1, -1, -1):
            spiking_this_time = get_spiking(f_h_weights=self.h_matured_weights,
                                            one_hot_signal_repeat=one_hot_signal_repeat,
                                            time_index=time_index,
                                            thresholds=self.h_matured_thresholds)
            h_spiked[spiking_this_time] = True    # Set the newly spiking neurons to True
                                                  # They may already be True if they have previously spiked

        total_h_spiking = np.count_nonzero(h_spiked)
        result = np.sum(self.h_matured_weights_out[h_spiked])

        if not learning and total_h_spiking > 0:
            this_time = self.h_matured_thresholds[h_spiked]
            self.thresholds_firing = np.append(self.thresholds_firing, this_time)       # For diagnostics only

        if not learning:
            if self.verbose:
                print('    Prediction for signal subnet classifier {}. Spiking: {}. Permanent: {}. Output: {}'
                      .format(self.classification_label, total_h_spiking, total_permanent, round(result, 4)))

        return result, total_h_spiking

    # -----------------------------------------------------------------------------------------------------------------
    # Methods for updating the weights and permanence of the hidden neurons
    # -----------------------------------------------------------------------------------------------------------------
    def update_weights_for_hidden(self, h_index: int, signal_time_slice: np.ndarray):
        """
        Updates the weights for a hidden neuron that spikes during learning. The weights that contributed to the
        spike are potentiated. Those that did not contribute to the spike are deprecated.
        Weights of zero indicate that the pre-synaptic connection no longer exists, so these are not subject to
        potentiation or deprecation.
        :param h_index:
            Index of the hidden neuron
        :param signal_time_slice:
            Time slice of the signal that caused the hidden neuron to fire. This is a Boolean of length f where a
            True indicates that the feature was active for that time slice.
        """
        h_weights = self.h_learning_weights[h_index]
        conns_active = h_weights[np.logical_and(signal_time_slice, h_weights != 0)]
        num_conns_active = len(conns_active)

        if num_conns_active == 0: # Should never occur as threshold must be greater than 1. See get_thresholds method.
            raise RuntimeError('Error: No connections to hidden neuron id {}. Threshold {}  Weights \n{}'
                  .format(h_index, self.h_learning_thresholds[h_index], self.h_learning_weights[h_index]))

        h_weights[np.logical_and(signal_time_slice, h_weights != 0)] += (
                self.h_learning_w_pots[h_index]/num_conns_active) if num_conns_active > 0 else 0

        h_weights[np.logical_and(~signal_time_slice, h_weights != 0)] -= self.learn_weight_d
        # Clip any that have got too high or too low
        np.clip(h_weights, a_min=0, a_max=None, out=self.h_learning_weights[h_index])

    def make_permanent(self, h_index: int) -> bool:
        """
        Make a hidden neuron that has reached the permanence threshold permanent.
        :param h_index:
        :return:
        """
        # Prune weights that are too low
        self.h_learning_weights[h_index][self.h_learning_weights[h_index] <= self.pre_synap_weights_init] = 0
        pre_syn_weight_total = np.sum(self.h_learning_weights[h_index])

        # Create a scaling factor to normalise the weights across the neurons
        # Calculate the target maximum pre-synaptic weight.
        target_weight_total = self.h_threshold_mean * self.noise_tolerance_ratio
        scaled_weight = self.h_learning_weights[h_index] * target_weight_total / pre_syn_weight_total

        # Add the hidden neuron to the mature set
        self.h_matured_weights = np.vstack([self.h_matured_weights, scaled_weight])
        self.h_matured_thresholds = np.append(self.h_matured_thresholds, self.h_learning_thresholds[h_index])

        # Weight the output according to the neuron's selectivity
        weight = self.h_learning_thresholds[h_index]/self.h_threshold_mean
        self.h_matured_weights_out = np.append(self.h_matured_weights_out, weight)

        # And remove the hidden neuron from the learning set
        self.refresh_immature_hidden_neuron(h_index)

        total_matured = self.h_matured_thresholds.shape[0]

        self.matured_data['training samples'] = np.append(self.matured_data['training samples'], self.total_training_samples)
        self.matured_data['number matured'] = np.append(self.matured_data['number matured'], total_matured)

        return True

    def increase_permanence(self, spiking: np.ndarray):
        num_new_perm = 0
        for i in spiking:
            if not self.h_learning_redundant[i]:
                self.h_learning_permanence[i] += self.p_potentiate
                if self.h_learning_permanence[i] >= self.p_mature_threshold:
                    if self.make_permanent(i):
                        num_new_perm += 1

        if num_new_perm > 0 and self.verbose:
            print('+ Subnet for label {}. {} hidden neurons matured to permanent state'
                  .format(self.classification_label, num_new_perm))

    def decrease_permanence(self, non_spiking: np.ndarray):
        for i in non_spiking:
            if not self.h_learning_redundant[i]:
                self.h_learning_permanence[i] -= self.p_deprecate * (1-self.h_learning_permanence[i])
                if self.h_learning_permanence[i] < 0: self.h_learning_permanence[i] = 0

    def update_permanence(self, spiking: np.ndarray, non_spiking: np.ndarray):
        self.increase_permanence(spiking)
        self.decrease_permanence(non_spiking)

    def refresh_immature_hidden_neuron(self, index: int):
        if not self.replenish_learning_pool:
            self.nullify_immature_hidden_neuron(index)
        else:
            new_weights = initialise_weights(source_population_size=self.f,
                                             target_population_size=1,
                                             sparsity=self.f_h_sparsity,
                                             w_init=self.pre_synap_weights_init,
                                             proportion=self.equal_connections_per_h)[0]
            self.h_learning_weights[index] = new_weights

            threshold, w_pot  = self.get_thresholds(number=1)
            self.h_learning_thresholds[index] = threshold[0]
            self.h_learning_w_pots[index] = w_pot[0]

            self.h_learning_permanence[index] = self.p_init
            self.total_refreshed += 1
        self.refreshed_data['training samples'] = np.append(self.refreshed_data['training samples'], self.total_training_samples)
        self.refreshed_data['number refreshed'] = np.append(self.refreshed_data['number refreshed'], self.total_refreshed)

    def nullify_immature_hidden_neuron(self, index: int):
        # Called if the neuron slot won't be re-used
        #if self.verbose:
        #    print('Removing hidden neuron {} from the learning pool.'.format(index))
        self.h_learning_weights[index] = 0
        self.h_learning_thresholds[index] = -1
        self.h_learning_permanence[index] = -1 # To prevent continual replenishment
        self.h_learning_redundant[index] = True

    def get_training_diagnostics(self):
        df_matured = pd.DataFrame()
        df_matured['training samples'] = np.append(self.matured_data['training samples'], self.total_training_samples)
        df_matured['number matured'] = np.append(self.matured_data['number matured'], self.matured_data['number matured'][-1])
        df_matured['Subnetwork'] = np.repeat(self.classification_label, repeats=len(self.matured_data['training samples'])+1)

        df_refreshed = pd.DataFrame()
        df_refreshed['training samples'] = np.append(self.refreshed_data['training samples'], self.total_training_samples)
        df_refreshed['number refreshed'] = np.append(self.refreshed_data['number refreshed'], self.refreshed_data['number refreshed'][-1])
        df_refreshed['Subnetwork'] = np.repeat(self.classification_label, repeats=len(self.refreshed_data['training samples'])+1)

        return df_matured, df_refreshed

    def get_thresholds(self, number: int) -> Tuple:
        """
        Establish thresholds for new hidden neurons.
        :param number: (int)
            Number of hidden neurons to consider.
        """
        thresholds = np.random.normal(self.h_threshold_mean, self.h_threshold_sd, size=number)
        thresholds[thresholds < 1] = 1          # Prevents the case where there are no connections active.
        # Potentiation is threshold*w_p / num connections.
        # Calculating threshold*w_p and caching it in an array is for efficiency.
        w_potentiations = thresholds * self.learn_weight_p
        return thresholds, w_potentiations
