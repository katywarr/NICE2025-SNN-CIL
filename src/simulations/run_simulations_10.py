from simulation_runner import SimulationRunner

if __name__ == '__main__':

    data_dir = '../../datasets/split_mnist_784_5000_1000'
    results_dir_root = '../../simulation_results'
    sims_per_test = 5

    learning_params = {'learning_pool_size': 250,
                       'f_h_sparsity': 0.2,
                       'h_threshold_mean': 24,
                       'h_threshold_sd': 3.5,
                       'h_weights_init': 1.0,
                       'h_weights_p': 0.4,
                       'h_weights_d': 0.4,
                       'novelty_threshold': 60,
                       # This will be mult by the threshold to get the pre_synaptic weight total.
                       # ratio * threshold indicates the maximum voltage of a hidden neuron.
                       'noise_tolerance_ratio': 1.3,
                       'p_init': 0.5,
                       'p_potentiate': 0.2,
                       'p_deprecate': 0.02,
                       'p_mature_threshold': 1.0,
                       'replenish_learning_pool': 1
                       }

    learning_pool_size_range = [25, 100, 250, 500]
    novelty_threshold_range = [10]

    runner = SimulationRunner(data_directory=data_dir, results_directory_root=results_dir_root, verbose=False)
    for learning_pool_size in learning_pool_size_range:
        for novelty_threshold in novelty_threshold_range:
            learning_params['learning_pool_size'] =  learning_pool_size
            learning_params['novelty_threshold'] =  novelty_threshold
            r_padded = f"{learning_params['novelty_threshold']:03d}"
            h_padded = f"{learning_params['learning_pool_size']:04d}"
            test_id = 'r_'+ r_padded + '_h_'+ h_padded
            runner.run_simulations_for_test(test_id=test_id,
                                            learning_params=learning_params,
                                            num_simulations=sims_per_test,
                                            generate_plots=False)
