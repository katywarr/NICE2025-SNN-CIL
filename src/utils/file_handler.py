import os
from pathlib import Path
import pandas as pd

def get_timestamp_str() -> str:
    time_stamp = pd.Timestamp.now()
    return time_stamp.strftime('%Y-%m-%d-%H-%M')

def generate_simulation_results_folders(simulation_folder_root: str):

    # Check that the top level folder exists. Don't generate this (to be sure that the data is stored in the
    # correct place)
    if not os.path.exists(simulation_folder_root):
        raise ValueError('The simulation results directory {} does not exist. '
                         'Check that this is the correct path and create the appropriate folder if required.'
                         'The current working directory is {}.'
                         .format(simulation_folder_root, os.getcwd()))

    # Create a folder dedicated to this simulation run.
    sim_identifier = get_timestamp_str()
    simulation_sub_folder = os.path.join(simulation_folder_root, sim_identifier)
    tag = 1
    while Path(simulation_sub_folder).is_dir():
        # This happens if the tests are run too quickly. Add a unique tag.
        simulation_sub_folder = os.path.join(simulation_folder_root, sim_identifier+'_'+str(tag))
        tag += 1
    os.mkdir(simulation_sub_folder)

    sim_plots_dir = os.path.join(simulation_sub_folder, 'plots')
    sim_networks_dir = os.path.join(simulation_sub_folder, 'networks')
    os.mkdir(sim_plots_dir)
    os.mkdir(sim_networks_dir)
    results_file = 'results_' + sim_identifier  # This is the root for the xlsx file that will be created later
    print('The simulation results folder {} has been created.'
          '    Results will be stored in the file:    {}.xlsx'
          '    Plots will be stored in the directory: {}'
          '    The network will be stored in the directory: {}'
          .format(simulation_folder_root, results_file, sim_plots_dir, sim_networks_dir))

    return simulation_sub_folder, sim_plots_dir, sim_networks_dir, results_file
