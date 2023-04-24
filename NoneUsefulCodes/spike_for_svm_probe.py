import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def indice_pic_calculator(dataframe):
    indice_list = []
    for i in range(dataframe.shape[0]):
        file_name = dataframe.iloc[i].split('.')
        indice_list.append(float(file_name[0].split('_')[-1]))
    return indice_list

def indice_stim_calculator(stim_table, idx):
    stim_images   = stim_table[stim_table.stimulus_name == 'natural_scenes']
    object_indice = stim_images[stim_images.frame == idx].index.values
    return object_indice

def spike_average_calculator(start_win, end_win, indices, stim_table, units):
    mean_spike_values   = np.zeros((len(indices), start_win.shape[0]))
    for win_idx in range(len(start_win)):
        for idx in range(len(indices)):
            startValues = stim_table.iloc[indices[idx]].start_time.values
            avg_spikes = np.zeros(startValues.shape[0])
            for start_idx in range(len(startValues)):
                unitSpikes = np.zeros(units.shape[0])
                for unit in range(len(units)):
                    unitSpikes[unit] = np.where((sampleSession.spike_times[units[unit]] > startValues[start_idx] + start_win[win_idx]) & 
                                        (sampleSession.spike_times[units[unit]] < startValues[start_idx] + end_win[win_idx]))[0].shape[0]
                avg_spikes[start_idx] = np.sum(unitSpikes)
            mean_spike_values[idx, win_idx] = np.mean(avg_spikes)
    return mean_spike_values

data_directory = 'raw_data/'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()
sampleSession = cache.get_session_data(sessions.index.values[0])

stimIters        = 50

## Extracting Probes

probes           = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
session_probes   = list(sampleSession.units.ecephys_structure_acronym.unique())
available_probes = list(set(probes) & set(session_probes))

## Extracting Stimuli

wholeDataFrame   = pd.read_csv('labels.csv')
treeDataFrame    = wholeDataFrame[wholeDataFrame.Tree == 1].name
felineDataFrame  = wholeDataFrame[wholeDataFrame.Feline == 1].name

stimTable        = sampleSession.get_stimulus_table()
treePicIndices   = indice_pic_calculator(treeDataFrame)
felinePicIndices = indice_pic_calculator(felineDataFrame)

treeStimIndices   = np.zeros((len(treePicIndices), stimIters))
felineStimIndices = np.zeros((len(felinePicIndices), stimIters))

for idx_tree in range(len(treePicIndices)):
    treeStimIndices[idx_tree, :] = indice_stim_calculator(stimTable, treePicIndices[idx_tree])

for idx_feline in range(len(felinePicIndices)):
    felineStimIndices[idx_feline, :] = indice_stim_calculator(stimTable, felinePicIndices[idx_feline])
    
# Extracting Spikes

probe       = probes[7]
probeUnits  = sampleSession.units[(sampleSession.units.ecephys_structure_acronym == probe)].index.values
startWindow = np.arange(-0.2, 0.75, 0.01, dtype=float)
endWindow   = np.arange(-0.15, 0.80, 0.01, dtype=float)

treeMeanSpikeValues   = spike_average_calculator(startWindow, endWindow, treeStimIndices, stimTable, probeUnits)
felineMeanSpikeValues = spike_average_calculator(startWindow, endWindow, felineStimIndices, stimTable, probeUnits)

with open('treeMeanSpikeValues_LP.npy', 'wb') as f:
    np.save(f, treeMeanSpikeValues)

with open('felineMeanSpikeValues_LP.npy', 'wb') as f:
    np.save(f, felineMeanSpikeValues)