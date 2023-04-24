import os
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import numpy as np
import pandas as pd

def spike_extractor(session, stim_indices, stim_table, start_win, end_win, probe_units):
    stim_mean_spike_vals = spike_average_calculator(start_win, end_win, stim_indices, stim_table, probe_units, session)
    return(stim_mean_spike_vals)

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

def spike_average_calculator(start_win, end_win, indices, stim_table, units, session):
    mean_spike_values   = np.zeros((units.shape[0], start_win.shape[0]))
    for win_idx in range(len(start_win)):
        startValues = stim_table.iloc[indices].start_time.values
        for unit in range(len(units)):
            unitSpikes = np.zeros(startValues.shape[0])
            for start_idx in range(len(startValues)):
                unitSpikes[start_idx] = np.where((session.spike_times[units[unit]] > startValues[start_idx] + start_win[win_idx]) & 
                                    (session.spike_times[units[unit]] < startValues[start_idx] + end_win[win_idx]))[0].shape[0]
            mean_spike_values[unit, win_idx] = np.mean(unitSpikes)
    return mean_spike_values

data_directory = 'raw_data/'
manifest_path = os.path.join(data_directory, "manifest.json")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

sessions = cache.get_session_table()

for sessNum in range(4):
    print(sessNum)
    thisSessionNum = sessions.index.values[sessNum]
    thisSession    = cache.get_session_data(thisSessionNum)

    ## Creating Relative Paths
    sessionPath = '/home/amirali/Desktop/Thesis/Codes/unit_data/Data/' + str(thisSessionNum)
    if not os.path.exists(sessionPath):
        os.makedirs(sessionPath)
    sessionDataPath = '/home/amirali/Desktop/Thesis/Codes/unit_data/Data/' + str(thisSessionNum) + '/'
    if not os.path.exists(sessionDataPath + 'tree_data'):
        os.makedirs(sessionDataPath + 'tree_data')
    if not os.path.exists(sessionDataPath + 'feline_data'):
        os.makedirs(sessionDataPath + 'feline_data')
    felineDataPath = sessionDataPath + 'feline_data/'
    treeDataPath   = sessionDataPath + 'tree_data/'

    ## Extracting Probes

    probes           = ['VISp', 'VISl', 'VISrl', 'VISal', 'VISpm', 'VISam', 'LGd', 'LP']
    session_probes   = list(thisSession.units.ecephys_structure_acronym.unique())
    available_probes = list(set(probes) & set(session_probes))

    ## Extracting Stimuli

    wholeDataFrame   = pd.read_csv('labels.csv')
    treeDataFrame    = wholeDataFrame[wholeDataFrame.Tree == 1].name
    felineDataFrame  = wholeDataFrame[wholeDataFrame.Feline == 1].name

    stimTable        = thisSession.get_stimulus_table()
    treePicIndices   = indice_pic_calculator(treeDataFrame)
    felinePicIndices = indice_pic_calculator(felineDataFrame)

    for probe in available_probes:
        probeUnits       = thisSession.units[(thisSession.units.ecephys_structure_acronym == probe)].index.values
        startWin         = np.arange(-0.2, 0.75, 0.01, dtype=float)
        endWin           = np.arange(-0.15, 0.80, 0.01, dtype=float)
        stimSpikesInTime = np.zeros((len(treePicIndices), len(probeUnits), len(startWin)))
        for idx_tree in range(len(treePicIndices)):
            treeStimIndices  = indice_stim_calculator(stimTable, treePicIndices[idx_tree])
            stimSpikesInTime[idx_tree, :, :] = spike_extractor(session=thisSession, stim_indices=treeStimIndices, stim_table=stimTable,
                                                                start_win=startWin, end_win=endWin, probe_units=probeUnits)
        with open(treeDataPath + 'treeMeanSpikeValues_' + probe + '.npy', 'wb') as f:
                np.save(f, stimSpikesInTime)
        stimSpikesInTime = np.zeros((len(felinePicIndices), len(probeUnits), len(startWin)))
        for idx_feline in range(len(felinePicIndices)):
            felineStimIndices  = indice_stim_calculator(stimTable, felinePicIndices[idx_feline])
            stimSpikesInTime[idx_feline, :, :] = spike_extractor(session=thisSession, stim_indices=felineStimIndices,  stim_table=stimTable,
                                                                start_win=startWin, end_win=endWin, probe_units=probeUnits)
        with open(felineDataPath + 'felineMeanSpikeValues_' + probe + '.npy', 'wb') as f:
                np.save(f, stimSpikesInTime)