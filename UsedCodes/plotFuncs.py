import os
import numpy as np
import matplotlib.pyplot as plt

def PSTHPlotter(stim_list, stop_times, sess_units, session):
    sess_dir = os.getcwd()
    for unit in sess_units:
        os.chdir(sess_dir)
        if not os.path.exists(unit):
            os.mkdir(unit)
            os.chdir(unit)
        unit_dir = os.getcwd()
        for stim in range(len(stim_list)):
            os.chdir(unit_dir)
            if not os.path.exists(stim_list[stim]):
                os.mkdir(stim_list[stim])
                os.chdir(stim_list[stim])
            presentations = session.get_stimulus_table(stim_list[stim])
            units = session.units[session.units["ecephys_structure_acronym"] == unit]

            time_step = 0.01
            time_bins = np.arange(-0.1, stop_times[stim] + time_step, time_step)

            histograms = session.presentationwise_spike_counts(
                stimulus_presentation_ids=presentations.index.values,  
                bin_edges=time_bins,
                unit_ids=units.index.values
            )
            
            mean_histograms = histograms.mean(dim="stimulus_presentation_id")

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pcolormesh(
                mean_histograms["time_relative_to_stimulus_onset"], 
                np.arange(mean_histograms["unit_id"].size),
                mean_histograms.T, 
                vmin=0,
                vmax=1
            )

            ax.set_ylabel("unit", fontsize=11)
            ax.set_xlabel("time relative to stimulus onset (s)", fontsize=11)
            ax.set_title("peristimulus time histograms for " + unit + " units on " + stim_list[stim] + " presentations", fontsize=11)

            curr_dir = os.getcwd()
            save_dir = curr_dir
            plt.savefig(save_dir + '/PSTH.png')
            
def SPEEDPlotter(session):
    sess_dir         = os.getcwd()
    speed_df         = session.running_speed
    speed_df['time'] = speed_df[['start_time', 'end_time']].mean(axis = 1)
    end_time_idx     = np.where(speed_df.time > 9 + speed_df.time.iloc[0])[0][0]
    
    fig, ax = plt.subplots(figsize = (10, 3))
    plt.plot(speed_df.time[0:end_time_idx] - speed_df.time.iloc[0], speed_df.velocity[0:end_time_idx], 
            color = 'gray', linewidth = 2)
    plt.xlabel('time (s)')
    plt.ylabel('speed (cm s-1)')
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    plt.title('Running Speed for First 9 Second of Task')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(False)
    plt.savefig(sess_dir + '/' + 'SPEED.png')
    
def RASTERPlotter(session):
    sess_dir = os.getcwd()
    unit_ids = session.units.index.to_numpy()
    V1     = 'VISp'
    LM     = 'VISl'
    RL     = 'VISrl'
    AL     = 'VISal'
    PM     = 'VISpm'
    AM     = 'VISam'
    LGN    = 'LGd'
    LP     = 'LP'
    desired_probes = [V1, LM, RL, AL, PM, AM, LGN, LP]
    session_probes = list(session.units.ecephys_structure_acronym.unique())
    available_probes = list(set(desired_probes) & set(session_probes))
    for probe in available_probes:
        os.chdir(sess_dir)
        if not os.path.exists(probe):
            os.mkdir(probe)
            os.chdir(probe)
        available_probe_idx = np.where(session.units.ecephys_structure_acronym == probe)[0]
        unit_ids_per_probe  = unit_ids[available_probe_idx]
        ax, fig = plt.subplots(figsize = (10, 5)) 
        counter_id  = 1
        for id in unit_ids_per_probe:
            unit_spikes = session.spike_times[id]
            t           = counter_id * np.ones(unit_spikes.shape[0])
            counter_id  = counter_id + 1
            plt.plot(unit_spikes, t, '.', color = 'black')
        plt.xlim((2000, 2200))
        plt.ylabel('unit')
        plt.xlabel('time (s)')
        plt.title('Raster Plot for ' + probe + ' probe')
        curr_dir = os.getcwd()
        save_dir = curr_dir
        plt.savefig(save_dir + '/RASTER') 