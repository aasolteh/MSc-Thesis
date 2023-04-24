import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns

def svm(first_data, second_data, n_iters, kernel, C):
    first_labels  = np.zeros(first_data.shape[0])
    second_labels = np.ones(second_data.shape[0])
    total_data    = np.concatenate((first_data, second_data), axis=0)
    total_labels  = np.concatenate((first_labels, second_labels))    
    accuracy      = np.zeros((n_iters, total_data.shape[2]))

    for iter in range(accuracy.shape[0]):
        if iter % 20 == 0:
            print("Epoch Number: ", iter)
        X_train, X_test, y_train, y_test = train_test_split(total_data, total_labels, test_size=0.3, stratify=total_labels)
        for t in range(accuracy.shape[1]):
            clf                  = make_pipeline(SVC(kernel=kernel, C=C))
            clf.fit(X_train[:, :, t], y_train)
            y_pred               = clf.predict(X_test[:, :, t])
            accuracy[iter, t]    = accuracy_score(y_test, y_pred)
    
    return accuracy

def probe_data_extractor(probe_name, image_feature, probe_dict, data_path):
    image_number = 0
    for img in image_feature:
        probe_dict[probe][1] = 0
        image_path = os.path.join(data_path, str(img))
        for session in os.listdir(image_path):
            session_path = os.path.join(image_path, session)
            probe_path   = os.path.join(session_path, probe)
            if os.path.exists(probe_path):
                for data in os.listdir(probe_path):
                    data_dir = os.path.join(probe_path, data)
                    if probe_dict[probe][1] == 0:
                        spike_data            = np.load(data_dir)
                        probe_dict[probe][1] += 1
                    else:
                        session_data = np.load(data_dir)
                        spike_data   = np.concatenate((spike_data, session_data), axis=0)
        
        if image_number == 0:
            probe_data  = np.zeros((len(image_feature), spike_data.shape[0], spike_data.shape[1]))
        probe_data[image_number, :, :] = spike_data
        image_number += 1
    return probe_data

def plot_accuracy(accuracy, probe, feature):
    acc      = accuracy
    iters    = accuracy.shape[0]
    accuracy = np.mean(acc, axis=0)
    error    = np.std(acc, axis=0) / np.sqrt(iters)
    time     = np.arange(-200, 780, 10)
    
    sns.set()
    plt.figure(figsize=(10, 5))
    plt.plot(time, accuracy, linewidth = 2)
    plt.fill_between(time, accuracy - 3 * error, accuracy + 3 * error, alpha=0.5)
    plt.xlabel('time (ms)')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.axvline(x = 0, ymin=0, ymax=1, color = 'r', label = 'stimuli onset', linestyle='dashed')
    plt.legend()
    plt.title('Accuracy of ' + feature +  ' over time for ' + probe)

probeDict = {
    "VISp"  : ["PrimaryVisualCortex (V1)", 0],
    "VISl"  : ["LateroMedial (LM)", 0],
    "VISrl" : ["RostroLateral (RL)", 0],
    "VISal" : ["AnteroLateral (AL)", 0],
    "VISpm" : ["PosteroMedial (PM)", 0],
    "VISam" : ["AnteroMedial (AM)", 0],
    "LP"    : ["LateralPosteriorNuc (LP)", 0],
    "LGd"   : ["LateralGeniculateNuc (LGn)", 0]
}
allProbes = list(probeDict.keys())

featuresDF       = pd.read_csv('/home/amirali/Desktop/Thesis/Codes/LuminanceContrast.csv')
LLuminanceImages = featuresDF.LLuminance.values.astype('int')
HLuminanceImages = featuresDF.HLuminance.values.astype('int')
LContrastImages  = featuresDF.LContrast.values.astype('int')
HContrastImages  = featuresDF.HContrast.values.astype('int')

dataPath = '/home/amirali/Desktop/Thesis/Codes/unit_data/Data/'

for probe in allProbes:
    print("Probe Name: ", probe)
    LLuminanceProbeData = probe_data_extractor(probe_name=probe, image_feature=LLuminanceImages, probe_dict=probeDict, data_path=dataPath)
    HLuminanceProbeData = probe_data_extractor(probe_name=probe, image_feature=HLuminanceImages, probe_dict=probeDict, data_path=dataPath)
    LContrastProbeData  = probe_data_extractor(probe_name=probe, image_feature=LContrastImages, probe_dict=probeDict, data_path=dataPath)
    HContrastProbeData  = probe_data_extractor(probe_name=probe, image_feature=HContrastImages, probe_dict=probeDict, data_path=dataPath)
    luminanceAccuracy   = svm(first_data=LLuminanceProbeData, second_data=HLuminanceProbeData, n_iters=100)
    contrastAccuracy    = svm(first_data=LContrastProbeData, second_data=HContrastProbeData, n_iters=100)
    plot_accuracy(luminanceAccuracy, probeDict[probe][0], "Luminance")
    plot_accuracy(contrastAccuracy, probeDict[probe][0], "Contrast")