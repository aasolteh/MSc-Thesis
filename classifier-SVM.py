import numpy as np
from sklearn.model_selection import LeavePOut
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from numpy_da import DynamicArray
from sklearn.decomposition import PCA

def apply_pca(in_data):
    n_components = 15
    out_data = np.zeros((in_data.shape[0], n_components-1, in_data.shape[2]))
    pca = PCA(n_components=n_components)
    for i in range(in_data.shape[0]):
        out_data[i, :, :] = pca.fit_transform(in_data[i, :, :].T)[:, :-1].T
    return out_data

def svm_and_plot_iters(high_data, low_data, probe, n_iter):
    high_labels  = np.zeros(high_data.shape[0]) 
    low_labels   = np.ones(low_data.shape[0])
    total_data    = np.concatenate((high_data, low_data), axis=0)
    total_labels  = np.concatenate((high_labels, low_labels), axis=0)

    # iters    = n_iter
    lpo      = LeavePOut(p)
    iters    = lpo.get_n_splits(total_data)
    accuracy = np.zeros((total_data.shape[2], iters))
    
    for train_index, test_index in lpo.split(total_data):
        X_train, X_test = total_data[train_index], total_data[test_index]
        y_train, y_test = total_labels[train_index], total_labels[test_index]
        for t in range(accuracy.shape[0]):
            clf                  = make_pipeline(SVC(kernel='rbf'))
            clf.fit(X_train[:, :, t], y_train)
            y_pred               = clf.predict(X_test[:, :, t])
            accuracy[t, n_split] = accuracy_score(y_test, y_pred)
        n_split = n_split + 1

    # for iter in range(iters):
    #     print(iter)
    #     X_train, X_test, y_train, y_test = train_test_split(total_data, total_labels, test_size=0.3, stratify=total_labels)
    #     for t in range(accuracy.shape[0]):
    #         clf                  = make_pipeline(SVC(kernel='rbf'))
    #         clf.fit(X_train[:, :, t], y_train)
    #         y_pred               = clf.predict(X_test[:, :, t])
    #         accuracy[t, iter] = accuracy_score(y_test, y_pred)
            
    # with open('/home/amirali/Desktop/Thesis/Codes/unit_data/Metrics/HiLoLuminance/SVM/' + probe + '_100iters_accuracy.npy', 'wb') as f:
    #     np.save(f, accuracy)
        
    acc      = accuracy
    accuracy = np.mean(acc, axis=1)
    error    = np.std(acc, axis=1) / np.sqrt(iters)
    time     = np.arange(-200, 750, 10)
    sns.set()
    plt.figure(figsize=(10, 5))
    plt.plot(time, accuracy, linewidth = 2)
    plt.fill_between(time, accuracy-error, accuracy+error, alpha=0.5)
    plt.xlabel('time (ms)')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.axvline(x = 0, ymin=0, ymax=1, color = 'r', label = 'stimuli onset', linestyle='dashed')
    plt.legend()
    plt.title('Accuracy over time for ' + probe)
    plt.savefig('/home/amirali/Desktop/Thesis/Codes/unit_data/Plots/HiLoLuminance/SVM/RBF/100-iters-' + probe + '_PCA.png')
    
probDict = {
    "VISp" : "PrimaryVisualCortex (V1)",
    "VISl" : "LateroMedial (LM)",
    "VISrl" : "RostroLateral (RL)",
    "VISal" : "AnteroLateral (AL)",
    "VISpm" : "PosteroMedial (PM)",
    "VISam" : "AnteroMedial (AM)",
    "LP" : "LateralPosteriorNuc (LP)",
    "LGd" : "LateralGeniculateNuc (LGn)",
}

if __name__ == "__main__":
    dataPath      = '/home/amirali/Desktop/Thesis/Codes/unit_data/Data/HiLoLuminance/'
    wholeSessions = os.listdir(dataPath)
    probeUnits    = []
    
    highV1DataArray  = DynamicArray(shape=(52,20,95))
    highLMDataArray  = DynamicArray(shape=(40,20,95))
    highRLDataArray  = DynamicArray(shape=(10,20,95))
    highALDataArray  = DynamicArray(shape=(9,20,95))
    highPMDataArray  = DynamicArray(shape=(18,20,95))
    highAMDataArray  = DynamicArray(shape=(37,20,95))
    highLPDataArray  = DynamicArray(shape=(28,20,95))
    highLGnDataArray = DynamicArray(shape=(71,20,95))

    lowV1DataArray  = DynamicArray(shape=(52,20,95))
    lowLMDataArray  = DynamicArray(shape=(40,20,95))
    lowRLDataArray  = DynamicArray(shape=(10,20,95))
    lowALDataArray  = DynamicArray(shape=(9,20,95))
    lowPMDataArray  = DynamicArray(shape=(18,20,95))
    lowAMDataArray  = DynamicArray(shape=(37,20,95))
    lowLPDataArray  = DynamicArray(shape=(28,20,95))
    lowLGnDataArray = DynamicArray(shape=(71,20,95))
    
    for session in wholeSessions:
        sessionPath = os.path.join(dataPath, session + '/high_data')
        sessionData = os.listdir(sessionPath)
        for np_data in sessionData:
            npPath      = os.path.join(sessionPath, np_data)
            data        = np.load(npPath)
            data        = np.reshape(data, (data.shape[1], data.shape[0], data.shape[2]))
            probeRegion = np_data.split("_")[-1].split(".")[0]
            if probeRegion == "VISp":
                highV1DataArray.append(data[:, :, :])
            elif probeRegion == "VISl":
                highLMDataArray.append(data[:, :, :])
            elif probeRegion == "VISrl":
                highRLDataArray.append(data[:, :, :])
            elif probeRegion == "VISal":
                highALDataArray.append(data[:, :, :])
            elif probeRegion == "VISpm":
                highPMDataArray.append(data[:, :, :])
            elif probeRegion == "VISam":
                highAMDataArray.append(data[:, :, :])
            elif probeRegion == "LP":
                highLPDataArray.append(data[:, :, :])
            elif probeRegion == "LGd":
                highLGnDataArray.append(data[:, :, :])
                
    for session in wholeSessions:
        sessionPath = os.path.join(dataPath, session + '/low_data')
        sessionData = os.listdir(sessionPath)
        for np_data in sessionData:
            npPath      = os.path.join(sessionPath, np_data)
            data        = np.load(npPath)
            data        = np.reshape(data, (data.shape[1], data.shape[0], data.shape[2]))
            probeRegion = np_data.split("_")[-1].split(".")[0]
            if probeRegion == "VISp":
                lowV1DataArray.append(data[:, :, :])
            elif probeRegion == "VISl":
                lowLMDataArray.append(data[:, :, :])
            elif probeRegion == "VISrl":
                lowRLDataArray.append(data[:, :, :])
            elif probeRegion == "VISal":
                lowALDataArray.append(data[:, :, :])
            elif probeRegion == "VISpm":
                lowPMDataArray.append(data[:, :, :])
            elif probeRegion == "VISam":
                lowAMDataArray.append(data[:, :, :])
            elif probeRegion == "LP":
                lowLPDataArray.append(data[:, :, :])
            elif probeRegion == "LGd":
                lowLGnDataArray.append(data[:, :, :])
         
         
    highData     = highAMDataArray
    lowData      = lowAMDataArray
    highSpike    = np.reshape(highData, (highData.shape[1], highData.shape[0], highData.shape[2]))
    lowSpike     = np.reshape(lowData, (lowData.shape[1], lowData.shape[0], lowData.shape[2]))
    highSpikePCA = apply_pca(highSpike)
    lowSpikePCA  = apply_pca(lowSpike)
    svm_and_plot_iters(high_data=highSpike, low_data=lowSpike, probe='AM', n_iter=100)
    
    highData   = highLGnDataArray
    lowData     = lowLGnDataArray
    highSpike  = np.reshape(highData, (highData.shape[1], highData.shape[0], highData.shape[2]))
    lowSpike    = np.reshape(lowData, (lowData.shape[1], lowData.shape[0], lowData.shape[2]))
    highSpikePCA = apply_pca(highSpike)
    lowSpikePCA  = apply_pca(lowSpike)
    svm_and_plot_iters(high_data=highSpike, low_data=lowSpike, probe='LGn', n_iter=100)
           
    highData   = highPMDataArray
    lowData     = lowPMDataArray
    highSpike  = np.reshape(highData, (highData.shape[1], highData.shape[0], highData.shape[2]))
    lowSpike    = np.reshape(lowData, (lowData.shape[1], lowData.shape[0], lowData.shape[2]))
    highSpikePCA = apply_pca(highSpike)
    lowSpikePCA  = apply_pca(lowSpike)
    svm_and_plot_iters(high_data=highSpike, low_data=lowSpike, probe='PM', n_iter=100)
    
    highData   = highLPDataArray
    lowData     = lowLPDataArray
    highSpike  = np.reshape(highData, (highData.shape[1], highData.shape[0], highData.shape[2]))
    lowSpike    = np.reshape(lowData, (lowData.shape[1], lowData.shape[0], lowData.shape[2]))
    highSpikePCA = apply_pca(highSpike)
    lowSpikePCA  = apply_pca(lowSpike)
    svm_and_plot_iters(high_data=highSpike, low_data=lowSpike, probe='LP', n_iter=100)
    
    highData   = highALDataArray
    lowData     = lowALDataArray
    highSpike  = np.reshape(highData, (highData.shape[1], highData.shape[0], highData.shape[2]))
    lowSpike    = np.reshape(lowData, (lowData.shape[1], lowData.shape[0], lowData.shape[2]))
    highSpikePCA = apply_pca(highSpike)
    lowSpikePCA  = apply_pca(lowSpike)
    svm_and_plot_iters(high_data=highSpike, low_data=lowSpike, probe='AL', n_iter=100)
    
    highData   = highRLDataArray
    lowData     = lowRLDataArray
    highSpike  = np.reshape(highData, (highData.shape[1], highData.shape[0], highData.shape[2]))
    lowSpike    = np.reshape(lowData, (lowData.shape[1], lowData.shape[0], lowData.shape[2]))
    highSpikePCA = apply_pca(highSpike)
    lowSpikePCA  = apply_pca(lowSpike)
    svm_and_plot_iters(high_data=highSpike, low_data=lowSpike, probe='RL', n_iter=100)
    
    highData   = highLMDataArray
    lowData     = lowLMDataArray
    highSpike  = np.reshape(highData, (highData.shape[1], highData.shape[0], highData.shape[2]))
    lowSpike    = np.reshape(lowData, (lowData.shape[1], lowData.shape[0], lowData.shape[2]))
    highSpikePCA = apply_pca(highSpike)
    lowSpikePCA  = apply_pca(lowSpike)
    svm_and_plot_iters(high_data=highSpike, low_data=lowSpike, probe='LM', n_iter=100)
    
    highData   = highV1DataArray
    lowData     = lowV1DataArray
    highSpike  = np.reshape(highData, (highData.shape[1], highData.shape[0], highData.shape[2]))
    lowSpike    = np.reshape(lowData, (lowData.shape[1], lowData.shape[0], lowData.shape[2]))
    highSpikePCA = apply_pca(highSpike)
    lowSpikePCA  = apply_pca(lowSpike)
    svm_and_plot_iters(high_data=highSpike, low_data=lowSpike, probe='V1', n_iter=100)