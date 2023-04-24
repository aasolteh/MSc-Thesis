import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeavePOut
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import interpolate

def svm_and_plot(feline_data, tree_data, probe):
    feline_labels = np.zeros(feline_data.shape[0]) 
    tree_labels   = np.ones(tree_data.shape[0])
    total_data    = np.concatenate((feline_data, tree_data), axis=0)
    total_labels  = np.concatenate((feline_labels, tree_labels), axis=0)

    lpo = LeavePOut(2)
    accuracy = np.zeros((total_data.shape[2], lpo.get_n_splits(total_data)))

    n_split = 0
    for train_index, test_index in lpo.split(total_data):
        X_train, X_test = total_data[train_index], total_data[test_index]
        y_train, y_test = total_labels[train_index], total_labels[test_index]
        for t in range(accuracy.shape[0]):
            clf                  = make_pipeline(SVC(gamma='auto', kernel='linear'))
            clf.fit(X_train[:, :, t], y_train)
            y_pred               = clf.predict(X_test[:, :, t])
            accuracy[t, n_split] = accuracy_score(y_test, y_pred)
        n_split = n_split + 1
        
    with open('unit_data/accuracy/' + probe + '.npy', 'wb') as f:
        np.save(f, accuracy)
        
    acc = accuracy
    accuracy = np.mean(acc, axis=1)
    error    = np.std(acc, axis=1) / np.sqrt(lpo.get_n_splits(total_data))
    t = np.arange(-200, 750, 10)
    x_new   = np.linspace(-200, 740, 1000)
    print(accuracy.shape)
    print(error.shape)
    bspline = interpolate.make_interp_spline(t, accuracy)
    bspline2 = interpolate.make_interp_spline(t, error)
    y_new   = bspline(x_new)
    err_new = bspline2(x_new)
    print(err_new.shape)
    
    sns.set()
    plt.plot(x_new, y_new, linewidth = 2)
    plt.fill_between(x_new, y_new-err_new, y_new+err_new, alpha=0.5)
    plt.xlabel('time (ms)')
    plt.ylabel('accuracy')
    plt.axvline(x = 0, ymin=0.1, ymax=0.9, color = 'r', label = 'stimuli onset', linestyle='dashed')
    plt.legend()
    plt.title('Accuracy over time for ' + probe)
    plt.savefig()

if __name__ == "__main__":
    felineSpike = np.load('unit_data/feline_data/felineMeanSpikeValues_VISp.npy')
    felineSpike = np.reshape(felineSpike, (felineSpike.shape[1], felineSpike.shape[0], felineSpike.shape[2]))
    treeSpike   = np.load('unit_data/tree_data/treeMeanSpikeValues_VISp.npy')
    treeSpike   = np.reshape(treeSpike, (treeSpike.shape[1], treeSpike.shape[0], treeSpike.shape[2]))
    svm_and_plot(felineSpike, treeSpike, 'V1')