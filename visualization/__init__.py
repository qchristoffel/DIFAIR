import sklearn.metrics as metrics
import visualization.features
import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(real_labels, predicted_labels, label_names=None, save_path=None):
    cm = metrics.confusion_matrix(real_labels, predicted_labels)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot()
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
        
def confusion_unknown(unknown_labels, predicted_labels, 
                      classes_dict, nb_classes, save_path=None):
    true_unknown_labels = [k for k, v in classes_dict.items() if k < 0]
    print("True unknown labels:", true_unknown_labels)
    
    matrix = np.zeros((len(true_unknown_labels), nb_classes), dtype=np.int32)
    
    for i, true_label in enumerate(true_unknown_labels):
        for j in range(nb_classes):
            matrix[i, j] = np.sum((unknown_labels == true_label) & (predicted_labels == j))
    
    print(matrix)
    
    plt.figure()
    plt.imshow(matrix)
    for i, true_label in enumerate(true_unknown_labels):
        for j in range(nb_classes):
            plt.text(j, i, matrix[i, j], ha="center", va="center", color="white")
    
    plt.colorbar()
    plt.xlabel("Predicted labels")
    plt.xticks(range(nb_classes), [classes_dict[i] for i in range(nb_classes)], rotation=45)
    
    plt.ylabel("True unknown labels")
    plt.yticks(range(len(true_unknown_labels)), [classes_dict[i] for i in true_unknown_labels])

def vizualize_weights(weights, class_index, nb_features, title, nb_splits=3, save_path=None, ncols=3):
    begin_index = class_index*nb_features
    end_index = (class_index+1)*nb_features
    weights = weights[:,:,:, begin_index:end_index]
    weights = weights.reshape(-1, nb_features)

    split_values = np.quantile(weights, np.linspace(0,1,nb_splits, endpoint=False))
    split_values = split_values[1:]

    new_weights = np.zeros(weights.shape)
    for i in range(nb_splits-2):
        print("i:", i)
        new_weights[np.logical_and(weights >= split_values[i], weights < split_values[i+1])] = i+1
    new_weights[weights >= split_values[-1]] = nb_splits-1

    nrows = np.ceil(nb_features / ncols).astype(int)
    width = int(3 * ncols)
    height = int(4 * nrows)
    fig, axs = plt.subplots(ncols, nrows, sharey=True, figsize=(width, height))
    axs = axs.flatten()

    fig.suptitle(title, fontsize=16)

    for i in range(nb_features):
        weights_feature = new_weights[:,i].flatten()
        size = weights_feature.shape[0]
        side_size = int(np.ceil(np.sqrt(size))) 
        weights_feature = np.concatenate([weights_feature, np.zeros(int(side_size**2 - size))])

        axs[i].imshow(weights_feature.reshape(side_size, side_size), cmap="OrRd")
        axs[i].set_title(f"Feature {i}")
    
    for i in range(nb_features, len(axs)):
        axs[i].axis('off')
    
