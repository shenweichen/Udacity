import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus
import itertools


def plot_pr(y_test,y_score,):
    """
    y_test: label_binarize labels
    y_score: return value of decision_function
    """
    # setup plot details
    colors = itertools.cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    lw = 2
    # Compute Precision-Recall and plot curve
    n_classes = y_test.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = metrics.average_precision_score(y_test[:, i], y_score[:, i])

    # Compute micro-average ROC curve and ROC area
    precision["micro"], recall["micro"], _ = metrics.precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = metrics.average_precision_score(y_test, y_score,
                                                         average="micro")


    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[0], precision[0], lw=lw, color='navy',
             label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision[0]))
    plt.legend(loc="lower left")
    plt.show()

    # Plot Precision-Recall curve for each class
    plt.clf()
    plt.plot(recall["micro"], precision["micro"], color='gold', lw=lw,
             label='micro-average Precision-recall curve (area = {0:0.2f})'
                   ''.format(average_precision["micro"]))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=lw,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    class: class names
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def onehot_transform(X, names=None,prefix_sep='_'):
    dummies_X = pd.get_dummies(X,prefix_sep=prefix_sep)
    if names is None:
        return dummies_X, dummies_X.columns.values
    else:
        return pd.DataFrame(dummies_X, columns=names).fillna(0)


def report(results, n_top=3):
    """print grid_search.cv_results_ """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def get_name_value(features, values):
    """get feature and values"""
    return pd.DataFrame({'name': features, 'value': values, 'abs_': np.abs(values)})


def count_show(series):
    """return series count_table and draw plot"""
    index = series.name
    count = series.value_counts()
    table = pd.DataFrame(
        {index: count.index, 'Count': count.values}, columns=[index, 'Count'])
    table['Percent'] = table['Count'].apply(
        lambda x: float(x) / series.count())
    sns.countplot(series, order=table[index])
    return table


def count_words(X, voc):
    """
    X: the return matrix of CountVectorizer.transform
    voc : vect.vocabulary_
    """
    rvoc = dict((v, k) for k, v in voc.iteritems())

    def count(row_id):
        dic = dict()
        for ind in X[row_id, :].indices:
            dic[rvoc[ind]] = X[row_id, ind]
        return dic
    word_count = map(count, range(0, X.shape[0]))
    return word_count


def plot_tree(decision_tree, out_file=None, max_depth=None,
              feature_names=None, class_names=None, label='all',
              filled=True, leaves_parallel=False, impurity=True,
              node_ids=False, proportion=False, rotate=False,
              rounded=True, special_characters=True):

    dot_data = tree.export_graphviz(decision_tree, out_file=out_file,
                                    max_depth=max_depth, feature_names=feature_names,
                                    class_names=class_names, label=label, filled=filled,
                                    leaves_parallel=leaves_parallel, impurity=impurity, node_ids=node_ids,
                                    proportion=proportion, rotate=rotate, rounded=rounded,
                                    special_characters=special_characters)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return Image(graph.create_png())


def plot_roc(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True):
    u"""draw bianry classification roc_curve
    Parameters

    ----------

    y_true : array, shape = [n_samples]

        True binary labels in range {0, 1} or {-1, 1}.  If labels are not

        binary, pos_label should be explicitly given.



    y_score : array, shape = [n_samples]

        Target scores, can either be probability estimates of the positive

        class, confidence values, or non-thresholded measure of decisions

        (as returned by "decision_function" on some classifiers).



    pos_label : int or str, default=None

        Label considered as positive and others are considered negative.



    sample_weight : array-like of shape = [n_samples], optional

        Sample weights.



    drop_intermediate : boolean, optional (default=True)

        Whether to drop some suboptimal thresholds which would not appear

        on a plotted ROC curve. This is useful in order to create lighter

        ROC curves.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        y_true, y_score, pos_label=1, drop_intermediate=False)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
