import numpy

def tp_tn_fp_fn(detection, gt, nb_classes, mask=None):
    """
    tp_tn_fp_fn: Calculate the number of true positives, false positives, and false negatives between detection and gt
    the calculation is pixel-wise comparison

    :param detection: (numpy.array int) the result of the detection on an image
    :param gt: (numpy.array int) the ground truth associated with the image, contains one pixel per class
    :param nb_classes: (int) the number of classes contained in the ground truth and the detection
    :param mask: (numpy.array int) a custom mask used for reducing the valid region. Only the area in which the mask has
     a value greater than zero is considered. If not, whole image is considered
    :return: (array of int, array of int, array of int) returns the True Positives (TP), True Negative (TN), False
    Positives (FP) and the False Negatives (FN) for each class, the size of the three arrays are equals to nb_classes
    """

    tp = numpy.zeros(nb_classes, numpy.uint)
    tn = numpy.zeros(nb_classes, numpy.uint)
    fp = numpy.zeros(nb_classes, numpy.uint)
    fn = numpy.zeros(nb_classes, numpy.uint)

    if mask is None:
        for i in range(nb_classes):
            tp[i] = numpy.sum(numpy.logical_and(detection == i, gt == i))
            tn[i] = numpy.sum(numpy.logical_and(detection != i, gt != i))
            fp[i] = numpy.sum(numpy.logical_and(detection == i, gt != i))
            fn[i] = numpy.sum(numpy.logical_and(detection != i, gt == i))
    else:
        mask = mask > 0
        for i in range(nb_classes):
            tp[i] = numpy.sum(numpy.logical_and(numpy.logical_and(detection == i, gt == i), mask))
            tn[i] = numpy.sum(numpy.logical_and(numpy.logical_and(detection != i, gt != i), mask))
            fp[i] = numpy.sum(numpy.logical_and(numpy.logical_and(detection == i, gt != i), mask))
            fn[i] = numpy.sum(numpy.logical_and(numpy.logical_and(detection != i, gt == i), mask))

    return tp, tn, fp, fn


def precision_recall_f1_accuracy(tp, tn, fp, fn):
    """
    precision_recall_f1: Calculate the precision, recall, and F1 score based on the true positives, false positives, and
    false negatives

    :param tp: (numpy.array int) the true positives
    :param fp: (numpy.array int) the false positives
    :param fn: (numpy.array int) the false negatives
    :return: (numpy.array float, numpy.array float, numpy.array float) the  precision, recall, and F1 score
    """

    tp = tp.astype(numpy.float)
    tn = tn.astype(numpy.float)
    fp = fp.astype(numpy.float)
    fn = fn.astype(numpy.float)

    precision = numpy.divide(tp, numpy.add(tp, fp) + 1e-6)
    recall = numpy.divide(tp, numpy.add(tp, fn) + 1e-6)
    f1 = numpy.divide(2 * numpy.multiply(precision, recall), numpy.add(recall, precision) + 1e-06)
    accuracy = numpy.divide(numpy.add(tp, tn), numpy.add(numpy.add(tp, tn), numpy.add(fp, fn)))

    return precision, recall, f1, accuracy


def evaluate_detection(gt, detection, nb_classes, mask=None):
    """

    evaluate_detection: evaluate for one detection the pixel-wise score according to the ground truth for all classes

    :param gt: (numpy.array int) the ground truth of the image, containing one class label for each pixel
    :param detection: (numpy.array int) the detection result
    :param nb_classes: (int) the number of classes contained in the ground truth and the detection
    :param mask: (numpy.array int) a custom mask used for reducing the valid region. Only the area in which the mask has
    a value greater than zero is considered. If not, whole image is considered
    :return: (numpy.array int, numpy.array int, numpy.array int, numpy.array float, numpy.array float, numpy.array float)
    In order: the True Positives, False Positives, and False Negatives, the size of each array is equal to nb_classes.
    The Precision, Recall and F1 score list excluding the negative class, the array sizes are nb_Classes-1.
    """

    cl_tp, cl_tn, cl_fp, cl_fn = tp_tn_fp_fn(detection, gt, nb_classes, mask)

    cl_p, cl_r, cl_f1, cl_acc = precision_recall_f1_accuracy(cl_tp, cl_tn, cl_fp, cl_fn)


    # Ignore the background class
    p = numpy.mean([cl_p[0],cl_p[2]])
    r = numpy.mean([cl_r[0],cl_r[2]])
    f1 = numpy.mean([cl_f1[0],cl_f1[2]])
    acc = numpy.mean([cl_acc[0],cl_acc[2]])

    return cl_p, cl_r, cl_f1, cl_acc,[f1,p,r,acc]