# -*- coding: utf-8 -*-
# Odyssee Merveille 25/10/17

import numpy
import sys
import tensorflow.keras.backend as K
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from joblib import Parallel, delayed
import multiprocessing


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

    precision = numpy.divide(tp, numpy.add(tp, fp) + K.epsilon())
    recall = numpy.divide(tp, numpy.add(tp, fn) + K.epsilon())
    f1 = numpy.divide(2 * numpy.multiply(precision, recall), numpy.add(recall, precision) + K.epsilon())
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

    #ps, rs, f1s, _ = precision_recall_fscore_support(gt.flatten(), detection.flatten(), average=None)

    # Ignore the negative class
    p = numpy.mean(cl_p[1:])
    r = numpy.mean(cl_r[1:])
    f1 = numpy.mean(cl_f1[1:])
    acc = numpy.mean(cl_acc[1:])

    return cl_tp, cl_tn, cl_fp, cl_fn, cl_p, cl_r, cl_f1, cl_acc, p, r, f1, acc


def threshold_evaluation_parallel(class_probabilities, gt, threshold_min, threshold_max, threshold_levels, nb_classes, mask=None, verbose=0):
    """

    threshold_evaluation: evaluation the range of threshold values and find the best F1 score, returing its associated
    Precision and Recall

    :param class_probabilities: (numpy.array) in which the 3rd dimension represents the probability of a pixel belonging
    to each class
    :param gt: (numpy.array) the image's ground truth
    :param threshold_min: (int) the minimal threshold to test
    :param threshold_max: (int) the maximal threshold to test
    :param threshold_levels: (int) the number of steps between threshold_min and threshold_max
    :param nb_classes: (int) the number of classes used (negative should be included)
    :param mask: (numpy.array int) a custom mask used for reducing the valid region. Only the area in which the mask has
    a value greater than zero is considered. If not, whole image is considered
    :param verbose: (int) if 1 the function outputs the thresholding and the temporary results
    :return: (array of int, array of int, array of int, float, float, float, float) the results in the order:
        -True Positives: array of size threshold_levels that contains the TP results at each threshold
        -False Negatives: array of size of threshold_levels that contains the FN result at each threshold
        -False Positives: array of size of threshold_levels that contains the FP result at each threshold
        -Best Precision: the precision associated with the highest F1 score
        -Best Recall: the recall associated with the highest F1 score
        -Best F1: the highest F1 score found
        -Best Threshold: the threshold that results in the F1 score
    """

    ###
    # verbose = 0 or 1
    ###

    if len(class_probabilities.shape) == 2:
        class_probabilities = class_probabilities[:, :, None]

    # Compute thresholds
    s_liste = numpy.linspace(threshold_min, threshold_max, num=threshold_levels)

    bestf1 = -1
    bestp = -1
    bestr = -1
    threshold = -1

    f1s = numpy.zeros(s_liste.size, numpy.uint)
    ps = numpy.zeros(s_liste.size, numpy.uint)
    rs = numpy.zeros(s_liste.size, numpy.uint)
    tps = numpy.zeros((s_liste.size, nb_classes), numpy.uint)
    fps = numpy.zeros((s_liste.size, nb_classes), numpy.uint)
    fns = numpy.zeros((s_liste.size, nb_classes), numpy.uint)

    class_probabilities[..., 0] = 0
    class_probabilities[class_probabilities != class_probabilities.max(axis=2, keepdims=1)] = 0

    def test_threshold(i):
        idx = i(0)
        t = i(1)

        detection = numpy.zeros(class_probabilities.shape[:2], dtype=numpy.uint8)
        numpy.argmax(class_probabilities > t, axis=-1, out=detection)

        cl_tps, cl_tns, cl_fps, cl_fns, _, _, _, _, p, r, f1, _ = evaluate_detection(gt, detection, nb_classes, mask=mask)

        if verbose == 1:
            sys.stdout.write("threshold: {:6.2f} --- F1: {:6.4f}, p: {:6.4f}, r: {:6.4f} \r".format(t, f1, p, r))

        return idx, cl_tps, cl_fps, cl_fns, p, r, f1

    num_cores = 2 #multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(test_threshold)(i) for i in enumerate(s_liste))

    for r in results:

        if r(6) > bestf1:
            bestf1 = r(6)
            bestp = r(4)
            bestr = r(5)
            threshold = s_liste(r(0))

        f1s[r(0)] = r(6)
        ps[r(0)] = r(4)
        rs[r(0)] = r(5)
        tps[r(0), :] = r(1)
        fps[r(0), :] = r(2)
        fns[r(0), :] = r(3)

    if verbose == 1:
        print('best threshold: {:6.2f} --- F1: {:6.4f}, p: {:6.4f}, r: {:6.4f}\r'.format(threshold, bestf1, bestp, bestr))

    return tps, fps, fns, bestp, bestr, bestf1, threshold


def threshold_evaluation(class_probabilities, gt, threshold_min, threshold_max, threshold_levels, nb_classes, mask=None, verbose=0):
    """

    threshold_evaluation: evaluation the range of threshold values and find the best F1 score, returing its associated
    Precision and Recall

    :param class_probabilities: (numpy.array) in which the 3rd dimension represents the probability of a pixel belonging
    to each class
    :param gt: (numpy.array) the image's ground truth
    :param threshold_min: (int) the minimal threshold to test
    :param threshold_max: (int) the maximal threshold to test
    :param threshold_levels: (int) the number of steps between threshold_min and threshold_max
    :param nb_classes: (int) the number of classes used (negative should be included)
    :param mask: (numpy.array int) a custom mask used for reducing the valid region. Only the area in which the mask has
    a value greater than zero is considered. If not, whole image is considered
    :param verbose: (int) if 1 the function outputs the thresholding and the temporary results
    :return: (array of int, array of int, array of int, float, float, float, float) the results in the order:
        -True Positives: array of size threshold_levels that contains the TP results at each threshold
        -False Negatives: array of size of threshold_levels that contains the FN result at each threshold
        -False Positives: array of size of threshold_levels that contains the FP result at each threshold
        -Best Precision: the precision associated with the highest F1 score
        -Best Recall: the recall associated with the highest F1 score
        -Best F1: the highest F1 score found
        -Best Threshold: the threshold that results in the F1 score
    """

    ###
    # verbose = 0 or 1
    ###

    if len(class_probabilities.shape) == 2:
        class_probabilities = class_probabilities[:, :, None]

    # Compute thresholds
    s_liste = numpy.linspace(threshold_min, threshold_max, num=threshold_levels)

    bestp = -1
    bestr = -1
    bestf1 = -1
    bestacc = -1
    bestclps = -1
    bestclrs = -1
    bestclf1s = -1
    bestclaccs = -1
    threshold = -1

    ps = numpy.zeros(s_liste.size, numpy.uint)
    rs = numpy.zeros(s_liste.size, numpy.uint)
    f1s = numpy.zeros(s_liste.size, numpy.uint)
    accs = numpy.zeros(s_liste.size, numpy.uint)
    tps = numpy.zeros((s_liste.size, nb_classes), numpy.uint)
    tns = numpy.zeros((s_liste.size, nb_classes), numpy.uint)
    fps = numpy.zeros((s_liste.size, nb_classes), numpy.uint)
    fns = numpy.zeros((s_liste.size, nb_classes), numpy.uint)

    class_probabilities = class_probabilities[..., 1:]
    change = class_probabilities.max(axis=-1)

    detection = numpy.zeros(class_probabilities.shape[:2], dtype=numpy.uint8)
    for i in range(class_probabilities.shape[0]):
        detection[i, :] = numpy.argmax(class_probabilities[i, ...], axis=-1) + 1

    for idx, t in enumerate(s_liste):

        detection[change <= t] = 0

        cl_tps, cl_tns, cl_fps, cl_fns, cl_ps, cl_rs, cl_f1s, cl_accs, p, r, f1, acc = evaluate_detection(gt, detection, nb_classes, mask=mask)

        if verbose == 1:
            sys.stdout.write("threshold: {:6.2f} --- F1: {:6.4f}, p: {:6.4f}, r: {:6.4f}, acc: {:6.4f}\r".format(t, f1, p, r, acc))

        if f1 > bestf1:
            bestp = p
            bestr = r
            bestf1 = f1
            bestacc = acc
            bestclps = cl_ps
            bestclrs = cl_rs
            bestclf1s = cl_f1s
            bestclaccs = cl_accs
            threshold = t

        ps[idx] = p
        rs[idx] = r
        f1s[idx] = f1
        accs[idx] = acc
        tps[idx, :] = cl_tps
        tns[idx, :] = cl_tns
        fps[idx, :] = cl_fps
        fns[idx, :] = cl_fns

    if verbose == 1:
        print('best threshold: {:6.2f} --- F1: {:6.4f}, p: {:6.4f}, r: {:6.4f}, acc: {:6.4f}\r'.format(threshold, bestf1, bestp, bestr, bestacc))

    return tps, tns, fps, fns, bestp, bestr, bestf1, bestacc, bestclps, bestclrs, bestclf1s, bestclaccs, threshold


"""
EXPERIMENTAL VERSION
"""

def update(nb_classes, thresholds, classes, probabilities, gt, mask = None):
    tp = numpy.zeros((nb_classes, len(thresholds)), numpy.int)
    fp = numpy.zeros((nb_classes, len(thresholds)), numpy.int)
    fn = numpy.zeros((nb_classes, len(thresholds)), numpy.int)

    for i in range(probabilities.shape[0]):
        #if i > 0:
        #    sys.stdout.write("{:d}, {:d}, {:6.3f}, {:d}, {:f}\r".format(i, probabilities.shape[0],
        #                                                                (float(i) / probabilities.shape[0]) * 100,
        #                                                                time.time() - t0,
        #                                                                (float(time.time() - t0) / i) * (
        #                                                                            probabilities.shape[0] - i)))
        for j in range(probabilities.shape[1]):

            if mask and mask[i, j]:
                valid_thresholds = numpy.where(thresholds <= probabilities[i, j])
                # if a threshold can act upon the probability then proceed...
                #     if probability is greater than all thresholds, then nothing can change
                #     if probability is less than or equal to all thresholds, then nothing can change
                if valid_thresholds and numpy.any(thresholds < probabilities[i, j]):
                    idx = numpy.argmax(valid_thresholds)
                    if classes[i, j] == gt[i, j]:
                        if gt[i, j] > 0:
                            tp[gt[i, j], idx] -= 1
                            fn[gt[i, j], idx] += 1
                            fp[0, idx] += 1
                    else:
                        if gt[i, j] == 0:
                            tp[gt[i, j], idx] += 1
                            fn[gt[i, j], idx] -= 1
                            fp[classes[i, j], idx] -= 1
                        elif classes[i, j] > 0:
                            fp[0, idx] += 1
                            fp[classes[i, j], idx] -= 1

    return tp, fp, fn


def threshold_evaluation_experimental(class_probabilities, gt, threshold_min, threshold_max, threshold_levels, nb_classes, mask=None, verbose=0):
    """

    threshold_evaluation: evaluation the range of threshold values and find the best F1 score, returing its associated
    Precision and Recall

    :param class_probabilities: (numpy.array) in which the 3rd dimension represents the probability of a pixel belonging
    to each class
    :param gt: (numpy.array) the image's ground truth
    :param threshold_min: (int) the minimal threshold to test
    :param threshold_max: (int) the maximal threshold to test
    :param threshold_levels: (int) the number of steps between threshold_min and threshold_max
    :param nb_classes: (int) the number of classes used (negative should be included)
    :param mask: (numpy.array int) a custom mask used for reducing the valid region. Only the area in which the mask has
    a value greater than zero is considered. If not, whole image is considered
    :param verbose: (int) if 1 the function outputs the thresholding and the temporary results
    :return: (array of int, array of int, array of int, float, float, float, float) the results in the order:
        -True Positives: array of size threshold_levels that contains the TP results at each threshold
        -False Negatives: array of size of threshold_levels that contains the FN result at each threshold
        -False Positives: array of size of threshold_levels that contains the FP result at each threshold
        -Best Precision: the precision associated with the highest F1 score
        -Best Recall: the recall associated with the highest F1 score
        -Best F1: the highest F1 score found
        -Best Threshold: the threshold that results in the F1 score
    """

    ###
    # verbose = 0 or 1
    ###

    if len(class_probabilities.shape) == 2:
        class_probabilities = class_probabilities[:, :, None]

    # Compute thresholds
    s_liste = numpy.linspace(threshold_min, threshold_max, num=threshold_levels)

    # Todo: doesnt work when probabilities are split exactly between classes
    class_probabilities[..., 0] = 0
    class_probabilities[class_probabilities != class_probabilities.max(axis=2, keepdims=1)] = 0
    detection = numpy.zeros(class_probabilities.shape[:2], dtype=numpy.uint8)
    numpy.argmax(class_probabilities > t, axis=-1, out=detection)
    tps, tns, fps, fns, _, _, _, _, p, r, f1, _ = evaluate_detection(gt, detection, nb_classes, mask=mask)

    classes = numpy.argmax(class_probabilities, axis=2)
    probabilities = numpy.max(class_probabilities, axis=2)
    tps_u, fps_u, fns_u = update(nb_classes, s_liste, classes, probabilities, gt, mask=mask)

    tps = numpy.add(tps, numpy.transpose(tps_u), casting="unsafe", dtype=numpy.uint)
    tns = numpy.add(tns, numpy.transpose(tns_u), casting="unsafe", dtype=numpy.uint)  # todo: Need to determine update rule for tns
    fps = numpy.add(fps, numpy.transpose(fps_u), casting="unsafe", dtype=numpy.uint)
    fns = numpy.add(fns, numpy.transpose(fns_u), casting="unsafe", dtype=numpy.uint)

    pss, rss, f1ss, _ = precision_recall_f1_accuracy(tps, tns, fps, fns)
    ps = numpy.mean(pss[:, 1:], axis=1)     # Ignore negative class
    rs = numpy.mean(rss[:, 1:], axis=1)
    f1s = numpy.mean(f1ss[:, 1:], axis=1)

    bestf1 = numpy.amax(f1s)
    bestf1_ind = numpy.argmax(f1s)
    bestp = ps[bestf1_ind]
    bestr = rs[bestf1_ind]
    best_threshold = s_liste[bestf1_ind]

    if verbose == 1:
        print('best threshold: {:6.2f} --- F1: {:6.4f}, p: {:6.4f}, r: {:6.4f}\r'.format(best_threshold, bestf1, bestp, bestr))

    return tps, fps, fns, bestp, bestr, bestf1, best_threshold
