"""
RJ stain deconvolution methods
"""

import stain_normalisation.utils.misc_utils as mu
import numpy as np

class RuifrokJohnstonDeconvolution(object):
    """
    Stain deconvolution method according to:
    A. C. Ruifrok, D. A. Johnston et al., “Quantification of histochemical
    staining by color deconvolution,” Analytical and quantitative cytology
    and histology, vol. 23, no. 4, pp. 291–299, 2001.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_stain_matrix(staincode, *args):
        """
        Get RJ stain matrix.
        A. C. Ruifrok, D. A. Johnston et al., “Quantification of histochemical
        staining by color deconvolution,” Analytical and quantitative cytology
        and histology, vol. 23, no. 4, pp. 291–299, 2001.
        :param args: a dummy
        :return:
        """
        if staincode == '02':
            # PAS [Haematoxyln and PAS]
            stain_matrix = np.array([[0.641223, 0.712470, 0.284988],
                                     [0.175411, 0.972178, 0.154589],
                                     [0.000000, 0.000000, 0.000000]])
            zerodchannels = [2]
            #stain_matrix[2, :] = np.cross(stain_matrix[0, :], stain_matrix[1, :])
        elif staincode == '03':
            # Jones H&E [Jones and Haematoxyln and Eosin]
            #stain_matrix = np.array([[0.644211, 0.716556, 0.266844],
            #                         [0.092789, 0.954111, 0.283111],
            #                         [-0.0903, -0.2752, 0.9571]])
            # Todo: need to add Jones (silver)
            stain_matrix = np.array([[0.641223, 0.712470, 0.284988],
                                     [0.076500, 0.994100, 0.105000],
                                     [0.000000, 0.000000, 0.000000]])
            zerodchannels = [2]
            #stain_matrix[2, :] = np.cross(stain_matrix[0, :], stain_matrix[1, :])
        elif staincode == '16':
            # CD68 (PG-M1) [Haematoxyln and DAB]
            stain_matrix = np.array([[0.641223, 0.712470, 0.284988],
                                     [0.268000, 0.570000, 0.776000],
                                     [0.000000, 0.000000, 0.000000]])
            zerodchannels = [2]
            stain_matrix[2, :] = np.cross(stain_matrix[0, :], stain_matrix[1, :])
        elif staincode == '32' or staincode == '55':
            # Sirius Red & Haematoxyln
            # [26.863525, 54.626385, 50.577736] / length = [0.3394, 0.6902, 0.6391]
            # Sequestration of Vascular Endothelial Growth Factor (VEGF) Induces Restrictive Lung Disease
            # https://doi.org/10.1371/journal.pone.0148323.s001
            # Todo: check correct sirius red definition
            stain_matrix = np.array([[0.641223, 0.712470, 0.284988],
                                     [0.339425, 0.690214, 0.639058],
                                     [0.000000, 0.000000, 0.000000]])
            zerodchannels = [2]
        elif staincode == '39':
            # CD34: ultraView Universal Alkaline Phosphatase Red Detection Kit
            # Todo: need to add Universal Alkaline Phosphatase Red Detection Kit definition
            stain_matrix = np.array([[0.641223, 0.712470, 0.284988],
                                     [0.339425, 0.690214, 0.639058],
                                     [0.000000, 0.000000, 0.000000]])
            zerodchannels = [2]
        else:
            raise ValueError('Unknown Stain Code')

        # Fill missing vectors
        lengths = np.sqrt(np.sum(np.square(stain_matrix), axis=1))
        lengths[lengths == 0] = 1
        stain_matrix = stain_matrix / lengths[:, np.newaxis]

        if np.all(stain_matrix[1, :] == 0.0):
            stain_matrix[1, 0] = stain_matrix[0, 2]
            stain_matrix[1, 1] = stain_matrix[0, 0]
            stain_matrix[1, 2] = stain_matrix[0, 1]

        if np.all(stain_matrix[2, :] == 0.0):
            if stain_matrix[0, 0] * stain_matrix[0, 0] + stain_matrix[1, 0] * stain_matrix[1, 0] > 1:
                stain_matrix[2, 0] = 0.0
            else:
                stain_matrix[2, 0] = np.sqrt(
                    1.0 - (stain_matrix[0, 0] * stain_matrix[0, 0]) - (stain_matrix[1, 0] * stain_matrix[1, 0]))

            if stain_matrix[0, 1] * stain_matrix[0, 1] + stain_matrix[1, 1] * stain_matrix[1, 1] > 1:
                stain_matrix[2, 1] = 0.0
            else:
                stain_matrix[2, 1] = np.sqrt(
                    1.0 - (stain_matrix[0, 1] * stain_matrix[0, 1]) - (stain_matrix[1, 1] * stain_matrix[1, 1]))

            if stain_matrix[0, 2] * stain_matrix[0, 2] + stain_matrix[1, 2] * stain_matrix[2, 2] > 1:
                stain_matrix[2, 2] = 0.0
            else:
                stain_matrix[2, 2] = np.sqrt(
                    1.0 - (stain_matrix[0, 2] * stain_matrix[0, 2]) - (stain_matrix[1, 2] * stain_matrix[1, 2]))

        stain_matrix[2, :] = np.divide(stain_matrix[2, :], np.sqrt(np.sum(np.square(stain_matrix[2, :]))))

        stain_matrix[stain_matrix == 0] = 0.001

        lengths = np.sqrt(np.sum(np.square(stain_matrix), axis=1))
        stain_matrix = stain_matrix / lengths[:, np.newaxis]

        return stain_matrix, zerodchannels

    @staticmethod
    def get_concentrations(I, stain_matrix):
        """
        Performs stain concentration extraction according to
        A. C. Ruifrok, D. A. Johnston et al., “Quantification of histochemical
        staining by color deconvolution,” Analytical and quantitative cytology
        and histology, vol. 23, no. 4, pp. 291–299, 2001.
        :param I:
        :return:
        """

        OD = mu.RGB_to_OD(I).reshape((-1, 3))
        source_concentrations = np.dot(OD, np.linalg.inv(stain_matrix))
        return source_concentrations
