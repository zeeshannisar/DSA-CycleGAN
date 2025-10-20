import glob
import os
from utils import config_utils
import argparse
import fnmatch
import re


def validlabel(label):
    return not '.' in label

class FilepathGenerator:
    """
        FilepathGenerator class used for extract and find the different filepaths for a project

    """

    def __init__(self, configfile):
        self.config = configfile
        self.regex = self.config["general.regexBaseName"]

        self.regexLod = "_lod(lod)"
        self.regexGT = "_gt" + self.regexLod
        self.regexGTlabels = "_gt_labels" + self.regexLod
        self.regexMask = "_mask_(class)" + self.regexLod
        self.regexTissue = "_mask_tissue" + self.regexLod
        self.regexPatches = "_(type)_patch_(number)"
        self.regexAugment = "(type)_" + self.regex + self.regexTissue + "_(numberAugment)"

    # General methods

    def __get_patient_zeropadding(self):
        # get the zero padding for the patient number (if any)
        """
          __get_patient_zeropadding: extract the zero padding definition from the regex

          :param reg: (string) the specific regex, the substring "(patient)" must exist

          :exception ValueError
          :return: (int) the patient zero padding found
        """

        elements = self.regex.rsplit("_")
        try:
            element = fnmatch.filter(elements, "(patient*)")[0]
            paddingelements = element.rsplit("$")
            if len(paddingelements) == 2:
                return int(paddingelements[1][:-1])
            else:
                return 0
        except Exception as inst:
            print("Cannot extract the zero padding from regex %s" % self.regex)
            raise inst

    def __replace_patient_string(self, string, patient):
        return re.sub(r'\(patient+(\$\d+)?\)', patient, string)

    def get_stain(self, ImageName):
        """
         get_stain: extract the stain of the specific imageName using the regex

         :param imageName: (string) the basename of the image (without the extension)
         :param reg: (string) the specific regex it need to have a substring "(stain)" to work

         :exception ValueError if the
         :return: (string) the stain found
        """

        elements = self.regex.rsplit("_")
        try:
            stainIndex = elements.index("(stain)")

            return os.path.splitext(ImageName)[0].rsplit("_")[stainIndex]
        except Exception as inst:
            print("Cannot extract the stain from the file name")
            print("FileName : %s " % ImageName)
            print("RegexUse : %s " % self.regex)
            raise inst


    def __get_patient(self, ImageName):
        """
          get_patient: extract the patient of the specific imageName using the regex

          :param imageName: (string) the basename of the image (without the extension)
          :param reg: (string) the specific regex it need to have a substring "(patient)" to work

          :exception ValueError
          :return: (string) the patient found
        """

        elements = self.regex.rsplit("_")
        try:
            element = fnmatch.filter(elements, "(patient*)")[0]
            patientIndex = elements.index(element)
        except Exception as inst:
            print("Cannot extract the patient from the file name")
            print("FileName : %s " % ImageName)
            print("RegexUse : %s " % self.regex)
            raise inst

        return ImageName.rsplit("_")[patientIndex].lstrip("0")

    def get_lod(self, ImageName):
        """
          get_lod: extract the lod of the specific imageName using the regex

          :param imageName: (string) the basename of the image (without the extension)
          :param reg: (string) the specific regex it need to have a substring "(lod)" to work

          :exception ValueError
          :return: (string) the lod found
        """

        elements = self.regex.rsplit("_")
        try:
            lodIndex = elements.index("(lod)")
        except Exception as inst:
            print("Cannot extract the lod from the file name")
            print("FileName : %s " % ImageName)
            print("RegexUse : %s " % self.regex)
            raise inst

        return ImageName.rsplit("_")[lodIndex]

    # For extract_masks_from_cytomine
    def generate_maskpath(self, imageName, className, lod, output_path=None):
        """
        generate_maskpath_from_cytomine generate the path used for the mask in the project
        :param output_path: (string)
        :param imageName: (string) the basename of the image used (without the extension)
        :param className: (string) the name of the class reprensented by the mask
        :param lod: (int) the level of detail
        :return: (string) the full path of the mask image generated
        """
        if output_path is None:
            output_path = self.config['extraction.maskpath']

        if not os.path.exists(os.path.join(output_path, className)):
            os.makedirs(os.path.join(output_path, className))

        maskstring = self.regexMask.replace("(class)", className).replace("(lod)", str(lod))

        return os.path.join(output_path, className, imageName + maskstring + ".png")

    # For create_ground_truths
    def find_filenames_with_specific_lod(self, classFolder, lod, maskDir=None):
        """

        find_filenames_with_specific_lod: get all files with the specified lod

        :param maskDir: (string) the folder where the file are
        :param classFolder: (string) the class folder used if not given the function takes all the classes
        :param lod: (int) the specific level of detail
        :return: (list of (string)) the list that contains
        """
        if maskDir is None:
            maskDir = self.config['extraction.maskpath']

        lodstr = self.regexLod.replace("(lod)", str(lod))

        listMasks = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(maskDir, classFolder, "*" + lodstr + ".png")))]

        res = []
        # Remove the _lod
        for mask in listMasks:
            bricks = mask.split("_")[:-3]
            res.append("_".join(bricks))
            #print ("find filenames %s" % res[-1])

        return res

    def generate_groundtruthpath(self, imageName, lod, output_path=None):
        """

        generate_groundtruthpath: generate the ground truth image path

        :param output_path: (string) the output folder used
        :param imageName: (string) the basename image name without the extension
        :param lod: (int) the level of detail
        :return: (string) the path of the image
        """

        if output_path is None:
            output_path = self.config['extraction.groundtruthpath']

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        gtstring = self.regexGT.replace("(lod)", str(lod))

        return os.path.join(output_path, imageName + gtstring + ".png")

    def generate_groundtruthlabelspath(self, imageName, lod):
        """

        generate_groundtruthpath: generate the ground truth image path

        :param output_path: (string) the output folder used
        :param imageName: (string) the basename image name without the extension
        :param lod: (int) the level of detail
        :return: (string) the path of the image
        """

        output_path = self.config['extraction.groundtruthpath']

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        gtlabelsstring = self.regexGTlabels.replace("(lod)", str(lod))

        return os.path.join(output_path, imageName + gtlabelsstring + ".json")

    def get_images(self, imagePath=None, staincode='*', patient='*'):
        """

        get_images: get a list of images in the specified folder. Can filter images by patient number and staincode

        :param imagePath: (string) the image folder
        :param staincode: (string) the image stain filter argument, if ignored all stain codes are returned
        :param patient: (string) the image patient filter argument, if ignored all patients are returned
        :return: list of (string, string) the list of images found in the path, filtered by staincode and patient if
        given. The format is a tuple of the image path and its basename without the extension
        """
        if imagePath is None:
            imagePath = self.config['extraction.imagepath']

        if patient != '*':
            patient = patient.zfill(self.__get_patient_zeropadding())

        namestr = self.regex.replace("(x)", "*").replace("(stain)", staincode).replace("(patient)", patient)
        namestr = self.__replace_patient_string(namestr, patient)

        listImage = [(f, f.split(os.sep)[-1].split(".")[0])
                for f in sorted(glob.glob(os.path.join(imagePath, namestr + ".*")))]

        return listImage

    def get_images_with_list_patients(self, imagePath=None, staincode='*', patients=None):
        """
        get_images_with_list_patients: get a list of images filtered by a list of patient numbers


        :param imagePath: (string) the image folder
        :param staincode: (string) the image stain filter argument, if ignored all stain codes are returned
        :param patients: list of (string) the image patient filter argument, if ignored all patients are returned
        :return: list of (string, string) the list of images found in the path, filtered by staincode and patients if
        given. The format is a tuple of the image path and its basename without the extension
        """
        if imagePath is None:
            imagePath = self.config['extraction.imagepath']

        if patients is None:
            return self.get_images(imagePath, staincode)
        else:
            listImages = []
            for patient in patients:
                resultImage = self.get_images(imagePath, staincode, patient)
                if len(resultImage) != 0:
                    listImages.extend(resultImage)
            #print("list Image %s" % listImages)
            return listImages

    # Extract patches
    # Normalisation
    def get_patient_from_image(self, imageName):
        """

        get_patient_from_image: extract the patient from a filename

        :param imageName: (string) the basename of the image (without the extension)
        :return:
        """

        return self.__get_patient(imageName)

    def getpatientlist(self, datapath=None):
        """
        getpatientlist: get the list of all patients in the dataset

        :param datapath: (string) the path of the data
        :return: (list of string) the list of existing patients
        """

        if datapath is None:
            datapath = self.config['extraction.imagepath']

        # Patients with images and gts
        images = self.get_images(datapath)

        patients = []
        for _, imageName in images:
            patient = self.get_patient_from_image(imageName)
            if patient not in patients:
                patients.append(patient)

        return patients

    def get_patient_images_with_gts(self, lod, datapath=None, patients=None):
        """
        getpatientimageswithgts: for a group of patients, this function finds the path of all their images that have
        ground truths associated with them (and returns them)

        :param datapath: (string) the path to the data
        :param lod: (int) level of detail required
        :param patients: (None, int, or list of int) the list of patients
        :return: list of (string,string) paths to the images and their related groundtruths
        """
        print('load: {}'.format(lod))
        if datapath is None:
            datapath = self.config['extraction.extractbasepath']
        print('datapath: {}'.format(datapath))
        print('patients: {}'.format(patients))

        images = []
        gts = []

        imagesList = self.get_images_with_list_patients(imagePath=os.path.join(datapath, "images"), patients=patients)
        print(imagesList)
        for imagePath, imageName in imagesList:

            gtPath = self.get_groundtruth(imageName, lod, os.path.join(datapath, 'groundtruths'))

            if os.path.exists(gtPath):
                images.append(imagePath)
                gts.append(gtPath)

        print(images, gts)
        return images, gts


    def get_groundtruth(self, imageName, lod, groundtruthpath=None):
        """

        get_groundtruth: get the specific ground truth from the path

        :param groundtruthpath: (string) the ground truth path
        :param imageName: (string) the image name (without the extension)
        :param lod: (string) the level of detail used
        :return: the ground truth path used
        """
        if groundtruthpath is None:
            groundtruthpath = self.config['extraction.groundtruthpath']

        #print("imageName %s" % imageName)
        #print("lod %s" % lod)

        gtstring = self.regexGT.replace("(lod)", str(lod))

        gt = os.path.join(groundtruthpath, imageName + gtstring + ".png")

        return gt

    def get_masks(self, maskpath, label, lod='*'):
        """

        get_masks: get all masks for the specified label

        :param maskpath: (string) the path of the mask folder
        :param label: (string) the mask label
        :param lod: (int) the level of detail used
        :return: list of (string, string) the list of the path and the basename of the images
        """

        lodstr = self.regexLod.replace("(lod)", str(lod))
        return [(f, f.split(os.sep)[-1].split(".")[0])
                for f in sorted(glob.glob(os.path.join(maskpath, label, "*" + lodstr + ".png"))) if os.path.isfile(f)]

    def get_mask_for_image(self, imageName, label, lod="*"):
        """

        get_mask_for_image get the mask corresponding to imageName

        :param imageName: (string) the image name specified (without the extension)
        :param label: (string) the mask label
        :param lod: (int) the level of detail
        :return: (string, string) the path and the basename of the tissue image
        """
        maskpath = self.config['extraction.maskpath']

        listMasks = self.get_masks(maskpath, label, lod)

        masks = [f for f in listMasks if imageName in f[1]]

        if len(masks) < 1:
            raise ValueError("no %s masks found" % imageName)
        elif len(masks) > 1:
            raise ValueError("not unique %s tissue masks found" % imageName)

        return masks[0]

    # For pixel_based_evaluation

    def get_result_path(self, uNet_model_rep):
        return os.path.join(self.config["detector.outputpath"], uNet_model_rep, 'results')

    def get_average_result_path(self, label):
        return os.path.join(self.config["detector.outputpath"], 'results', label, 'averages')

    # Detection Path


    # Segmentation Path

    def get_segmented_folder(self, label, folderPath=None):

        if folderPath is None:
            folderPath= self.config['segmentation.segmentationpath']

        pathSegmentation = os.path.join(folderPath, label, '*')

        return [(path, path.split('/')[-1]) for path in glob.glob(pathSegmentation)]

    def get_threshold_folder(self, label):

        return self.get_segmented_folder(label, self.config['segmentation.threscomppath'])


if __name__ == '__main__':
    """
    Debug and test propose only

    #SYS_TN_043_E_13_30222_16_005.svs
    regexName = "(x)_(x)_(x)_(x)_(x)_(patient)_(x)_(stain)"
    regexLod = "_lod(lod)"
    regexGT = "_gt" + lod
    regexMask = "_mask_(class)" + regexLod
    regexTissue = "_mask_tissue" + regexLod
    regexPatches = "_(type)_patch_(number)"
    regexAugment = "(type)_" + regexName + regexPatches + "_(numberAugment)"

    parser = argparse.ArgumentParser(description='Extract patches.' )

    parser.add_argument('-c', '--configfile', type=str, help='the configuration file to use')
    args = parser.parse_args()

    if args.configfile:
        config = config_utils.readconfig(args.configfile)
    else:
        config = config_utils.readconfig()

    path = os.path.join(config['extraction.extractbasepath'], "images")
    files = [f for f in glob.glob(os.path.join(path, "*")) if os.path.isfile(f)]

    for file in files:
        fileprefix = os.path.basename(file)
        print (fileprefix)

        print (get_patient(fileprefix, regexName))
        print (get_stain(fileprefix, regexName))
    """
