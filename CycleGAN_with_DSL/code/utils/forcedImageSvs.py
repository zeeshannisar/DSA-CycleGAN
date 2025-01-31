from openslide import OpenSlide
import numpy
import warnings
import skimage


class ForcedOpenSlide(OpenSlide):

    def __init__(self, imagePath):

        super(ForcedOpenSlide, self).__init__(imagePath)

    def _find_next_lod(self, scaleFactor):

        vecOfScales = numpy.asarray(self.level_downsamples)

        string = [str(int(a)) for a in self.level_downsamples]
        warnings.warn(
            'The svs image does not contain an image with scaleFactor %i \n\t scales are: %s' % (scaleFactor, string))

        # Find the higher lod which is smaller than "lod"
        lowerScaleFactor = -1
        for indice in range(vecOfScales.size):
            currentScaleFactor = vecOfScales[indice]
            if (currentScaleFactor > lowerScaleFactor and currentScaleFactor < scaleFactor):
                lowerScaleFactor = currentScaleFactor
                indiceLowerScaleFactor = indice

        if lowerScaleFactor == -1:
            raise ValueError(
                'The svs image does not contain an image with scaleFactor %i and no lower scale factor to interpolate it: %s' % (
                    scaleFactor, string))

        return lowerScaleFactor, indiceLowerScaleFactor

    def get_level_dimension(self, lod):
        scaleFactor = 2 ** lod
        vecOfScales = numpy.asarray(self.level_downsamples)
        if numpy.where(vecOfScales.astype(int) == scaleFactor)[0].size > 0:
            return self.level_dimensions[numpy.where(vecOfScales.astype(int) == scaleFactor)[0][0]]
        else:
            lowerScaleFactor, indiceLowerScaleFactor = self._find_next_lod(scaleFactor)

            dimX, dimY = self.level_dimensions[indiceLowerScaleFactor]

            return int(dimX / (scaleFactor / lowerScaleFactor)), int(dimY / (scaleFactor / lowerScaleFactor))

    def read_region(self, location, lod, size):

        scaleFactor = 2 ** lod
        vecOfScales = numpy.asarray(self.level_downsamples)
        origScaleFactor = int(vecOfScales[0])

        if numpy.where(vecOfScales.astype(int) == scaleFactor)[0].size > 0:
            level = numpy.where(vecOfScales.astype(int) == scaleFactor)[0][0]

            coordFactor = scaleFactor / origScaleFactor
            location = [int(l * coordFactor) for l in location]

            image = super(ForcedOpenSlide, self).read_region(location, level, size)

            return numpy.asarray(image.convert('RGB'))
        else:
            lowerScaleFactor, indiceLowerScaleFactor = self._find_next_lod(scaleFactor)
            coordFactor = scaleFactor / origScaleFactor
            lowerCoordFactor = scaleFactor / lowerScaleFactor

            location = [int(l * coordFactor) for l in location]

#            print("\t Rescaling from scaleFactor %i" % lowerScaleFactor)
            lower_size = [int(s * lowerCoordFactor) for s in size]

            # Read the image at the lower lod
            imageLowerLod = super(ForcedOpenSlide, self).read_region(location, indiceLowerScaleFactor, lower_size)
            imageLowerLod = numpy.asarray(imageLowerLod.convert('RGB'))

            image = skimage.transform.resize(imageLowerLod, (size[1], size[0], imageLowerLod.shape[2]), order=1, mode='reflect', clip=True, preserve_range=True, anti_aliasing=False)
            return image.astype(numpy.uint8)
