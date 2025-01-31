from PIL import Image
import glob
import os

def test_images(dir):
    for imagePath in glob.glob(os.path.join(dir, '*.svs')):
        try:
            imageSvs = openslide.OpenSlide(imagePath)
        except:
            print('failed opening:', imagePath)

        try:
            imageSvs.read_region((0, 0), (100,100), 0)
        except:
            print('failed reading:', imagePath)

if __name__ == '__main__':
    test_images('/')