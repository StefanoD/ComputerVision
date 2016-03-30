from skimage.data import imread
import numpy as np

class Img:
    @staticmethod
    def load_image(path, as_grey = False, to_float = True):
        # Load image
        image = imread(path, as_grey)

        if to_float:
            # Convert to floating point matrix
            image = image.astype(np.float32)

        return image