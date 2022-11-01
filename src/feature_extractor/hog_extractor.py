import logging

import matplotlib.pyplot as plt
import pandas as pd
from skimage import exposure
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import resize
from skimage.filters import gaussian

_logger = logging.getLogger(__name__)


def graph_hog_image(image, hog_image):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    
    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()

    return


class Hog_Extractor:
    def __init__(self, **kwargs):
        kwargs.setdefault('rescale_size', 200)
        kwargs.setdefault('blur_sigma', 3.0)
        kwargs.setdefault('blur_truncate', 3.0)

        [self.__setattr__(key, kwargs.get(key)) for key in kwargs]

    def load_resources(self, image_resources):
        self._images = [
            resize(imread(fn), (self.scale_factor, self.scale_factor)) for fn in image_resources
        ]

        return self

    def apply_filters(self):
        self._processed = [
            # Gaussian blurs the edges in road elements
            gaussian(
                # use grayscale to aid filtering
                # http://www.cs.columbia.edu/~vondrick/ihog/color/
                rgb2gray(image),
                sigma=(self.blur_sigma, self.blur_sigma), 
                truncate=self.blur_truncate, channel_axis=2
            ) for image in self._images
        ]

        return self
        

    def run_hog(self, graph_output: bool=False):
        self.hog_features = []

        for image in self._processed:
            descriptor, hog_image = hog(
                image, 
                # gamma correction to improve lighting response
                transform_sqrt=True,
                visualize=graph_output
            )

            self.hog_features.append(descriptor)

            if graph_output: graph_hog_image(image, hog_image)

        return self

    def extract(self): return self.hog_features


def extract_hog_descriptors(image_resources: pd.Series, config: dict):  
    try:
        return (
            Hog_Extractor(**config)
                .load_resources(image_resources[3:6])
                .apply_filters()
                .run_hog(graph_output=True)
                .extract()
        )
    except Exception as ex:
        _logger.error(f"error while extracting the HOG feature descriptors '{ex}'")
        raise
