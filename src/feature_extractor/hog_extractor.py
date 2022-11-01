import logging

import matplotlib.pyplot as plt
import pandas as pd
from skimage import exposure
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.filters import gaussian
from skimage.io import imread
from skimage.transform import resize

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

def load_fn(fn: str, sf: float): return resize(imread(fn), (sf, sf))

class Hog_Extractor:
    def __init__(self, **kwargs):
        kwargs.setdefault('rescale_size', 200)
        kwargs.setdefault('blur_sigma', 3.0)
        kwargs.setdefault('blur_truncate', 3.0)

        [self.__setattr__(key, kwargs.get(key)) for key in kwargs]

        self._images = []

    def load_resources(self, image_resources, pool=None):
        print("loading image resources")

        if not pool:
            for fn in image_resources:
                self._images.append(
                    resize(imread(fn), (self.scale_factor, self.scale_factor))
                )
        else:
            self._images = pool.starmap(
                load_fn,
                zip(image_resources, [self.scale_factor] * len(image_resources))
            )

        print("finished loading image resources")
        return self

    def apply_filters(self):
        print("applying image filters and resizing")
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
        print("running HOG extractor")
        self.hog_features = []

        # HOG params
        # orientations = 4
        # pixels_per_cell = (16, 16)
        # cells_per_block = (2, 2)

        for image in self._processed:
            if graph_output:
                descriptor, hog_image = hog(
                    image, 
                    transform_sqrt=True,
                    # gamma correction to improve lighting response
                    visualize=True,
                    block_norm='L2'
                )

                self.hog_features.append(descriptor)
                graph_hog_image(image, hog_image)

            else:
                descriptor = hog(
                    image, 
                    # gamma correction to improve lighting response
                    transform_sqrt=True, orientations=self.hog_orientations,
                    pixels_per_cell=(self.hog_pixels_per_cells, self.hog_pixels_per_cells),
                    cells_per_block=(self.hog_cells_per_block, self.hog_cells_per_block),
                    block_norm='L2'
                )

                self.hog_features.append(descriptor)

        return self

    def extract(self): return self.hog_features


def display_extracted_hog_descriptors(image_resources: pd.Series, config: dict): 

    try:
        return (
            Hog_Extractor(**config)
                .load_resources(image_resources[3:6], )
                .apply_filters()
                .run_hog(graph_output=True)
                .extract()
        )
    except Exception as ex:
        _logger.error(f"error while extracting the HOG feature descriptors '{ex}'")
        raise


def extract_hog_descriptors(image_resources: pd.Series, pool, config: dict):
    try:
        return (
            Hog_Extractor(**config)
                .load_resources(image_resources, pool=pool)
                .apply_filters()
                .run_hog()
                .extract()
        )
    except Exception as ex:
        _logger.error(f"error while extracting the HOG feature descriptors '{ex}'")
        raise
