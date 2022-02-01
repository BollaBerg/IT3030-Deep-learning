from enum import Enum
import random

import numpy as np

class ImageClass(Enum):
    RECTANGLE = 0
    TRIANGLE = 1
    CROSS = 2
    VERTICAL_BARS = 3
    HORIZONTAL_BARS = 4


class Generator:
    """Generates 2D, binary, pixel images

    """
    def __init__(self):
        initial_weights = [
            random.random() for _ in ImageClass
        ]
        self.weights = [
            float(weight) / sum(initial_weights) for weight in initial_weights
        ]

    def get_single_set(self,
                    image_dimension : int,
                    number_of_images : int,
                    noise_portion : float,
                    centering_factor : float = 0.0,
                    flatten : bool = False) -> list[np.ndarray]:
        """Generate and return a single image set (i.e. a single training set)

        Args:
            image_dimension (int): The height/width of the image. All images
                are square.
            number_of_images (int): Number of images in the returned image set.
            noise_portion (float): What portion of the image should be noise.
                If set to 1.0, the entire image will be noise. If set to 0.0,
                no noise will be generated.
            centering_factor (float, optional): How much centering should be
                applied to the image. If set to 1.0, the entire image will be 
                centered. If set to 0.0, no centering will be applied. Defaults
                to 0.0.
            flatten (bool, optional): Whether the images should be flattened.
                If True, images will be created as 1D vectors. If False, images
                will be created as 2D arrays. Defaults to False.

        Returns:
            list[np.ndarray]: List of generated images.
        """                
        distribution_of_images = [
            int(weight * number_of_images) for weight in self.weights
        ]
        # Ensure that we actually generate the correct amount of images by
        # manually changing how many images of the last class we generate
        # This shouldn't be more than a couple either way, so should not impact
        # the distribution by too much.
        distribution_of_images[-1] += (
            number_of_images - sum(distribution_of_images)
        )
        output = []

        for i, image_class in enumerate(ImageClass):
            for _ in range(distribution_of_images[i]):
                output.append(
                    self._generate_single_image(image_class,
                                                image_dimension,
                                                noise_portion,
                                                centering_factor,
                                                flatten)
                )
        
        random.shuffle(output)
        return output
    

    def _generate_single_image(self,
                               image_class : ImageClass,
                               image_dimension : int,
                               noise_portion : float,
                               centering_factor : float,
                               flatten : bool) -> np.ndarray:
        if image_class == ImageClass.RECTANGLE:
            image = self._generate_rectangle(image_dimension, centering_factor)
        else:
            raise NotImplementedError
        
        noise_pixels = int(noise_portion * image_dimension**2)
        for _ in noise_pixels:
            pixel = (random.randint(0, image_dimension), random.randint(0, image_dimension))
            image[pixel] = not image[pixel]

        if flatten:
            return image.flatten()
        else:
            return image

    
    def _generate_rectangle(self,
                            image_dimension : int,
                            centering_factor : float):
        image = np.zeros(
            (image_dimension, image_dimension),
            dtype=np.bool8
        )
        corner1 = (random.randint(0, image_dimension), random.randint(0, image_dimension))
        corner2 = (random.randint(0, image_dimension), random.randint(0, image_dimension))
        center = (abs(corner1[0] - corner2[0]), abs(corner1[1] - corner2[1]))
        actual_center = (int(image_dimension / 2), int(image_dimension / 2))
        delta = (
            (actual_center[0] - center[0]) * centering_factor,
            (actual_center[1] - center[1]) * centering_factor,
        )
        corner1 = (corner1[0] + delta[0], corner1[1] + delta[1])
        corner2 = (corner2[0] + delta[0], corner2[1] + delta[1])

        for x in range(min(corner1[0], corner2[0]), max(corner1[0], corner2[0])):
            image[x, corner1[1]] = True
            image[x, corner2[1]] = True
        for y in range(min(corner1[1], corner2[1]), max(corner1[1], corner2[1])):
            image[corner1[0], y] = True
            image[corner2[0], y] = True
        return image
