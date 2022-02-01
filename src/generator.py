from enum import Enum
import random
from typing import Tuple

import numpy as np

class ImageClass(Enum):
    RECTANGLE = 0
    X = 1
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
        elif image_class == ImageClass.X:
            image = self._generate_X(image_dimension, centering_factor)
        elif image_class == ImageClass.CROSS:
            image = self._generate_cross(image_dimension, centering_factor)
        elif image_class == ImageClass.VERTICAL_BARS:
            image = self._generate_vertical_bars(image_dimension, centering_factor)
        else:
            raise NotImplementedError
        
        noise_pixels = int(noise_portion * image_dimension**2)
        for _ in noise_pixels:
            pixel = (random.randrange(0, image_dimension), random.randrange(0, image_dimension))
            image[pixel] = not image[pixel]

        if flatten:
            return image.flatten()
        else:
            return image


    def _compute_centering_delta(self,
                                 image_dimension : int,
                                 centering_factor : float,
                                 center : Tuple[int, int]) -> Tuple[int, int]:
        actual_center = (int(image_dimension / 2), int(image_dimension / 2))
        delta = (
            int((actual_center[0] - center[0]) * centering_factor),
            int((actual_center[1] - center[1]) * centering_factor),
        )
        return delta


    def _generate_rectangle(self,
                            image_dimension : int,
                            centering_factor : float,
                            *,
                            debug_corners : Tuple[Tuple[int, int], Tuple[int, int]] = None) -> np.ndarray:
        image = np.zeros(
            (image_dimension, image_dimension),
            dtype=np.bool8
        )
        corner1 = (random.randrange(0, image_dimension), random.randrange(0, image_dimension))
        corner2 = (random.randrange(0, image_dimension), random.randrange(0, image_dimension))
        if debug_corners is not None:
            corner1, corner2 = debug_corners

        center = (abs(corner1[0] - corner2[0]), abs(corner1[1] - corner2[1]))
        delta = self._compute_centering_delta(
            image_dimension, centering_factor, center
        )
        corner1 = (corner1[0] + delta[0], corner1[1] + delta[1])
        corner2 = (corner2[0] + delta[0], corner2[1] + delta[1])

        for x in range(min(corner1[0], corner2[0]), max(corner1[0], corner2[0]) + 1):
            image[x, corner1[1]] = True
            image[x, corner2[1]] = True
        for y in range(min(corner1[1], corner2[1]), max(corner1[1], corner2[1]) + 1):
            image[corner1[0], y] = True
            image[corner2[0], y] = True
        return image


    def _generate_X(self,
                    image_dimension : int,
                    centering_factor : float,
                    *,
                    debug_center : Tuple[int, int] = None) -> np.ndarray:
        image = np.zeros(
            (image_dimension, image_dimension),
            dtype=np.bool8
        )
        center = (random.randrange(0, image_dimension), random.randrange(0, image_dimension))
        if debug_center is not None:
            center = debug_center

        delta = self._compute_centering_delta(
            image_dimension, centering_factor, center
        )
        center = (center[0] + delta[0], center[1] + delta[1])
        # Iterate through \ line
        x, y = center[0] - min(center), center[1] - min(center)
        while x < image_dimension and y < image_dimension:
            image[x, y] = True
            x += 1
            y += 1
        # Iterate through / line
        distance_to_line = (center[0], image_dimension - center[1])
        x, y = center[0] - min(distance_to_line), center[1] + min(distance_to_line)
        while x < image_dimension and y >= 0:
            image[x, y] = True
            x += 1
            y -= 1
        return image
    
    def _generate_cross(self,
                        image_dimension : int,
                        centering_factor : float,
                        *,
                        debug_center : Tuple[int, int] = None) -> np.ndarray:
        image = np.zeros(
            (image_dimension, image_dimension),
            dtype=np.bool8
        )
        center = (random.randrange(0, image_dimension), random.randrange(0, image_dimension))
        if debug_center is not None:
            center = debug_center
        
        delta = self._compute_centering_delta(
            image_dimension, centering_factor, center
        )
        center = (center[0] + delta[0], center[1] + delta[1])
        
        image[center[0], :] = True
        image[:, center[1]] = True
        return image
    

    def _generate_vertical_bars(self,
                                image_dimension : int,
                                centering_factor : float = None,
                                *,
                                debug_bars : Tuple[int] = None) -> np.ndarray:
        """Generate an image of vertical bars

        IMPORTANT: centering_factor is NOT used in this method, but is still
        an argument for API-consistency.

        Args:
            image_dimension (int): Height/width of image to generate
            centering_factor (float, optional): Centering factor. NOT USED!
            debug_bars (Tuple[int], optional): DEBUG: Manually choose where to
                place bars. Only used for debugging. Defaults to None.

        Returns:
            np.ndarray: [description]
        """
        image = np.zeros(
            (image_dimension, image_dimension),
            dtype=np.bool8
        )
        number_of_bars = random.randrange(1, int(image_dimension / 2))
        bars = random.sample(range(image_dimension), number_of_bars)
        if debug_bars is not None:
            bars = debug_bars
        for x in bars:
            image[:, x] = True
        return image


    def _generate_triangle(self,
                           image_dimension : int,
                           centering_factor : float) -> np.ndarray:
        raise NotImplementedError
        image = np.zeros(
            (image_dimension, image_dimension),
            dtype=np.bool8
        )
        corners = [
            (random.randrange(0, image_dimension), random.randrange(0, image_dimension))
            for _ in range(3)
        ]
        center = (
            sum(corner[0] for corner in corners) / 3,
            sum(corner[1] for corner in corners) / 3
        )
        delta = self._compute_centering_delta(
            image_dimension, centering_factor, center
        )
        for corner in corners:
            corner = (corner[0] + delta[0], corner[1] + delta[1])
