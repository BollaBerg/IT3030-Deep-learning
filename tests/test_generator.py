import numpy as np

from src.generator import Generator

def test_generator_generate_rectangle():
    """Test that Generator._generate_rectangle generates a rectangle"""
    generator = Generator()
    image = generator._generate_rectangle(
        4, 0.0,
        _debug_corners=((0, 1), (2, 3))
    )
    np.testing.assert_array_equal(
        image,
        np.array(
            [[False, True, True, True],
             [False, True, False, True],
             [False, True, True, True],
             [False, False, False, False]]
        )
    )

def test_generator_generate_rectangle_centering():
    """Test that Generator._generate_rectangle centers the rectangle"""
    generator = Generator()
    image = generator._generate_rectangle(
        4, 1.0,
        _debug_corners=((0, 0), (1, 1))
    )
    np.testing.assert_array_equal(
        image,
        np.array(
            [[False, False, False, False],
             [False, True, True, False],
             [False, True, True, False],
             [False, False, False, False],]
        )
    )



def test_generator_generate_X():
    """Test that Generator._generate_X generates an X"""
    generator = Generator()
    image = generator._generate_X(4, 0.0, _debug_center=(1, 2))
    np.testing.assert_array_equal(
        image,
        np.array(
            [[False, True, False, True],
             [False, False, True, False],
             [False, True, False, True],
             [True, False, False, False]]
        )
    )

def test_generator_generate_X_centering():
    """Test that Generator._generate_X centers the X"""
    generator = Generator()
    image = generator._generate_X(3, 1.0, _debug_center=(0, 1))
    np.testing.assert_array_equal(
        image,
        np.array(
            [[True, False, True],
             [False, True, False],
             [True, False, True]]
        )
    )


def test_generator_generate_cross():
    """Test that Generator._generate_cross generates a cross"""
    generator = Generator()
    image = generator._generate_cross(4, 0.0, _debug_center=(1, 2))
    np.testing.assert_array_equal(
        image,
        np.array(
            [[False, False, True, False],
             [True, True, True, True],
             [False, False, True, False],
             [False, False, True, False],]
        )
    )

def test_generator_generate_cross_centering():
    """Test that Generator._generate_cross centers the cross"""
    generator = Generator()
    image = generator._generate_cross(3, 1.0, _debug_center=(0, 0))
    np.testing.assert_array_equal(
        image,
        np.array(
            [[False, True, False],
             [True, True, True],
             [False, True, False]]
        )
    )

def test_generator_generate_vertical_bars():
    """Test that Generator._generate_vertical_bars generates bars"""
    generator = Generator()
    image = generator._generate_vertical_bars(5, 0.0, _debug_bars=(0, 1, 3))
    np.testing.assert_array_equal(
        image,
        np.array(
            [[True, True, False, True, False],
             [True, True, False, True, False],
             [True, True, False, True, False],
             [True, True, False, True, False],
             [True, True, False, True, False],]
        )
    )

def test_generator_generate_horizontal_bars():
    """Test that Generator._generate_horizontal_bars generate bars"""
    generator = Generator()
    image = generator._generate_horizontal_bars(5, 0.0, _debug_bars=(0, 1, 3))
    np.testing.assert_array_equal(
        image,
        np.array(
            [[True, True, True, True, True],
             [True, True, True, True, True],
             [False, False, False, False, False],
             [True, True, True, True, True],
             [False, False, False, False, False]]
        )
    )
