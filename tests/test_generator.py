from src.generator import Generator

def test_generator_rectangle_generates_rectangle():
    generator = Generator()

    assert isinstance(generator, Generator)
