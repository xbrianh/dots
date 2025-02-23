#!/usr/bin/env python
import os
from concurrent.futures import ProcessPoolExecutor

import dots


NUMBER_OF_FILES = 11
OUTPUT_FILE_TEMPLATE = "foo/{image_number:03}.png"
DOT_PARMS = {
    "number_of_dots": 40,
    "total_dot_area": 24000,
    "desired_hull" : 140000.0,
    "number_of_tries": 10,
    "shape": "circle",
}


def generate_and_save_dots(image_number: int):
    coords, radii = dots.generate_dots(**DOT_PARMS)
    image = dots.draw_dots(coords, radii)
    image.save(OUTPUT_FILE_TEMPLATE.format(image_number=image_number))


def gen_dots(number_of_files: int):
    with ProcessPoolExecutor() as e:
        e.map(generate_and_save_dots, [image_number for image_number in range(number_of_files)])


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE_TEMPLATE), exist_ok=True)
    gen_dots(NUMBER_OF_FILES)
