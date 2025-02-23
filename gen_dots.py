#!/usr/bin/env python
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import dots


NUMBER_OF_PAIRS = 11
OUTPUT_FILE_TEMPLATE = "foo/dots_{image_number}.png"

DOT_PARMS_A = {
    "number_of_dots": 10,
    "total_dot_area": 24000,
    "desired_hull" : 140000.0,
    "number_of_tries": 10,
    "shape": "circle",
}

DOT_PARMS_B = {
    "number_of_dots": 45,
    "total_dot_area": 24000,
    "desired_hull" : 140000.0,
    "number_of_tries": 10,
    "shape": "circle",
}


def generate_and_save_dots(image_number: int, **dot_parms):
    coords, radii = dots.generate_dots(**dot_parms)
    image = dots.draw_dots(coords, radii)
    image.save(OUTPUT_FILE_TEMPLATE.format(image_number=image_number))


def gen_dots(number_of_pairs: int):
    image_number = 1
    futures = list()
    with ProcessPoolExecutor() as e:
        for _ in range(number_of_pairs):
            futures.append(e.submit(generate_and_save_dots, image_number, **DOT_PARMS_A))
            image_number += 1
            futures.append(e.submit(generate_and_save_dots, image_number, **DOT_PARMS_B))
            image_number += 1
            for f in as_completed(futures):
                f.result()


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_FILE_TEMPLATE), exist_ok=True)
    gen_dots(NUMBER_OF_PAIRS)
