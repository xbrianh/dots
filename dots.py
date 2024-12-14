import numpy as np
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw


WIDTH = 500
HEIGHT = 500
MIN_DOT_AREA = 1  # 1 pixel
MAX_DOT_AREA = np.pi * min(WIDTH, HEIGHT) ** 2  # 250 this is the largest dot that an fit entirely within the image


def area(radius):
    return np.pi * radius ** 2


def size_dots(
    number_of_dots: int,
    *,
    total_area: int = 8000,
    min_dot_area: int | None = None,
    max_dot_area: int | None = None,
    standard_dev: float | None = 350.0,
):
    """Create dots. The area of each dot is drawn from a uniform distribution."""
    average_area = total_area / number_of_dots
    if min_dot_area is None and max_dot_area is None:
        bin_width = np.sqrt(12) * standard_dev  # this comes from the formula for variane of a uniform distribution
        min_dot_area = average_area - bin_width / 2
        max_dot_area = average_area + bin_width / 2
    elif min_dot_area is not None:
        if min_dot_area >= average_area:
            raise ValueError('min_dot_area must be less than average_area')
        max_dot_area = average_area + (average_area - min_dot_area)
        bin_width = max_dot_area - min_dot_area
    elif max_dot_area is not None:
        if average_area <= max_dot_area:
            raise ValueError('max_dot_area must be greater than average_area')
        min_dot_area = average_area - (max_dot_area - average_area)
        bin_width = max_dot_area - min_dot_area

    if MAX_DOT_AREA < max_dot_area:
        raise ValueError("max_dot_area cannot fit into image! Try different parameters")
    if MIN_DOT_AREA > min_dot_area:
        raise ValueError(f"min_dot_area smaller than {MIN_DOT_AREA}! Try different parameters")

    areas = np.random.random(number_of_dots) * bin_width + average_area - bin_width / 2
    ratio = average_area / (np.sum(areas) / number_of_dots)
    areas = ratio * areas

    assert np.all(areas > MIN_DOT_AREA)
    assert np.all(areas < MAX_DOT_AREA)

    return areas


def main():
    areas = size_dots(80, total_area=100000)
    print(areas)
    print(np.sum(areas))

    x_coords = np.random.randint(0, WIDTH, len(areas))
    y_coords = np.random.randint(0, HEIGHT, len(areas))
    coords = np.column_stack((x_coords, y_coords))
    radii = np.sqrt(areas / np.pi)

    image = Image.new('RGBA', (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (WIDTH, HEIGHT)], fill='black', outline='black')
    for c, r in zip(coords, radii):
        draw.circle(c, r, fill='cyan', outline='cyan')

    image.save('test.png')


if __name__ == '__main__':
    main()
