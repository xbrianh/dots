import numpy as np
import rtree
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw


WIDTH = 500
HEIGHT = 500
MAX_R = min(WIDTH, HEIGHT) / 2
MIN_DOT_AREA = 1  # 1 pixel
MAX_DOT_AREA = np.pi * MAX_R ** 2  # 250 this is the largest dot that an fit entirely within the image


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


def place_dots_square(radii):
    rdx = rtree.index.Index()
    rdx.insert(1 + len(radii), (0, -1, WIDTH, 0))
    rdx.insert(2 + len(radii), (0, HEIGHT, WIDTH, HEIGHT + 1))
    rdx.insert(3 + len(radii), (-1, 0, 0, HEIGHT))
    rdx.insert(3 + len(radii), (WIDTH, 0, WIDTH + 1, HEIGHT))
    centers = list()
    for i, r in enumerate(radii):
        for _try in range(10000):
            x, y = np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)
            box = np.array((x, y, x, y)) + 1.5 * np.array((-r, -r, r, r))
            hits = list(rdx.intersection(box))
            if 0 == len(hits):
                rdx.insert(i, box)
                centers.append((x, y))
                break
        else:
            raise RuntimeError(f"failed to place dot {i}")

    return np.array(centers)


def place_dots_circle(radii):
    rdx = rtree.index.Index()
    centers = list()
    for i, r in enumerate(radii):
        for _try in range(20000):
            x, y = np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)
            if np.sqrt((x - WIDTH / 2)**2 + (y - HEIGHT / 2)**2) > MAX_R - 20 - r:
                continue
            box = np.array((x, y, x, y)) + 1.5 * np.array((-r, -r, r, r))
            hits = list(rdx.intersection(box))
            if 0 == len(hits):
                rdx.insert(i, box)
                centers.append((x, y))
                break
        else:
            raise RuntimeError(f"failed to place dot {i}")

    return np.array(centers)


def main():
    areas = size_dots(10, total_area=8000)
    radii = np.sqrt(areas / np.pi)
    coords = place_dots_circle(radii)

    image = Image.new('RGBA', (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (WIDTH, HEIGHT)], fill='black', outline='black')
    draw.circle((WIDTH / 2, HEIGHT / 2), MAX_R - 20, fill='black', outline='red')
    for c, r in zip(coords, radii):
        draw.circle(c, r, fill='cyan', outline='cyan')

    hull = ConvexHull(coords)
    for i,j in zip(hull.vertices[:-1], hull.vertices[1:]):
        foo = np.concat([coords[i], coords[j]])
        draw.line(tuple(foo), fill="white")
    i, j = hull.vertices[-1], hull.vertices[0]
    foo = np.concat([coords[i], coords[j]])
    draw.line(tuple(foo), fill="white")

    image.save('test.png')


if __name__ == '__main__':
    main()
