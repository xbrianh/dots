import numpy as np
import rtree
from contextlib import suppress
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw


WIDTH = 500
HEIGHT = 500
MAX_R = min(WIDTH, HEIGHT) / 2
MIN_DOT_AREA = 1  # 1 pixel
MAX_DOT_AREA = np.pi * MAX_R ** 2  # 250 this is the largest dot that an fit entirely within the image


def compute_area(radius):
    return np.pi * radius ** 2


def compute_radius(area):
    return np.sqrt(area / np.pi)


def size_dots_uniform_distribution(
    number_of_dots: int,
    *,
    total_area: int = 8000,
    bin_width: int | None = None,
):
    """Create dots. The area of each dot is drawn from a uniform distribution.

    the bin width of the uniform distribution used to size dots. The bin will be centered on the average dot area (total_area / number_of_dots)
    The minimum and maximum areas are average_area - binw_width / 2, average_area + bin_width / 2
    """
    average_area = total_area / number_of_dots
    if bin_width is None:
        bin_width = average_area * 1.5

    min_dot_area = average_area - bin_width / 2
    max_dot_area = average_area + bin_width / 2

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


def place_dots_square(radii, enclosing_side_length):
    rdx = rtree.index.Index()
    rdx.insert(1 + len(radii), (0, -1, WIDTH, 0))
    rdx.insert(2 + len(radii), (0, HEIGHT, WIDTH, HEIGHT + 1))
    rdx.insert(3 + len(radii), (-1, 0, 0, HEIGHT))
    rdx.insert(3 + len(radii), (WIDTH, 0, WIDTH + 1, HEIGHT))
    centers = list()
    for i, r in enumerate(radii):
        for _try in range(10000):
            x, y = np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)
            if (abs(x - WIDTH / 2) > enclosing_side_length / 2) or (abs(y - HEIGHT / 2) > enclosing_side_length / 2):
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


def place_dots_circle(radii, enclosing_radius: int = MAX_R):
    rdx = rtree.index.Index()
    centers = list()
    for i, r in enumerate(radii):
        for _try in range(100000):
            x, y = np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)
            if np.sqrt((x - WIDTH / 2)**2 + (y - HEIGHT / 2)**2) > enclosing_radius - r:
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


def compute_hull(coords, radii):
    hull_input_coords = list()
    for c, r in zip(coords, radii):
        hull_input_coords.append(c + np.array((r, r)))
        hull_input_coords.append(c + np.array((r, -r)))
        hull_input_coords.append(c + np.array((-r, -r)))
        hull_input_coords.append(c + np.array((-r, r)))
    hull = ConvexHull(hull_input_coords)
    return hull


def draw_hull(draw, hull):
    for i,j in zip(hull.vertices[:-1], hull.vertices[1:]):
        foo = np.concat([hull.points[i], hull.points[j]])
        draw.line(tuple(foo), fill="white")
    i, j = hull.vertices[-1], hull.vertices[0]
    foo = np.concat([hull.points[i], hull.points[j]])
    draw.line(tuple(foo), fill="white")


def draw_dots(coords, radii, supersampling_factor: int = 4):
    coords *= supersampling_factor
    radii *= supersampling_factor
    width = supersampling_factor * WIDTH
    height = supersampling_factor * HEIGHT
    image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(image)
    draw.rectangle([(0, 0), (width, height)], fill='black', outline='black')
    for c, r in zip(coords, radii):
        draw.circle(c, r, fill='green', outline='green')
    image = image.resize((WIDTH, HEIGHT), resample=Image.LANCZOS)
    return image


def generate_dots(
    number_of_dots: int = 40,
    total_dot_area = 24000,
    desired_hull: float = 140000.0,
    number_of_tries: int = 10,
    shape: str = "circle",
):
    match shape:
        case "circle":
            enclosing_dimension = compute_radius(desired_hull)
            place_dots = place_dots_circle
        case "square":
            enclosing_dimension = np.sqrt(desired_hull)
            place_dots = place_dots_square
        case _:
            raise ValueError("Shape must be either 'circle' or 'square'")

    areas = size_dots_uniform_distribution(number_of_dots, total_area=total_dot_area)  # , bin_width=600)
    radii = np.sqrt(areas / np.pi)
    for _ in range(number_of_tries):
        with suppress(RuntimeError):
            coords = place_dots(radii, enclosing_dimension)
            break
    else:
        raise RuntimeError("too hard to generate the dots sorry. try fewer dots or larger hull or smaller total dot area.")
    for _ in range(3):
        hull = compute_hull(coords, radii)
        factor = desired_hull / hull.volume
        coords = coords - (WIDTH / 2, HEIGHT / 2)
        coords = coords * np.sqrt(factor)
        coords = coords + (WIDTH / 2, HEIGHT / 2)
        hull = compute_hull(coords, radii)
    return coords, radii


if __name__ == '__main__':
    coords, radii = generate_dots()
    image = draw_dots(coords, radii)
    image.save("test.png")
