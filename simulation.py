import argparse
import math
import os
import random
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity

# ================== Simulation parameters ==================
pixel_size = 0.2
L = 500.0
grid_n = int(L / pixel_size)
initial_cells = 15
T = 500
save_every = 20
cells_per_step = 1
frag_attempts_per_step = 200
buffer_epsilon = 1.0
no_new_cell_steps = 0

# Material-specific parameters for schematic simulations.
# These values represent distinct qualitative material behaviours
# and are not fitted to experimental data.
MATERIAL_PARAMS = {
    "SA": {
        "p_fragment": 0.8,
        "E1": 0.3,
        "E2": 1.7,
        "eta": 0.1,
        "fragment_area_range": (0.2, 5.0),
        "retention_a": 0.001,
    },
    "UC": {
        "p_fragment": 0.5,
        "E1": 1.7,
        "E2": 0.3,
        "eta": 0.1,
        "fragment_area_range": (0.2, 3.0),
        "retention_a": 0.006,
    },
    "CC": {
        "p_fragment": 0.2,
        "E1": 1.7,
        "E2": 0.3,
        "eta": 10.0,
        "fragment_area_range": (0.2, 2.0),
        "retention_a": 1.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Schematic simulation of cell growth, material fragmentation, uptake and fusion."
    )
    parser.add_argument(
        "--material", choices=MATERIAL_PARAMS, default="SA",
        help="Material condition: SA, UC or CC (default: SA)."
    )
    parser.add_argument(
        "--steps", type=int, default=T,
        help=f"Number of simulation steps (default: {T})."
    )
    parser.add_argument(
        "--seed", type=int, default=2,
        help="Random seed for reproducibility (default: 2)."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Root directory for simulation outputs (default: results/)."
    )
    return parser.parse_args()


args = parse_args()
MATERIAL = args.material
params = MATERIAL_PARAMS[MATERIAL]

P_FRAGMENT = params["p_fragment"]
E1 = params["E1"]
E2 = params["E2"]
ETA = params["eta"]
FRAGMENT_AREA_RANGE = params["fragment_area_range"]
RETENTION_A = params["retention_a"]

T = args.steps
np.random.seed(args.seed)
random.seed(args.seed)

save_dir = args.output_dir / MATERIAL
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Material: {MATERIAL}")
print(f"Parameters: {params}")
print(f"Output: {save_dir.resolve()}")

# ================== Helper functions ==================
def get_boundary(mask):
    kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    grad = convolve(mask.astype(float), kernel, mode='constant')
    boundary = grad != 0
    ys, xs = np.where(boundary)
    return list(zip(xs, ys))

def make_connected(shape, epsilon=buffer_epsilon, smooth_d=0.5, smooth_iters=3):
    shape = shape.buffer(epsilon).buffer(-epsilon)
    for _ in range(smooth_iters):
        shape = shape.buffer(smooth_d).buffer(-smooth_d)
    return shape

def consume_probability(a):
    return max(0.1, min(0.8, 0.6 - 0.03*(a-5)))

def can_place_cell(pos, new_radius, cells, max_overlap_ratio=0.1):
    new_cell_shape = Point(pos).buffer(new_radius)
    for c in cells:
        intersection_area = c.shape.intersection(new_cell_shape).area
        if intersection_area / new_cell_shape.area > max_overlap_ratio:
            return False
    return True

def remove_contained_fragments(cells, fragments):
    to_remove_ids = set()
    for c in cells:
        cell_geom = c.shape.buffer(1e-6)
        for f in fragments:
            frag_geom = Point(f.pos).buffer(f.radius)
            if cell_geom.covers(frag_geom):
                to_remove_ids.add(f.id)
    if to_remove_ids:
        fragments = [f for f in fragments if f.id not in to_remove_ids]
    return fragments


class Cell:
    _id = 1
    def __init__(self, pos, area, T=500):
        self.id = Cell._id
        Cell._id += 1
        self.area0 = area
        self.area = area
        # Cumulative area of all material fragments phagocytosed by this cell.
        # This quantity is conserved when cells fuse.
        self.Frag_area = 0.0
        self.radius = math.sqrt(area / math.pi)
        self.base_radius = self.radius
        self.pos = np.array(pos, float)
        self.shape = Point(self.pos).buffer(self.radius)
        self.nuclei = [np.array(self.pos)]
        self.n_fusions = 0
        self.last_fusion_step_material = -10
        self.last_consume_step = -10
        self.fusion_prob = 0.2
        self.E1 = E1
        self.E2 = E2
        self.eta = ETA
        self.sigma0 = 1
        self.strain_depth = 0.0
        self.T = T
        self.timestep = 0
        self.fusion_count_step = 0
    def update_area(self, frag, current_step):
        # Record the total phagocytosed material area independently of
        # the geometric area added to the cell.
        self.Frag_area += frag.area
        frag_poly = Point(frag.pos).buffer(frag.radius)
        new_shape = unary_union([self.shape, frag_poly]).buffer(0)
        self.shape = make_connected(new_shape)
        self.area = self.shape.area
        self.pos = np.array(self.shape.centroid.coords[0])
        self.radius = math.sqrt(self.area / math.pi)
        self.last_fusion_step_material = current_step
        self.last_consume_step = current_step
        self.fusion_prob = 0.0
    def update_fusion_prob(self, current_step):
        if self.area > 450 * len(self.nuclei) * np.exp(-0.3 * len(self.nuclei)) or current_step - self.last_consume_step <= 3:
            self.fusion_prob = 0.0
            return
        amin = 20
        amax = 450 * len(self.nuclei) * np.exp(-0.3 * len(self.nuclei))
        if amax <= amin:
            rate = 0.05
        else:
            frac = (self.area - amin) / (amax - amin)
            frac = max(0.0, min(1.0, frac))
            rate = 0.2 - 0.1 * frac
        self.fusion_prob = min(1.0, self.fusion_prob + rate)
    def local_strain_growth(self, t, k_growth=0.3, max_overlap_ratio=0.1):
        a = (self.E1 * self.E2) / (self.eta * (self.E1 + self.E2))
        mu = self.E2 / (self.eta * (self.E1 + self.E2)) * np.exp(-a * t)
        mu_factor = max(mu, 1)
        self.strain_depth = (self.sigma0 / self.E1) * (1 - np.exp(-a * t))
        strain_increment = k_growth * self.strain_depth * mu_factor * np.random.uniform(0.8, 1.2)
        max_radius = self.base_radius + self.sigma0 / self.E1
        max_increment = max_radius - self.radius
        if max_increment <= 0:
            return
        strain_increment = min(strain_increment, max_increment)
        target_radius = self.radius + strain_increment
        target_area = math.pi * target_radius**2
        if target_area <= self.area:
            return
        factor = math.sqrt(target_area / self.area)
        shape = self.shape
        if isinstance(shape, MultiPolygon):
            shape = unary_union(shape)
        if not shape.is_valid:
            return
        cx, cy = shape.centroid.x, shape.centroid.y
        new_shape = affinity.scale(shape, xfact=factor, yfact=factor, origin=(cx, cy))
        if cells is not None:
                for c_other in cells:
                    if c_other.id == self.id:
                        continue
                    intersection_area = new_shape.intersection(c_other.shape).area
                    if intersection_area / new_shape.area > max_overlap_ratio:
                        return
        if new_shape.is_valid:
            self.shape = new_shape
            self.area = new_shape.area
            self.radius = math.sqrt(self.area / math.pi)

def fuse_cells(c, c2, t=None):
    try:

        if c.fusion_count_step >= 2:
            return None

        force_merge = False
        for nuc in c.nuclei:
            if c2.shape.buffer(1e-6).covers(Point(nuc)):
                force_merge = True
                break
        if not force_merge:
            for nuc in c2.nuclei:
                if c.shape.buffer(1e-6).covers(Point(nuc)):
                    force_merge = True
                    break
        if force_merge:
            buff = 0.5
            new_shape = unary_union([
                c.shape.buffer(buff),
                c2.shape.buffer(buff)
            ]).buffer(-buff)
            if new_shape.is_empty or not new_shape.is_valid or new_shape.geom_type == "MultiPolygon":
                new_shape = unary_union([
                    c.shape.buffer(2.0),
                    c2.shape.buffer(2.0)
                ]).buffer(-2.0)
                if new_shape.is_empty or not new_shape.is_valid or new_shape.geom_type == "MultiPolygon":
                    return None
            c.shape = new_shape
            c.area = c.shape.area
            c.pos = np.array(c.shape.centroid.coords[0])
            c.radius = math.sqrt(c.area / math.pi)
            c.n_fusions += c2.n_fusions + 1
            c.Frag_area += c2.Frag_area
            c.nuclei.extend(c2.nuclei)
            c.fusion_prob = 0.0
            c.last_consume_step = t
            c.last_fusion_step_material = max(
                c.last_fusion_step_material,
                c2.last_fusion_step_material
            )
            c.fusion_prob = min(c.fusion_prob, c2.fusion_prob)
            c.fusion_count_step += 1
            return c, c2

        try:
            sample_n = 50
            probe_r = 4.0
            def boundary_coverage_ratio(shapeA, shapeB):
                boundary = shapeA.boundary
                length = boundary.length
                if length == 0:
                    return 0
                step = length / sample_n
                hit = 0
                for i in range(sample_n):
                    d = (i + 0.5) * step
                    pt = boundary.interpolate(d)
                    if pt.distance(shapeB) < probe_r:
                        hit += 1
                return hit / sample_n
            ratio1 = boundary_coverage_ratio(c.shape, c2.shape)
            ratio2 = boundary_coverage_ratio(c2.shape, c.shape)
            if ratio1 >= 0.4 or ratio2 >= 0.4:
                buff = 0.5
                new_shape = unary_union([
                    c.shape.buffer(buff),
                    c2.shape.buffer(buff)
                ]).buffer(-buff)
                if new_shape.is_empty or not new_shape.is_valid or new_shape.geom_type == "MultiPolygon":
                            new_shape = unary_union([
                                c.shape.buffer(2.0),
                                c2.shape.buffer(2.0)
                            ]).buffer(-2.0)
                            if new_shape.is_empty or not new_shape.is_valid:
                                return None
                c.shape = new_shape
                c.area = c.shape.area
                c.pos = np.array(c.shape.centroid.coords[0])
                c.radius = math.sqrt(c.area / math.pi)
                c.n_fusions += c2.n_fusions + 1
                c.nuclei.extend(c2.nuclei)
                c.fusion_prob = 0.0
                c.last_consume_step = t
                c.last_fusion_step_material = max(
                    c.last_fusion_step_material,
                    c2.last_fusion_step_material
                )
                c.fusion_prob = min(c.fusion_prob, c2.fusion_prob)
                c.fusion_count_step += 1
                return c, c2
        except Exception as e:
            pass

        if c.area > 450 * len(c.nuclei) * np.exp(-0.3 * len(c.nuclei)) or c2.area > 450 * len(c2.nuclei) * np.exp(-0.3 * len(c2.nuclei)):
            return None

        buff = 0.5
        new_shape = unary_union([
            c.shape.buffer(buff),
            c2.shape.buffer(buff)
        ]).buffer(-buff)
        if new_shape.is_empty or not new_shape.is_valid or new_shape.geom_type == "MultiPolygon":
            return None
        c.shape = new_shape
        c.area = c.shape.area
        c.pos = np.array(c.shape.centroid.coords[0])
        c.radius = math.sqrt(c.area / math.pi)
        c.n_fusions += c2.n_fusions + 1
        c.nuclei.extend(c2.nuclei)
        c.fusion_prob = 0.0
        c.last_consume_step = t
        c.last_fusion_step_material = max(
            c.last_fusion_step_material,
            c2.last_fusion_step_material
        )
        c.fusion_prob = min(c.fusion_prob, c2.fusion_prob)
        c.fusion_count_step += 1
        return c, c2
    except Exception as e:
        return None

class Fragment:
    _id = 1
    def __init__(self, pos, area):
        self.id = Fragment._id
        Fragment._id += 1
        self.pos = np.array(pos, float)
        self.area = float(area)
        self.radius = math.sqrt(self.area/math.pi)


a, b = 450, 300
cx, cy = L, 0
yy, xx = np.meshgrid(np.arange(grid_n), np.arange(grid_n), indexing='ij')
xx_real = xx * pixel_size
yy_real = yy * pixel_size
material_mask = ((xx_real - cx)**2 / a**2 + (yy_real - cy)**2 / b**2) <= 1

num_holes = 30
contact_boundary = list(set(get_boundary(material_mask)) & set(get_boundary(~material_mask)))
for _ in range(num_holes):
    if not contact_boundary:
        break
    x0, y0 = contact_boundary[np.random.randint(len(contact_boundary))]
    rx = np.random.randint(20, 60)
    ry = np.random.randint(20, 40)
    theta = np.random.rand() * 2 * np.pi
    x_shift = xx - x0
    y_shift = yy - y0
    x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
    y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)
    mask = (x_rot**2)/rx**2 + (y_rot**2)/ry**2 <= 1
    noise = np.random.randn(*mask.shape) * 0.3
    mask = mask.astype(float) + noise
    mask = gaussian_filter(mask, sigma=2)
    mask = mask > 0.5
    material_mask[mask] = False

void_mask = ~material_mask
retention_material_mask = material_mask.copy()
material_boundary = set(get_boundary(material_mask))
void_boundary = set(get_boundary(void_mask))
contact_boundary = list(void_boundary & material_boundary)


cells = []
tries = 0
while len(cells) < initial_cells and tries < 2000:
    pos_idx = random.choice(contact_boundary)
    pos = np.array([(pos_idx[0]+0.5)*pixel_size, (pos_idx[1]+0.5)*pixel_size])
    if can_place_cell(pos, math.sqrt(100/math.pi), cells):
        cells.append(Cell(pos, 100.0))
    tries += 1

fragments = []


def draw_snapshot(fname, cells, fragments, title=""):
    fig, ax = plt.subplots(figsize=(6,6))
    img = np.ones((grid_n, grid_n,3), float)
    img[material_mask] = [0.8,0.8,0.8]
    ax.imshow(img, extent=[0,L,0,L], origin='lower')
    for f in fragments:
        circ = plt.Circle(f.pos, radius=f.radius, color="#cccccc", alpha=0.5)
        ax.add_patch(circ)
    for c in cells:
        if isinstance(c.shape, MultiPolygon):
            for poly in c.shape.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, color="royalblue", alpha=0.5, edgecolor="darkblue", linewidth=1.5)
        elif isinstance(c.shape, Polygon):
            x, y = c.shape.exterior.xy
            ax.fill(x, y, color="royalblue", alpha=0.5, edgecolor="darkblue", linewidth=1.5)
        for nuc in c.nuclei:
            circ_n = plt.Circle(nuc, radius=1.2, color="red")
            ax.add_patch(circ_n)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close(fig)



def draw_retention_snapshot(fname, cells, title=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    img = np.ones((grid_n, grid_n, 3), float)
    img[retention_material_mask] = [0.8, 0.8, 0.8]
    ax.imshow(img, extent=[0, L, 0, L], origin='lower')

    for c in cells:
        if isinstance(c.shape, MultiPolygon):
            for poly in c.shape.geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, color="royalblue", alpha=0.5,
                        edgecolor="darkblue", linewidth=1.5)
        elif isinstance(c.shape, Polygon):
            x, y = c.shape.exterior.xy
            ax.fill(x, y, color="royalblue", alpha=0.5,
                    edgecolor="darkblue", linewidth=1.5)

        for nuc in c.nuclei:
            circ_n = plt.Circle(nuc, radius=1.2, color="red")
            ax.add_patch(circ_n)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close(fig)


def run_intracellular_retention(cells, n_steps=100, save_every=20):
    retention_states = []

    for c in cells:
        retention_states.append({
            "cell": c,
            "remaining_frag_area": c.Frag_area,
        })

    draw_retention_snapshot(
        save_dir / "retention_t0.png",
        cells,
        title=f"{MATERIAL} intracellular retention, t=0"
    )

    retention_frames = [
        imageio.imread(save_dir / "retention_t0.png")
    ]

    for t_ret in range(1, n_steps + 1):
        for state in retention_states:
            c = state["cell"]
            previous_area = c.area
            initial_frag_area = c.Frag_area

            remaining_frag_area = (
                initial_frag_area *
                np.exp(-RETENTION_A * t_ret)
            )
            state["remaining_frag_area"] = remaining_frag_area

            p_area = (
                remaining_frag_area / previous_area
                if previous_area > 0 else 0.0
            )
            p_area = max(0.0, min(1.0, p_area))

            factor = (
                1.0
                - p_area *
                (
                    np.exp(-RETENTION_A * (t_ret - 1))
                    - np.exp(-RETENTION_A * t_ret)
                )
            )
            factor = max(0.0, min(1.0, factor))

            current_radius = c.radius
            current_shape = c.shape

            if current_shape.is_empty or not current_shape.is_valid:
                continue

            if current_radius <= 0:
                continue

            current_centroid = current_shape.centroid
            cx, cy = current_centroid.x, current_centroid.y

            target_radius = current_radius * factor
            scale_factor = target_radius / current_radius

            new_shape = affinity.scale(
                current_shape,
                xfact=scale_factor,
                yfact=scale_factor,
                origin=(cx, cy)
            )

            if new_shape.is_valid and not new_shape.is_empty:
                c.shape = new_shape
                c.area = new_shape.area
                c.pos = np.array(new_shape.centroid.coords[0])
                c.radius = target_radius

        if t_ret % save_every == 0:
            fname = save_dir / f"retention_t{t_ret}.png"
            draw_retention_snapshot(
                fname,
                cells,
                title=f"{MATERIAL} intracellular retention, t={t_ret}"
            )
            retention_frames.append(imageio.imread(fname))

    out_gif = save_dir / "retention.gif"
    imageio.mimsave(out_gif, retention_frames, fps=5)

    print(f"Retention a: {RETENTION_A}")
    print(f"Intracellular retention calculation completed: {n_steps} steps.")
    print(f"Retention output: {out_gif.resolve()}")


draw_snapshot(os.path.join(save_dir,"snapshot_t0.png"), cells, fragments, "t=0")
frames = [imageio.imread(os.path.join(save_dir,"snapshot_t0.png"))]

for t in range(1, T+1):
    for c in cells:
        c.fusion_count_step = 0

    prev_ids = set(c.id for c in cells)
    prev_n = len(cells)

    material_boundary = set(get_boundary(material_mask))
    void_boundary = set(get_boundary(~material_mask))
    contact_boundary = list(void_boundary & material_boundary)
    # Fragment generation
    if contact_boundary:
        candidate_points = random.sample(contact_boundary, min(frag_attempts_per_step, len(contact_boundary)))
        for frag_idx in candidate_points:
            if np.random.rand() < P_FRAGMENT:
                frag_pos = np.array([(frag_idx[0]+0.5)*pixel_size, (frag_idx[1]+0.5)*pixel_size])
                a_frag = random.uniform(*FRAGMENT_AREA_RANGE)
                fragments.append(Fragment(frag_pos, a_frag))
                radius_pix = int(math.sqrt(a_frag/math.pi)//pixel_size) + 1
                ci, cj = frag_idx[1], frag_idx[0]
                for di in range(-radius_pix, radius_pix+1):
                    for dj in range(-radius_pix, radius_pix+1):
                        ni, nj = ci+di, cj+dj
                        if 0 <= ni < grid_n and 0 <= nj < grid_n:
                            if np.linalg.norm(np.array([ni,nj]) - np.array([ci,cj])) * pixel_size <= math.sqrt(a_frag/math.pi):
                                material_mask[ni,nj] = False

    new_cell_generated = False
    allow_generation = (t < 200)
    if allow_generation and t % 1 == 0:
        radius_new = math.sqrt(100.0 / math.pi)
        for _ in range(2):
            placed = False
            if not contact_boundary:
                break
            for _try in range(5):
                pos_idx = random.choice(contact_boundary)
                jitter = (np.random.rand(2) - 0.5) * pixel_size * 0.5
                pos = np.array([
                    (pos_idx[0] + 0.5) * pixel_size,
                    (pos_idx[1] + 0.5) * pixel_size
                ]) + jitter
                if can_place_cell(pos, radius_new, cells, max_overlap_ratio=0.05):
                    cells.append(Cell(pos, 100.0))
                    placed = True
                    new_cell_generated = True
                    break
    if not new_cell_generated:
        no_new_cell_steps += 1
    else:
        no_new_cell_steps = 0
    if allow_generation and no_new_cell_steps >= 5:
        for _ in range(3):
            if contact_boundary:
                shuffled = contact_boundary.copy()
                random.shuffle(shuffled)
                step = max(1, len(shuffled) // 200)
                sampled_boundary = shuffled[::step]
                radius_new = math.sqrt(100.0 / math.pi)
                for attempt in range(5):
                    best_pos = None
                    best_dist = -1
                    for _ in range(50):
                        pos_idx = random.choice(sampled_boundary)
                        pos = np.array([
                            (pos_idx[0] + 0.5) * pixel_size,
                            (pos_idx[1] + 0.5) * pixel_size
                        ])
                        if cells:
                            dmin = min(np.linalg.norm(pos - c.pos) for c in cells)
                        else:
                            dmin = 999
                        if dmin > best_dist:
                            best_dist = dmin
                            best_pos = pos
                    if best_pos is not None:
                        if can_place_cell(best_pos, radius_new, cells, max_overlap_ratio=0.6):
                            cells.append(Cell(best_pos, 100.0))
                            no_new_cell_steps = 0
                            break
# Remove fully contained fragments
    to_remove = []
    for c in cells:
        cell_geom = c.shape.buffer(1e-6)
        for f in fragments:
            frag_geom = Point(f.pos).buffer(f.radius)
            if cell_geom.covers(frag_geom):
                to_remove.append(f.id)
    if to_remove:
        fragments = [f for f in fragments if f.id not in to_remove]
    # Fragment uptake
    n_before = len(cells)
    consumed_ids = set()
    for c in cells:
        if fragments:
            dists = [np.linalg.norm(c.pos - f.pos) for f in fragments]
            nearest_idx = int(np.argmin(dists))
            nearest_f = fragments[nearest_idx]
            prob_consume = consume_probability(nearest_f.area)
            if dists[nearest_idx] <= c.radius + nearest_f.radius + 1.0 and np.random.rand() < prob_consume:
                c.update_area(nearest_f, t)
                consumed_ids.add(nearest_f.id)
    if consumed_ids:
        fragments = [f for f in fragments if f.id not in consumed_ids]
    # Cell fusion
    n_before = len(cells)
    i = 0
    while i < len(cells):
        c = cells[i]
        c.update_fusion_prob(t)
        j = i + 1
        merged = False
        while j < len(cells):
            c2 = cells[j]
            if abs(c.pos[0] - c2.pos[0]) + abs(c.pos[1] - c2.pos[1]) > (c.radius + c2.radius + 50):
                j += 1
                continue
            inter_area = c.shape.intersection(c2.shape).area
            overlap_ratio = inter_area / min(c.shape.area, c2.shape.area)
            nucleus_inside = False
            for nuc in c.nuclei:
                if c2.shape.contains(Point(nuc)):
                    nucleus_inside = True
                    break
            if not nucleus_inside:
                for nuc in c2.nuclei:
                    if c.shape.contains(Point(nuc)):
                        nucleus_inside = True
                        break
            contact = (c.shape.distance(c2.shape) <= 1.0)
        # Forced fusion
            if overlap_ratio > 0.4 or nucleus_inside:
                old_id1, old_id2 = c.id, c2.id
                fused_cell_tuple = fuse_cells(c, c2, t)
                if fused_cell_tuple:
                    c = fused_cell_tuple[0]
                    cells[i] = c
                    cells.pop(j)
                    merged = True
                    print(f"[Force] t={t}: {old_id1} + {old_id2} -> {c.id}")
                    continue
        # Probabilistic fusion
            elif contact:
                if np.random.rand() < c.fusion_prob:
                    old_id1, old_id2 = c.id, c2.id
                    fused_cell_tuple = fuse_cells(c, c2, t)
                    if fused_cell_tuple:
                        c = fused_cell_tuple[0]
                        cells[i] = c
                        cells.pop(j)
                        merged = True
                        print(f"[Prob] t={t}: {old_id1} + {old_id2} -> {c.id}")
                        continue
            j += 1
        if not merged:
            i += 1
    # Stress-relaxation-driven growth
    n_before = len(cells)
    for c in cells:
        c.local_strain_growth(t, k_growth=0.3)
    # Save snapshot
    if t % save_every == 0:
        fname = save_dir / f"snapshot_t{t}.png"
        draw_snapshot(fname, cells, fragments, title=f"t={t}")
        frames.append(imageio.imread(fname))

out_gif = save_dir / "simulation.gif"
imageio.mimsave(out_gif, frames, fps=5)

# ================== Post-simulation intracellular retention ==================
run_intracellular_retention(cells, n_steps=100, save_every=20)
