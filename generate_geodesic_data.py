import taichi as ti
import numpy as np
import math
import os
import random
from model.kerr_schild import calculate_g, calculate_dgdx, calculate_dgdy, calculate_dgdz
import torch
import yaml
PI = 3.14159265
# Canvas
aspect_ratio = 16 / 9
image_width = 160
steps = 80000
N = 200
image_height = int(image_width / aspect_ratio)
ti.init(arch=ti.gpu)


position_fields = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height, N))
speed_fields = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height, N))
acc_fields = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height, N))
tau_fields = ti.Vector.field(1, dtype=ti.f32, shape=(image_width, image_height, N))
selected_ids = ti.Vector.field(N, dtype=ti.i32, shape=(image_width, image_height))

record_position_fields = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height, steps // 10))
record_speed_fields = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height, steps // 10))
record_acc_fields = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height, steps // 10))
record_tau_fields = ti.Vector.field(1, dtype=ti.f32, shape=(image_width, image_height, steps // 10))

def reset_fields():
    position_fields.fill(0)
    speed_fields.fill(0)
    acc_fields.fill(0)
    tau_fields.fill(0)
    record_position_fields.fill(0)
    record_speed_fields.fill(0)
    record_acc_fields.fill(0)
    record_tau_fields.fill(0)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


@ti.func
def outer_product3(v: ti.template()):
    result = ti.Matrix.zero(ti.f32, 3, 3)
    for i in ti.static(range(3)):
        for j in ti.static(range(3)):
            result[i, j] = v[i] * v[j]
    return result

@ti.func
def outer_product4(v: ti.template()):
    result = ti.Matrix.zero(ti.f32, 4, 4)
    for i in ti.static(range(4)):
        for j in ti.static(range(4)):
            result[i, j] = v[i] * v[j]
    return result


@ti.data_oriented
class VirtualCamera:
    def __init__(self, image_size=(1920, 1080)):
        self.christoffles = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(*image_size, 3))
        self.dg_down = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(*image_size, 4))


    @ti.func
    def ray_step_func_kerr(self, i_index, j_index, position, direction, center_position, bh_mass, bh_position, bh_spin,
                           max_radius, min_radius, near_bh_distance, steps):
        x, y, z = position
        dtau = 0.01
        center_position = ti.Vector([center_position[0], center_position[1], center_position[2]])
        vx, vy, vz = direction.normalized()
        farest_step = steps - 1
        tau = 0.0

        ## calculate initial v^t by solving g_{\mu \nu} v^\mu v^\nu = 0
        speed = ti.Vector([vx, vy, vz])
        g_down = ti.Matrix([[-1., 0, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]])
        for i in range(bh_mass.shape[0]):
            x_bh, y_bh, z_bh = bh_position[i]
            delta_x = x - x_bh
            delta_y = y - y_bh
            delta_z = z - z_bh
            g_down += calculate_g(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
        ## solve g_{\mu \nu} v^\mu v^\nu = 0 to obtain vt
        l_vv = outer_product3(speed)
        c = (g_down[1:, 1:] * l_vv).sum()
        b = 2 * (g_down[0, 1:] * speed).sum()
        a = g_down[0, 0]
        vt = 0.0
        if a == 0:
            vt += -c / b
        else:
            vt += -b / (2 * a)
            if b ** 2 - 4 * a * c >= 0:
                vt += ti.sqrt(b ** 2 - 4 * a * c) / (2 * a)
        in_near_bh_distance_taus = 0.0
        for step in range(steps):

            self.christoffles[i_index, j_index, 0].fill(0.0)
            self.christoffles[i_index, j_index, 1].fill(0.0)
            self.christoffles[i_index, j_index, 2].fill(0.0)
            self.dg_down[i_index, j_index, 0].fill(0.0)
            self.dg_down[i_index, j_index, 1].fill(0.0)
            self.dg_down[i_index, j_index, 2].fill(0.0)
            self.dg_down[i_index, j_index, 3].fill(0.0)
            position = ti.Vector([x, y, z])
            speed = ti.Vector([vx, vy, vz]).normalized()
            vx, vy, vz = speed
            r = (position - center_position).norm()
            if r > max_radius:
                farest_step = step
                break
            if in_near_bh_distance_taus > near_bh_distance * 2 * PI:
                farest_step = step
                break
            # check if the ray falls into the black hole
            fall_into_bh = False
            for i in range(bh_position.shape[0]):
                x_bh, y_bh, z_bh = bh_position[i]
                distance = ti.sqrt((x - x_bh) ** 2 + (y - y_bh) ** 2 + (z - z_bh) ** 2)
                if distance < near_bh_distance:
                    in_near_bh_distance_taus += dtau
                if distance < min_radius:
                    farest_step = step
                    fall_into_bh = True
                    break
            if fall_into_bh:
                break
            g_down = ti.Matrix([[-1., 0, 0, 0], [0, 1, 0, 0],
                                [0, 0, 1, 0], [0, 0, 0, 1]])
            for i in range(bh_mass.shape[0]):
                x_bh, y_bh, z_bh = bh_position[i]
                delta_x = x - x_bh
                delta_y = y - y_bh
                delta_z = z - z_bh
                g_down += calculate_g(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
                self.dg_down[i_index, j_index, 1] += calculate_dgdx(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
                self.dg_down[i_index, j_index, 2] += calculate_dgdy(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
                self.dg_down[i_index, j_index, 3] += calculate_dgdz(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
            ## solve g_{\mu \nu} v^\mu v^\nu = 0 to obtain vt
            l_vv = outer_product3(speed)
            c = (g_down[1:, 1:] * l_vv).sum()
            b = 2 * (g_down[0, 1:] * speed).sum()
            a = g_down[0, 0]
            vt = 0.0
            if a == 0:
                vt += -c / b
            else:
                vt += -b / (2 * a)
                if b ** 2 - 4 * a * c >= 0:
                    vt += ti.sqrt(b ** 2 - 4 * a * c) / (2 * a)
            g_up = g_down.inverse()
            if g_up.sum() != g_up.sum():  # nan
                g_up = (g_down + 0.0001 * ti.Matrix.identity(ti.f32, 4)).inverse()
            multiplied_christoffel = ti.Vector([0., 0., 0., 0.])
            l_tvv = outer_product4(ti.Vector([vt, vx, vy, vz]))
            for mu in ti.static(range(4)):
                christoffel = ti.Matrix.zero(ti.f32, 4, 4)
                for a in ti.static(range(4)):
                    for b in ti.static(range(4)):
                        for sigma in ti.static(range(4)):
                            christoffel[a, b] += 1 / 2 * g_up[mu, sigma] * (
                                    self.dg_down[i_index, j_index, a][b, sigma] + self.dg_down[i_index, j_index, b][
                                a, sigma] - self.dg_down[i_index, j_index, sigma][a, b])
                multiplied_christoffel[mu] = (christoffel * l_tvv).sum()
            tau += dtau
            vt -= multiplied_christoffel[0] * dtau
            vx -= multiplied_christoffel[1] * dtau
            vy -= multiplied_christoffel[2] * dtau
            vz -= multiplied_christoffel[3] * dtau
            v_norm = ti.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
            vx /= v_norm
            vy /= v_norm
            vz /= v_norm
            x += vx * dtau
            y += vy * dtau
            z += vz * dtau
            acc = ti.Vector([-multiplied_christoffel[1], -multiplied_christoffel[2], -multiplied_christoffel[3]])
            if step % 10 == 0:
                record_position_fields[i_index, j_index, step // 10] = position
                record_speed_fields[i_index, j_index, step // 10] = speed
                record_acc_fields[i_index, j_index, step // 10] = acc
                record_tau_fields[i_index, j_index, step // 10] = tau

        record_step_size = (farest_step + 1) // (N * 10)
        for record_step in range(N):
            step_idx = record_step_size * record_step
            position_fields[i_index, j_index, record_step] = record_position_fields[i_index, j_index, step_idx]
            speed_fields[i_index, j_index, record_step] = record_speed_fields[i_index, j_index, step_idx]
            acc_fields[i_index, j_index, record_step] = record_acc_fields[i_index, j_index, step_idx]
            tau_fields[i_index, j_index, record_step] = record_tau_fields[i_index, j_index, step_idx]



@ti.kernel
def calculate(position: ti.template(), direction: ti.template(), black_hole_positions: ti.template(), black_hole_masses: ti.template(), black_hole_spins: ti.template(), center_position: ti.template(), max_radius: int, min_radius: float, near_bh_distance:float):
    for i, j in canvas:
        ## kerr superpositon metric
        camera.ray_step_func_kerr(i, j, position[i, j], direction[i, j], center_position[0],
                                  black_hole_masses, black_hole_positions, black_hole_spins, max_radius,
                                  min_radius, near_bh_distance, steps)



def save_data(save_path):
    np_positions = position_fields.to_numpy()
    np_speeds = speed_fields.to_numpy()
    np_accs = acc_fields.to_numpy()
    np_taus = tau_fields.to_numpy()
    assert np.isnan(np_positions).sum() == 0
    assert np.isnan(np_speeds).sum() == 0
    assert np.isnan(np_accs).sum() == 0
    assert np.isnan(np_taus).sum() == 0
    np_data = np.concatenate([np_positions, np_speeds, np_accs, np_taus], axis=3)
    assert np_data.all()
    np.save(save_path, np_data)

def xyz_to_spherical(batch_xyz, bh_position):
    x = batch_xyz[..., 0] - bh_position[0]
    y = batch_xyz[..., 1] - bh_position[1]
    z = batch_xyz[..., 2] - bh_position[2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    epsilon = 1e-9
    theta = torch.acos(z / (r + epsilon))
    phi = torch.atan2(y, x)
    spherical_coords = torch.stack([r, theta, phi], dim=-1)
    return spherical_coords


def generate_vectors_with_angle_range(A, theta_max=np.pi / 2):
    # Get the shape of the input array A
    size = A.shape[:-1]

    # Generate random angles within the range (0, theta_max] for each vector
    theta = np.random.uniform(0, theta_max, size=size)

    # Generate random unit vectors in 3D space for each vector
    R = np.random.rand(*size, 3)
    R = R / np.linalg.norm(R, axis=2, keepdims=True)

    # Calculate the rotation axis for each vector
    axis = np.cross(A, R)
    axis = axis / np.linalg.norm(axis, axis=2, keepdims=True)

    # Rotate each vector a towards its corresponding vector r by angle theta using Rodrigues' rotation formula
    B = A * np.cos(theta)[:, :, np.newaxis] + np.cross(axis, A) * np.sin(theta)[:, :, np.newaxis] + axis * np.sum(
        axis * A, axis=2, keepdims=True) * (1 - np.cos(theta)[:, :, np.newaxis])

    return B

def random_sample_outerspace_position_direction(max_radius, black_hole_positions, size=(1, ), min_bh_radius=8, loose_boundary=0.1):
    if random.random() < 0.5:
        r = np.random.rand(*size) * max_radius
        theta = np.random.rand(*size) * np.pi
        phi = np.random.rand(*size) * 2 * np.pi
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        position = np.array([x, y, z])
        in_bh = (np.sqrt(((position.reshape(3, -1)[np.newaxis] - black_hole_positions[:, :, np.newaxis]) ** 2).sum(1)) < min_bh_radius).sum(0).astype(bool)
        while in_bh.any():
            in_bh_size = (in_bh.sum(), )
            in_bh_r = np.random.rand(*in_bh_size) * max_radius
            in_bh_theta = np.random.rand(*in_bh_size) * np.pi
            in_bh_phi = np.random.rand(*in_bh_size) * 2 * np.pi
            in_bh_x = in_bh_r * np.sin(in_bh_theta) * np.cos(in_bh_phi)
            in_bh_y = in_bh_r * np.sin(in_bh_theta) * np.sin(in_bh_phi)
            in_bh_z = in_bh_r * np.cos(in_bh_theta)
            in_bh_position = np.array([in_bh_x, in_bh_y, in_bh_z])
            temp_position = position.reshape(3, -1)
            temp_position[:, in_bh] = in_bh_position
            in_bh = (np.sqrt(((position.reshape(3, -1)[np.newaxis] - black_hole_positions[:, :, np.newaxis]) ** 2).sum(1)) < min_bh_radius).sum(0).astype(bool)

        r_dir = np.random.rand(*size) * max_radius
        theta_dir = np.random.rand(*size) * np.pi
        phi_dir = np.random.rand(*size) * 2 * np.pi
        vx = r_dir * np.sin(theta_dir) * np.cos(phi_dir)
        vy = r_dir * np.sin(theta_dir) * np.sin(phi_dir)
        vz = r_dir * np.cos(theta_dir)
        direction = np.array([vx - x, vy - y, vz - z])
        direction /= np.sqrt((direction ** 2).sum(0))
        shape = tuple(range(1, len(size) + 1))
        return position.transpose(*shape, 0).astype(np.float32), direction.transpose(*shape, 0).astype(np.float32)
    else:
        number_black_holes = black_hole_positions.shape[0]

        # do not repeat
        r_bh = np.random.rand(*size) * loose_boundary + min_bh_radius
        theta_bh = np.random.rand(*size) * np.pi
        phi_bh = np.random.rand(*size) * 2 * np.pi
        x_bh = r_bh * np.sin(theta_bh) * np.cos(phi_bh)
        y_bh = r_bh * np.sin(theta_bh) * np.sin(phi_bh)
        z_bh = r_bh * np.cos(theta_bh)
        shape = tuple(range(1, len(size) + 1))
        bh2pos = np.array([x_bh, y_bh, z_bh]).transpose(*shape, 0)
        random_indice = np.random.randint(0, number_black_holes, size=size).reshape(-1)
        position = (bh2pos.reshape(-1, 3) + black_hole_positions[random_indice]).reshape(*size, 3)
        norm_vertical_direction = bh2pos / np.sqrt((bh2pos ** 2).sum(-1, keepdims=True))
        direction = generate_vectors_with_angle_range(norm_vertical_direction, np.pi / 2).reshape(*size, 3)
        return position.astype(np.float32), direction.astype(np.float32)


def random_sample_innerspace_position_direction(max_radius, black_hole_position, size=(1, ), min_bh_radius=8, loose_boundary=0.1):
    r = np.random.rand(*size) * (max_radius - min_bh_radius) + min_bh_radius
    theta = np.random.rand(*size) * np.pi
    phi = np.random.rand(*size) * 2 * np.pi
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    position = np.array([x, y, z])
    in_bh = (np.sqrt((position.reshape(3, -1) ** 2).sum(0)) < min_bh_radius).astype(bool)
    while in_bh.any():
        in_bh_size = (in_bh.sum(), )
        in_bh_r = np.random.rand(*in_bh_size) * (max_radius - min_bh_radius) + min_bh_radius
        in_bh_theta = np.random.rand(*in_bh_size) * np.pi
        in_bh_phi = np.random.rand(*in_bh_size) * 2 * np.pi
        in_bh_x = in_bh_r * np.sin(in_bh_theta) * np.cos(in_bh_phi)
        in_bh_y = in_bh_r * np.sin(in_bh_theta) * np.sin(in_bh_phi)
        in_bh_z = in_bh_r * np.cos(in_bh_theta)
        in_bh_position = np.array([in_bh_x, in_bh_y, in_bh_z])
        temp_position = position.reshape(3, -1)
        temp_position[:, in_bh] = in_bh_position
        in_bh = (np.sqrt((position.reshape(3, -1)[np.newaxis] ** 2).sum(1)) < min_bh_radius).sum(0).astype(bool)

    position[0] = position[0] + black_hole_position[0]
    position[1] = position[1] + black_hole_position[1]
    position[2] = position[2] + black_hole_position[2]
    r_dir = np.random.rand(*size) * max_radius
    theta_dir = np.random.rand(*size) * np.pi
    phi_dir = np.random.rand(*size) * 2 * np.pi
    vx = r_dir * np.sin(theta_dir) * np.cos(phi_dir)
    vy = r_dir * np.sin(theta_dir) * np.sin(phi_dir)
    vz = r_dir * np.cos(theta_dir)
    direction = np.array([vx - x, vy - y, vz - z])
    direction /= np.sqrt((direction ** 2).sum(0))
    shape = tuple(range(1, len(size) + 1))
    return position.transpose(*shape, 0).astype(np.float32), direction.transpose(*shape, 0).astype(np.float32)


def sample_indices():
    indices = np.arange(0, steps, steps // N).astype(np.int32)
    return indices[np.newaxis, :]

if __name__ == "__main__":
    set_seed(0)
    canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

    camera = VirtualCamera()
    cnt = 0
    data_length = 1000
    max_radius = 100
    near_bh_distance = 2.4
    near_field_radius = 20
    number_black_holes = 2
    if number_black_holes == 2:
        black_hole_positions = np.array([[-30., 0., 0.], [30., 0., 0.]]).astype(np.float32)
        black_hole_masses = np.array([1., 0.5]).astype(np.float32)
        black_hole_spins = np.array([1., -0.5]).astype(np.float32)
    elif number_black_holes == 3:
        a = 60
        black_hole_positions = np.array([[a/2, a*math.sqrt(3) / 6, 0.],[-a/2, a*math.sqrt(3) / 6, 0.], [0, -a*math.sqrt(3) / 3, 0.]]).astype(np.float32)
        black_hole_masses = np.array([1., 1., 1]).astype(np.float32)
        black_hole_spins = np.array([1., 1., -1]).astype(np.float32)
    else:
        raise NotImplementedError("Please specify the black hole parameters.")


    bh_pos_field = ti.Vector.field(3, dtype=ti.f32, shape=black_hole_positions.shape[0])
    bh_pos_field.from_numpy(black_hole_positions)
    bh_mass_field = ti.field(dtype=ti.f32, shape=black_hole_positions.shape[0])
    bh_mass_field.from_numpy(black_hole_masses)
    bh_spin_field = ti.field(dtype=ti.f32, shape=black_hole_positions.shape[0])
    bh_spin_field.from_numpy(black_hole_spins)
    with open('dataset.yaml', 'r', encoding='utf-8') as f:
        yaml_dir = yaml.load(f.read(), Loader=yaml.FullLoader)['dataset_dir']
    suffix_directory = os.path.join(f"Kerr{black_hole_positions.shape[0]}BH/Multiple",
                                  str(20) + "radius")
    save_directory = os.path.join(yaml_dir, suffix_directory)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    print(f"Kerr{black_hole_positions.shape[0]}BH")
    while cnt < data_length:
        for i in range(0, number_black_holes + 1):
            canvas.fill(0)
            if i == 0:
                max_radius = 105
                sample_max_radius = 100
                sample_min_radius = near_field_radius
                min_radius = near_field_radius - 2
                center_position = np.array([0., 0., 0.]).astype(np.float32)
                position, direction = random_sample_outerspace_position_direction(sample_max_radius, black_hole_positions, (image_width, image_height), min_bh_radius=sample_min_radius)
                dir_prefix = "Outer"
            else:
                max_radius = near_field_radius + 2
                sample_max_radius = near_field_radius
                sample_min_radius = 8
                if number_black_holes == 2:
                    min_radius = 1.55
                elif number_black_holes == 3:
                    min_radius = 2.15
                else:
                    raise NotImplementedError("Please specify the black hole parameters.")
                center_position = black_hole_positions[i - 1]
                position, direction = random_sample_innerspace_position_direction(sample_max_radius, black_hole_positions[i - 1], (image_width, image_height), min_bh_radius=sample_min_radius)
                dir_prefix = "Inner{}".format(i)
            position = np.ascontiguousarray(position)
            direction = np.ascontiguousarray(direction)
            position_field = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
            direction_field = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
            center_position_field = ti.Vector.field(3, dtype=ti.f32, shape=1)
            position_field.from_numpy(position)
            direction_field.from_numpy(direction)
            center_position_field.from_numpy(center_position[np.newaxis])
            assert np.isnan(position).sum() == 0
            assert np.isnan(direction).sum() == 0
            reset_fields()
            calculate(position_field, direction_field, bh_pos_field, bh_mass_field, bh_spin_field, center_position_field, max_radius, min_radius, near_bh_distance)


            save_path = os.path.join(save_directory, dir_prefix, f"{cnt}.npy")
            save_data(save_path)
        cnt += 1
        print(f"iteration {cnt}")
