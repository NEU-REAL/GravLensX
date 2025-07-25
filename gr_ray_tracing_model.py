# from taichi_glsl import *
from model.kerr_schild import *
import taichi as ti
import numpy as np
import math
import torch

PI = 3.14159265


@ti.data_oriented
class Camera:
    def __init__(self, fov=60, aspect_ratio=16/9, image_size=(1920, 1080), accretion_textures=None, sky_sphere_texture=None):
        # Camera parameters
        self.lookfrom = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vup = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fov = fov
        self.aspect_ratio = aspect_ratio

        self.cam_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.christoffles = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(*image_size, 3))
        self.dg_down = ti.Matrix.field(4, 4, dtype=ti.f32, shape=(*image_size, 4))
        self.reset()
        self.image_size = image_size
        if accretion_textures is not None:
            self.accretion_texture = ti.Vector.field(3, dtype=ti.f32, shape=accretion_textures.shape[:-1])
            self.accretion_texture.from_numpy(accretion_textures.astype(np.float32) / 255.0)
        if sky_sphere_texture is not None:
            self.sky_sphere_texture = ti.Vector.field(3, dtype=ti.f32, shape=sky_sphere_texture.shape[:-1])
            self.sky_sphere_texture.from_numpy(sky_sphere_texture.astype(np.float32) / 255.0)


    @ti.kernel
    def reset(self):
        self.lookfrom[None] = [10.0, 0.0, 10.0]
        self.lookat[None] = [0.0, 0.0, 0.0]
        self.vup[None] = [0, 1, 0.0]
        # new defined vup
        self.vup[None] = ti.Vector([0.0, 0.0, 1.0]).cross(self.lookfrom[None]).normalized()
        theta = self.fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        self.cam_origin[None] = self.lookfrom[None]
        w = (self.lookfrom[None] - self.lookat[None]).normalized()
        u = (self.vup[None].cross(w)).normalized()
        v = w.cross(u)
        self.cam_lower_left_corner[None] = ti.Vector([-half_width, -half_height, -1.0])
        self.cam_lower_left_corner[
            None] = self.cam_origin[None] - half_width * u - half_height * v - w
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v


    @ti.kernel
    def set_pos_dir(self, look_from: ti.types.ndarray(dtype=ti.float32, ndim=1), look_at: ti.types.ndarray(dtype=ti.float32, ndim=1)):
        self.lookfrom[None] = [look_from[i] for i in range(3)]
        self.lookat[None] = [look_at[i] for i in range(3)]
        self.cam_origin[None] = self.lookfrom[None]
        # new defined vup
        self.vup[None] = ti.Vector([0.0, 0.0, 1.0])  # .cross(self.lookfrom[None]).normalized()
        w = (self.lookfrom[None] - self.lookat[None]).normalized()
        u = (self.vup[None].cross(w)).normalized()
        v = w.cross(u)
        theta = self.fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        self.cam_lower_left_corner[None] = ti.Vector([-half_width, -half_height, -1.0])
        self.cam_lower_left_corner[
            None] = self.cam_origin[None] - half_width * u - half_height * v - w
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v

    @ti.kernel
    def reset_after_move(self):
        self.cam_origin[None] = self.lookfrom[None]
        # new defined vup
        self.vup[None] = ti.Vector([0.0, 0.0, 1.0])#.cross(self.lookfrom[None]).normalized()
        w = (self.lookfrom[None] - self.lookat[None]).normalized()
        u = (self.vup[None].cross(w)).normalized()
        v = w.cross(u)
        theta = self.fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        self.cam_lower_left_corner[None] = ti.Vector([-half_width, -half_height, -1.0])
        self.cam_lower_left_corner[
            None] = self.cam_origin[None] - half_width * u - half_height * v - w
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v

    def rot_z(self, t):
        self.lookfrom[None][0] = 100.0 * ti.cos(t/100)
        self.lookfrom[None][1] = 100.0 * ti.sin(t/100)
        self.reset_after_move()
        # print("t", t, "lookfrom", self.lookfrom[None])

    def rot_y(self, t):
        self.lookfrom[None][0] = 100.0 * ti.cos(t/100)
        self.lookfrom[None][2] = 100.0 * ti.sin(t/100)
        self.reset_after_move()


    @ti.func # Instead the pixel ratio, we take the pixel index as the input
    def get_ray_kerr(self, i, j, mass, position, spin, time):
        u = (i + ti.random()) / self.image_size[0]
        v = (j + ti.random()) / self.image_size[1]
        return self.ray_color_func_kerr(self.cam_origin[None],
                   self.cam_lower_left_corner[None] + u * self.cam_horizontal[None] + v * self.cam_vertical[None] -
                   self.cam_origin[None], mass, position, spin, time, i, j)



    @ti.func
    def ray_color_func_kerr(self, origin, direction, bh_mass, bh_position, bh_spin, time, pixel_i, pixel_j):
        color = ti.Vector([0.0, 0.0, 0.0])
        transmittance = 1.0
        x, y, z = origin
        dtau = 0.01
        vx, vy, vz = direction.normalized()

        position = ti.Vector([x, y, z])
        speed = ti.Vector([vx, vy, vz])
        vx, vy, vz = speed.normalized()
        r = position.norm()
        g_down = ti.Matrix([[-1., 0, 0, 0], [0, 1, 0, 0],
                            [0, 0, 1, 0], [0, 0, 0, 1]])
        for i in range(len(bh_mass)):
            x_bh, y_bh, z_bh = bh_position[i, :]
            delta_x = x - x_bh
            delta_y = y - y_bh
            delta_z = z - z_bh
            g_down += calculate_g(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
            self.dg_down[pixel_i, pixel_j, 1] += calculate_dgdx(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
            self.dg_down[pixel_i, pixel_j, 2] += calculate_dgdy(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
            self.dg_down[pixel_i, pixel_j, 3] += calculate_dgdz(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
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

        fake_dtau = 1
        in_near_bh_distance_taus = 0.0
        near_bh_distance = 2.4
        for step in range(80000):
            self.christoffles[pixel_i, pixel_j, 0].fill(0.0)
            self.christoffles[pixel_i, pixel_j, 1].fill(0.0)
            self.christoffles[pixel_i, pixel_j, 2].fill(0.0)
            self.dg_down[pixel_i, pixel_j, 0].fill(0.0)
            self.dg_down[pixel_i, pixel_j, 1].fill(0.0)
            self.dg_down[pixel_i, pixel_j, 2].fill(0.0)
            self.dg_down[pixel_i, pixel_j, 3].fill(0.0)
            position = ti.Vector([x, y, z])
            speed = ti.Vector([vx, vy, vz]).normalized()
            vx, vy, vz = speed
            r = position.norm()
            if r > 100:
                volume_color, sigma = self.sky_update_color(position)
                color += volume_color * sigma * transmittance * fake_dtau
                break
            # check if the ray falls into the black hole
            fall_into_bh = False
            if in_near_bh_distance_taus > near_bh_distance * 2 * PI:
                break
            for i in range(len(bh_position)):
                x_bh, y_bh, z_bh = bh_position[i, :]
                relative_position = ti.Vector([x - x_bh, y - y_bh, z - z_bh])
                distance = ti.sqrt((x - x_bh) ** 2 + (y - y_bh) ** 2 + (z - z_bh) ** 2)
                if distance < near_bh_distance:
                    in_near_bh_distance_taus += dtau
                if 6 < distance < 15:
                    if (z - z_bh - vz * dtau) * (z - z_bh) <= 0:
                        volume_color, sigma = self.volume_update_color(relative_position, bh_spin[i], time, i)
                        color += volume_color * sigma * transmittance * fake_dtau
                        transmittance *= ti.exp(-sigma * fake_dtau)


                if distance < 1.6:
                    fall_into_bh = True
                    break
            if fall_into_bh:
                break
            g_down = ti.Matrix([[-1., 0, 0, 0], [0, 1, 0, 0],
                                [0, 0, 1, 0], [0, 0, 0, 1]])
            for i in range(len(bh_mass)):
                x_bh, y_bh, z_bh = bh_position[i, :]
                delta_x = x - x_bh
                delta_y = y - y_bh
                delta_z = z - z_bh
                g_down += calculate_g(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
                self.dg_down[pixel_i, pixel_j, 1] += calculate_dgdx(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
                self.dg_down[pixel_i, pixel_j, 2] += calculate_dgdy(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])
                self.dg_down[pixel_i, pixel_j, 3] += calculate_dgdz(delta_x, delta_y, delta_z, bh_mass[i], bh_spin[i])

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
            multiplied_christoffel = ti.Vector([0., 0., 0., 0.])
            l_tvv = outer_product4(ti.Vector([vt, vx, vy, vz]))
            for mu in ti.static(range(4)):
                christoffel = ti.Matrix.zero(ti.f32, 4, 4)
                for a in ti.static(range(4)):
                    for b in ti.static(range(4)):
                        for sigma in ti.static(range(4)):
                            christoffel[a, b] += 1 / 2 * g_up[mu, sigma] * (self.dg_down[pixel_i, pixel_j, a][b, sigma] + self.dg_down[pixel_i, pixel_j, b][a, sigma] - self.dg_down[pixel_i, pixel_j, sigma][a, b])
                multiplied_christoffel[mu] = (christoffel * l_tvv).sum()
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
        return color


    @ti.func
    def volume_update_color(self, relative_position: ti.template(), bh_spin: float, global_time: float, i: int, outer_radius=16.0):
        x_norm, y_norm, z_norm = relative_position / (outer_radius)
        spin_speed = bh_spin
        rotate_angle = -spin_speed * global_time * PI / 6
        x_norm, y_norm = x_norm * ti.cos(rotate_angle) - y_norm * ti.sin(rotate_angle), x_norm * ti.sin(
            rotate_angle) + y_norm * ti.cos(rotate_angle)
        pixel_half_x = self.accretion_texture.shape[1] // 2
        pixel_half_y = self.accretion_texture.shape[2] // 2
        pixel_x = int(ti.round(x_norm * pixel_half_x)) + pixel_half_x
        pixel_y = int(ti.round(y_norm * pixel_half_y)) + pixel_half_y
        color = self.accretion_texture[i, pixel_x, pixel_y]
        sigma = color.norm()
        return color, sigma

    @ti.func
    def sky_update_color(self, relative_position: ti.template()):
        x, y, z = relative_position
        x, y, z = relative_position / ti.sqrt(x**2 + y**2 + z**2)
        # Calculate spherical coordinates
        theta = ti.atan2(y, x)  # Azimuthal angle
        phi = ti.asin(z)  # Polar angle, clamped to avoid domain errors

        # Convert spherical coordinates to UV coordinates on the image
        image_height = self.sky_sphere_texture.shape[0]
        image_width = self.sky_sphere_texture.shape[1]
        u = int((theta + math.pi) / (2 * math.pi) * image_width)
        v = int((phi + math.pi / 2) / math.pi * image_height)
        pixel_x = u % image_width
        pixel_y = ti.math.clamp(v, 0, image_height - 1)
        color = self.sky_sphere_texture[pixel_y, pixel_x]
        sigma = 1
        return color, sigma

    @ti.func
    def obtain_direction(self, u, v):
        direction = self.cam_lower_left_corner[None] + u * self.cam_horizontal[None] + v * self.cam_vertical[None] - self.cam_origin[None]
        return direction.normalized()



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