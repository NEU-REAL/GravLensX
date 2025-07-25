import os
import taichi as ti
import numpy as np
from gr_ray_tracing_model import Camera, PI
import torch
from model.pinn import positional_encoding
import math
import time
from PIL import Image
ti.init(arch=ti.cuda, default_fp=ti.f32)
# Canvas
aspect_ratio = 16/9
image_width = 1920
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
# Rendering parameters
samples_per_pixel = 1


a = 60
bh_masses = np.array([1., 1., 1.], dtype=np.float32)
bh_positions = np.array([[a / 2, a * ti.sqrt(3) / 6, 0.], [-a / 2, a * ti.sqrt(3) / 6, 0.], [0, -a * ti.sqrt(3) / 3, 0.]], dtype=np.float32)
bh_spins = np.array([1., -1., 1], dtype=np.float32)
num_bh = bh_masses.shape[0]
bh_mass = ti.field(ti.f32, shape=num_bh)  # Mass field
bh_position = ti.Matrix.field(3, 3, dtype=ti.f32, shape=1)  # Position field
bh_spin = ti.field(ti.f32, shape=num_bh)  # Spin field
bh_time = 0.
bh_mass.from_numpy(bh_masses)
bh_position.from_numpy(bh_positions[np.newaxis, ...])
bh_spin.from_numpy(bh_spins)




@ti.kernel
def render_kerr():
    for i, j in canvas:
        color = ti.Vector([0.0, 0.0, 0.0])
        for n in range(samples_per_pixel):
            mass_vec = ti.Vector([bh_mass[k] for k in range(num_bh)])
            pos_vec = bh_position[0]
            spin_vec = ti.Vector([bh_spin[k] for k in range(num_bh)])


            color += camera.get_ray_kerr(i, j, mass_vec, pos_vec, spin_vec, bh_time)
        color /= samples_per_pixel
        canvas[i, j] += color



if __name__ == "__main__":
    fov = 60
    textures = []
    texture_dir = "texture"
    for i in range(1, 4):
        texture_path = f"{texture_dir}/texture_real{i}.png"
        textures.append(np.array(Image.open(texture_path)))

    textures = np.stack(textures, axis=0)
    sky_sphere_path = f"{texture_dir}/texture_sky.png"
    sky_sphere_texture = np.array(Image.open(sky_sphere_path))
    camera = Camera(fov, image_size=(image_width, image_height), accretion_textures=textures, sky_sphere_texture=sky_sphere_texture)
    gui = ti.GUI("Black Hole", res=(image_width, image_height), show_gui=False)
    canvas.fill(0)
    canvas.fill(0)
    look_from = torch.tensor([0., 60, 5])
    look_at = torch.tensor([0., 0., 0.])
    camera.set_pos_dir(look_from, look_at)
    t1 = time.time()
    render_kerr()
    gui.set_image(canvas.to_numpy())
    image_dir = "images/iteration"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    save_path = image_dir + "/euler.png"
    gui.show(save_path)
    t2 = time.time()
    print("render time: ", t2 - t1)

