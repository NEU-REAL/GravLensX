import os
import taichi as ti
import numpy as np
from gr_ray_tracing_model import Camera, PI
import torch
from model.pinn import positional_encoding
import math
from model.pinn import GeodesicNet
import argparse
import time
import torch.nn.functional as F
from PIL import Image
ti.init(arch=ti.cuda)



@ti.kernel
def update_direction_fields():
    for i, j in direction_fields:
        u = i / image_width
        v = j / image_height
        direction_fields[i, j] = camera.obtain_direction(u, v)


def dir_to_2d_coord(directions, image_width, image_height):
    """
    Convert a batch of normalized 3D direction vectors in PyTorch tensor format to
    2D coordinates on an equirectangular image projection.

    Parameters:
        directions (torch.Tensor): Batch of normalized 3D direction vectors (batch_size, 3).
        image_width (int): Width of the equirectangular image.
        image_height (int): Height of the equirectangular image.

    Returns:
        torch.Tensor: Batch of 2D coordinates (batch_size, 2) on the equirectangular image.
    """
    # Normalize the direction vectors just in case they're not normalized
    directions = directions / torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))

    # Calculate spherical coordinates
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    theta = torch.atan2(y, x)   # Azimuthal angle
    phi = torch.asin(z.clamp(-1, 1))  # Polar angle, clamped to avoid domain errors

    # Convert spherical coordinates to UV coordinates on the image
    u = ((theta + torch.pi) / (2 * torch.pi) * image_width).to(torch.int64)
    v = ((phi + torch.pi/2) / torch.pi * image_height).to(torch.int64)

    # Ensure the coordinates are within the image bounds
    u = u % image_width
    v = torch.clamp(v, 0, image_height - 1)

    return torch.stack((u, v), dim=1)

def update_color_tensor(pos, distances, i, outer_radius=16.0):
    distances = distances.unsqueeze(1)
    x_norm, y_norm, z_norm = (pos / (outer_radius)).permute(1, 0).contiguous()
    pixel_x = (torch.round(x_norm * Wa / 2) + Wa / 2).to(torch.int64)
    pixel_y = (torch.round(y_norm * Ha / 2) + Ha / 2).to(torch.int64)
    flatten_indices = pixel_x + pixel_y * Wa + i * Wa * Ha
    color = flatten_accretion_texture[flatten_indices]
    return color


def color_calculating_torch(ray_positions, batch_indices, ignore_distance, region):
    ray_1 = ray_positions[:-1]
    ray_2 = ray_positions[1:]
    z1 = ray_1[:, 2].unsqueeze(1) - pos_bh[:, 2].unsqueeze(0)
    z2 = ray_2[:, 2].unsqueeze(1) - pos_bh[:, 2].unsqueeze(0)
    same_region = region[1:] == region[:-1]
    across_bh = torch.logical_and(torch.logical_and((z1 * z2 < 0).any(dim=1), ~ignore_distance[:-1]), same_region)
    across_ray1 = ray_1[across_bh]
    across_ray2 = ray_2[across_bh]
    across_z1 = z1[across_bh]
    across_z2 = z2[across_bh]
    across_batch_indices = batch_indices[:-1][across_bh]
    interpolated_positions = across_ray1.unsqueeze(1) + (across_ray2 - across_ray1).unsqueeze(1) * (-across_z1 / (across_z2 - across_z1)).unsqueeze(2)
    relative_positions = interpolated_positions - pos_bh.unsqueeze(0)
    distances = (relative_positions).norm(dim=-1)
    colors = torch.zeros((distances.size(0), 3), device=device)
    for i in range(pos_bh.shape[0]):
        distance_bh = distances[:, i]
        relative_positions_bh = relative_positions[:, i]
        in_accretion_idx = (5 < distance_bh) & (distance_bh < 15)
        colors[in_accretion_idx] += update_color_tensor(relative_positions_bh[in_accretion_idx], distance_bh[in_accretion_idx], i)
    return colors, across_batch_indices

def check_across_z(ray_positions, batch_indices, ignore_distance, region):
    ray_1 = ray_positions[:-1]
    ray_2 = ray_positions[1:]
    z1 = ray_1[:, 2].unsqueeze(1) - pos_bh[:, 2].unsqueeze(0)
    z2 = ray_2[:, 2].unsqueeze(1) - pos_bh[:, 2].unsqueeze(0)
    same_region = region[1:] == region[:-1]
    across_bh = torch.logical_and(torch.logical_and((z1 * z2 < 0).any(dim=1), ~ignore_distance[:-1]), same_region)
    across_bh = torch.cat([across_bh, torch.tensor([False], device=device)])
    # across_batch_indices = batch_indices[:-1][across_bh]
    return across_bh

def aabb_intersection_calculate(models, positions, directions, bh_pos, bh_render_radius, args):
    """
    :param model: the ray tracing model
    :param positions: the initial positions of the rays (batch_size, 3)
    :param directions: the initial directions of the rays (batch_size, 3)
    :param bh_pos: the positions of the black hole (N, 3)
    :param bh_render_radius: the rendering radius of the black holes which contain all the essential rendering information
    :
    :return: the start and end time pairs of the rays intersecting with the black holes
    """
    # tolerable error for the intersection
    sky_sphere_direction = torch.zeros(positions.size(0), 3, device=positions.device)
    tolerable_error = 1e-4
    total_time = torch.zeros(3)
    batch_size = args.aabb_batch_size
    cos_distance_ratio = args.cos_distance_ratio
    constant_tau = args.constant_tau
    for model in models:
        model.eval()
    data_pairs = torch.zeros(positions.size(0), 14, 8).to(positions.device) # the tensor to store the position, direction, and tau pairs belonging to each region
    # The last dimension of the data_pairs tensor is organized as (3, 3, 1, 1), position, direction, region index, end tau
    data_pairs[:, 0, :3] = positions
    data_pairs[:, 0, 3:6] = directions
    diff2bhs = bh_pos.unsqueeze(0) - positions.unsqueeze(1)
    distance2bhs = torch.norm(diff2bhs, dim=-1)
    current_status = torch.zeros(positions.size(0), distance2bhs.size(1), dtype=torch.bool).to(positions.device)
    current_status[
        torch.logical_and(distance2bhs < bh_render_radius, distance2bhs == distance2bhs.min(1, keepdim=True)[0])] = True
    current_status = torch.cat([~current_status.sum(1, keepdim=True).bool(), current_status],
                               dim=1)  # the first column is the outside status
    nonzero_inside_idx = current_status.nonzero(as_tuple=True)[1]
    data_pairs[:, 0, 6] = nonzero_inside_idx
    data_pairs_idx = torch.zeros(positions.size(0), dtype=torch.int64).to(positions.device) # the index of the data pairs
    remaining_indices = torch.arange(positions.size(0)).to(positions.device)
    current_positions = positions.clone()
    current_directions = directions.clone()
    current_taus = torch.zeros(positions.size(0), dtype=positions.dtype).to(positions.device)
    half_batch_size = math.ceil(batch_size / 2)
    # check whether the initial points fall into the rendering space
    def update_inside_state(distance2bhs):
        current_status = torch.zeros(current_positions.size(0), distance2bhs.size(1), dtype=torch.bool).to(positions.device)
        current_status[torch.logical_and(distance2bhs <= bh_render_radius, distance2bhs == distance2bhs.min(1, keepdim=True)[0])] = True
        current_status = torch.cat([~current_status.sum(1, keepdim=True).bool(), current_status], dim=1) # the first column is the outside status
        nonzero_inside_idx = current_status.nonzero(as_tuple=True)[1]
        turn_idx = nonzero_inside_idx != data_pairs[remaining_indices, data_pairs_idx[remaining_indices], 6]
        if turn_idx.any():
            remaining_data_pairs = data_pairs[remaining_indices]
            remaining_data_pairs[turn_idx, data_pairs_idx[remaining_indices][turn_idx], 7] = current_taus[turn_idx]
            valid_turn_idx = torch.logical_and(turn_idx, data_pairs_idx[remaining_indices] < data_pairs.size(1) - 1)
            if valid_turn_idx.any():
                valid_turn_idx = torch.logical_and(turn_idx, data_pairs_idx[remaining_indices] < data_pairs.size(1) - 1)
                remaining_data_pairs[valid_turn_idx, data_pairs_idx[remaining_indices][valid_turn_idx] + 1, :7] = torch.cat(
                    [current_positions[valid_turn_idx], current_directions[valid_turn_idx],
                     nonzero_inside_idx[valid_turn_idx].unsqueeze(1)], dim=1)
            data_pairs[remaining_indices] = remaining_data_pairs
            # reset current taus, positions, and directions
            current_taus[turn_idx] = 0
            positions[turn_idx] = current_positions[turn_idx]
            directions[turn_idx] = current_directions[turn_idx]
            remaining_data_pairs_idx = data_pairs_idx[remaining_indices]
            remaining_data_pairs_idx[turn_idx] += 1
            data_pairs_idx[remaining_indices] = remaining_data_pairs_idx
        assert (nonzero_inside_idx == data_pairs[remaining_indices, data_pairs_idx[remaining_indices], 6]).all()
        return current_status


    def calculate_outer_step_tau(positions, diff2bhs, distance2bhs, directions):
        """
        :param positions: the current positions of the rays (batch_size, 3)
        :param diff2bhs: the difference between the black hole positions and the current positions (batch_size, N, 3)
        :param distance2bhs: the distance between the black hole positions and the current positions (batch_size, N)
        :param directions: the current directions of the rays (batch_size, 3)
        :return: delta_taus:the outer step tau for the rays (batch_size)
        """
        loose_bh_render_radius = bh_render_radius - args.loose_boundary
        horizontal_distance = (diff2bhs * directions.unsqueeze(1)).sum(dim=-1) # equal to cos(theta)
        vertical_distance = (distance2bhs ** 2 - horizontal_distance ** 2).sqrt()
        near_bh = distance2bhs <= bh_render_radius + args.bh_radius_tolerance
        cross_bh = torch.logical_and(torch.logical_and(vertical_distance <= loose_bh_render_radius, horizontal_distance > 0), near_bh)
        nearly_cross_bh = torch.logical_and(
            torch.logical_and(vertical_distance <= (loose_bh_render_radius + args.bh_radius_tolerance),
                              horizontal_distance > 0), distance2bhs > bh_render_radius + args.bh_radius_tolerance)
        bh_parallel_distance = (loose_bh_render_radius ** 2 - vertical_distance ** 2).sqrt()
        nearly_cross_bh_parallel_distance = ((loose_bh_render_radius + args.bh_radius_tolerance) ** 2 - vertical_distance ** 2).sqrt()
        nonzero_horizontal_distance = torch.where(cross_bh, horizontal_distance - bh_parallel_distance, 1e9 * torch.ones_like(horizontal_distance))
        nonzero_horizontal_distance = torch.where(nearly_cross_bh, horizontal_distance - nearly_cross_bh_parallel_distance, nonzero_horizontal_distance)
        delta_taus = nonzero_horizontal_distance.min(-1)[0]

        loose_max_radius = args.max_radius + args.loose_boundary
        horizontal_distance2outside = (positions * directions).sum(dim=-1)
        vertical_distance2outside = (positions.norm(dim=-1) ** 2 - horizontal_distance2outside ** 2).sqrt()
        vertical_distance2outside = torch.where(torch.isnan(vertical_distance2outside), torch.zeros_like(vertical_distance2outside), vertical_distance2outside)
        horizontal_global_distance2outside = (loose_max_radius ** 2 - vertical_distance2outside ** 2).sqrt()
        nonzero_distance2outside = horizontal_global_distance2outside - horizontal_distance2outside
        outside_idx = ~torch.logical_or(cross_bh.any(dim=-1), nearly_cross_bh.any(dim=-1))
        delta_taus[outside_idx] = nonzero_distance2outside[outside_idx]
        assert torch.isnan(delta_taus).sum() == 0
        assert (delta_taus > 1e7).sum() == 0
        return delta_taus

    def calculate_inner_step_tau(diff2bhs, distance2bhs, directions):
        """
        :param diff2bhs: the difference between the black hole positions and the current positions (batch_size, N, 3)
        :param distance2bhs: the distance between the black hole positions and the current positions (batch_size, N)
        :param directions: the current directions of the rays (batch_size, 3)
        :return: delta_taus: the inner step tau for the rays (batch_size)
        """
        loose_bh_render_radius = bh_render_radius + args.loose_boundary
        nearest_bh_distance, nearest_bh_idx = distance2bhs.min(dim=-1)
        diff2nearest_bh = diff2bhs[torch.arange(diff2bhs.size(0)), nearest_bh_idx] # (batch_size, 3)
        dot_products = (diff2nearest_bh * directions).sum(dim=-1)
        vertical_distances = torch.sqrt((nearest_bh_distance.clamp(max=loose_bh_render_radius) ** 2 - dot_products ** 2).clamp(min=0))
        distance2intersection = torch.sqrt(loose_bh_render_radius ** 2 - vertical_distances ** 2) + dot_products
        fall_in_bh_radius = args.fall_in_bh_coefficient * mass_bh
        in_fall_in_bh_radius = (distance2bhs < fall_in_bh_radius.unsqueeze(0))[torch.arange(diff2bhs.size(0)), nearest_bh_idx]
        intersect_fall_in_radius = torch.logical_and(dot_products > 0, vertical_distances < fall_in_bh_radius.unsqueeze(0).repeat(vertical_distances.size(0), 1)[torch.arange(diff2bhs.size(0)), nearest_bh_idx])
        probable_fall_in_bh_radius_idx = torch.logical_or(in_fall_in_bh_radius, intersect_fall_in_radius)
        delta_taus = distance2intersection#.clamp(min=2)
        assert torch.isnan(delta_taus).sum() == 0
        distance2minradius = nearest_bh_distance - args.min_radius
        # if the black hole neural network is not stable, clamp the delta tau to the distance to the nearest black hole
        delta_taus = torch.where(probable_fall_in_bh_radius_idx, distance2minradius, delta_taus)
        return delta_taus

    def reach_edge_update(positions):
        """
        :param positions: current ray positions
        :return: whether the rays reach the edge
        """
        near_bh_distance = args.near_bh_distance
        outside = positions.norm(dim=-1) >= args.max_radius - tolerable_error
        remaining_sky_sphere_direction = sky_sphere_direction[remaining_indices]
        remaining_sky_sphere_direction[outside] = positions[outside] / positions[outside].norm(dim=-1).unsqueeze(-1)
        sky_sphere_direction[remaining_indices] = remaining_sky_sphere_direction
        inside_bh = (positions.unsqueeze(1) - bh_pos.unsqueeze(0)).norm(dim=-1).min(1)[0] <= near_bh_distance
        reduce_idx = torch.logical_or(outside, inside_bh)
        return reduce_idx

    iter = 0
    region_count = torch.zeros(bh_pos.shape[0] + 1, dtype=torch.int64).to(positions.device)
    region_time = torch.zeros(bh_pos.shape[0] + 1, dtype=torch.float32).to(positions.device)
    while positions.shape[0] > 0:
        torch.cuda.synchronize()
        t1 = time.time()
        diff2bhs = bh_pos.unsqueeze(0) - current_positions.unsqueeze(1)  # (batch_size, N, 3)
        distance2bhs = torch.norm(diff2bhs, dim=-1)  # (batch_size, N)
        current_status = update_inside_state(distance2bhs) # (batch_size, num_bh + 1)
        region_count += current_status.sum(0)
        outer_status = current_status[:, 0]
        inner_status = current_status[:, 1:].sum(1).to(torch.bool)
        # calculate the outer step tau
        delta_taus = calculate_outer_step_tau(current_positions[outer_status], diff2bhs[outer_status],
                                              distance2bhs[outer_status], current_directions[outer_status])
        current_taus[outer_status] += delta_taus
        # calculate the inner step tau
        if inner_status.any():
            delta_taus = calculate_inner_step_tau(diff2bhs[inner_status],
                                                  distance2bhs[inner_status], current_directions[inner_status])
            current_taus[inner_status] += delta_taus
        t2 = time.time()
        for model_idx in range(bh_pos.shape[0] + 1):
            inradius_idx = current_status[:, model_idx]
            inradius_position = positions[inradius_idx]
            inradius_direction = directions[inradius_idx]
            inradius_taus = current_taus[inradius_idx]
            inradius_output_positions = []
            if model_idx == 0:
                max_radius = args.max_radius
                min_radius = -args.max_radius
            else:
                max_radius = pos_bh[model_idx - 1].unsqueeze(0) + bh_render_radius
                min_radius = pos_bh[model_idx - 1].unsqueeze(0) - bh_render_radius
            for idx in range(math.ceil(len(positions) / half_batch_size)):
                start_batch = idx * half_batch_size
                end_batch = (idx + 1) * half_batch_size
                batched_positions = inradius_position[start_batch:end_batch].repeat(2, 1)
                batched_directions = inradius_direction[start_batch:end_batch].repeat(2, 1)
                batched_taus = inradius_taus[start_batch:end_batch].unsqueeze(0).repeat(2, 1)
                batched_taus[1] += constant_tau
                batched_taus = batched_taus.flatten()
                # Normalize positions
                batched_positions = (batched_positions - min_radius) / (max_radius - min_radius)
                encoded_init_p = positional_encoding(batched_positions)
                encoded_direction = positional_encoding(batched_directions)
                torch.cuda.synchronize()
                model_ts = time.time()
                with torch.no_grad():
                    batched_output_positions = models[model_idx](torch.cat([encoded_init_p, encoded_direction, batched_taus.unsqueeze(1)], dim=1))
                    if args.norm_output:
                        if model_idx == 0:
                            batched_output_positions = batched_output_positions
                        else:
                            batched_output_positions = batched_output_positions + pos_bh[model_idx - 1].unsqueeze(0)
                torch.cuda.synchronize()
                model_te = time.time()
                inradius_output_positions.append(batched_output_positions.view(2, batched_output_positions.shape[0] // 2, 3))
            inradius_output_positions = torch.cat(inradius_output_positions, dim=1)
            inradius_output_directions = inradius_output_positions[1] - inradius_output_positions[0]
            assert torch.isnan(inradius_output_directions / inradius_output_directions.norm(dim=-1).unsqueeze(-1)).sum() == 0
            inradius_output_directions = inradius_output_directions / inradius_output_directions.norm(dim=-1).unsqueeze(-1)
            inradius_output_positions = inradius_output_positions[0]
            current_directions[inradius_idx] = inradius_output_directions
            assert torch.isnan(current_directions).sum() == 0
            current_positions[inradius_idx] = inradius_output_positions
            region_time[model_idx] += model_te - model_ts
        torch.cuda.synchronize()
        t3 = time.time()
        # drop the rays that have reached the edge
        drop_idx = reach_edge_update(current_positions)

        update_idx = torch.logical_and(drop_idx, data_pairs[remaining_indices, data_pairs_idx[remaining_indices], -1] == 0)
        if update_idx.any():
            remaining_data_pairs = data_pairs[remaining_indices]
            remaining_data_pairs[update_idx, data_pairs_idx[remaining_indices][update_idx], -1] = current_taus[update_idx]
            data_pairs[remaining_indices] = remaining_data_pairs

        drop_idx = torch.logical_or(drop_idx, data_pairs_idx[remaining_indices] >= data_pairs.size(1))
        if drop_idx.any():
            reserve_idx = ~drop_idx
            remaining_indices = remaining_indices[reserve_idx]
            positions = positions[reserve_idx]
            directions = directions[reserve_idx]
            current_directions = current_directions[reserve_idx]
            current_taus = current_taus[reserve_idx]
            current_positions = current_positions[reserve_idx]
        iter += 1
        torch.cuda.synchronize()
        t4 = time.time()
        total_time += torch.tensor([t2 - t1, t3 - t2, t4 - t3])
    print(f"delta step time: {total_time[0]}, model time: {total_time[1]}, update time: {total_time[2]}")
    print("region time", region_time)
    return data_pairs, sky_sphere_direction

def neural_render(models, data_pairs, sky_sphere_direction, args):
    """
    :param model: ray tracing model
    :param positions: the initial positions of the rays (batch_size, 3)
    :param directions: the initial directions of the rays (batch_size, 3)
    :param data_pairs: the position, direction, end tau of the rays (batch_size, 14, 8)
    :return: colors: the colors of the rays (batch_size, 3)
    """
    t1 = time.time()
    batch_size = args.color_batch_size
    colors1 = torch.zeros((data_pairs.size(0)), device=device)
    colors2 = torch.zeros((data_pairs.size(0)), device=device)
    colors3 = torch.zeros((data_pairs.size(0)), device=device)
    # Flatten tau pairs and compute all possible taus

    # Filter to valid ranges where start < end
    batch_indices = torch.arange(data_pairs.size(0), device=device).repeat_interleave(data_pairs.size(1))
    inside_bh_idx = torch.logical_and(data_pairs[:, :, -1] != 0, data_pairs[:, :, 6] != 0) # select data that both has tau value and inside black hole regions
    outside_bh_idx = torch.logical_and(data_pairs[:, :, -1] != 0, data_pairs[:, :, 6] == 0) # select data that both has tau value and outside black hole regions
    inside_batch_indices = batch_indices[inside_bh_idx.view(-1)]
    flatten_data_pairs = data_pairs[inside_bh_idx]

    # a function reserved for calculating the far-field colors
    def calculate_outside_colors(outside_bh_idx):
        max_radius = args.max_radius
        min_radius = -args.max_radius
        flatten_data_pairs = data_pairs[outside_bh_idx]
        outside_batch_indices = batch_indices[outside_bh_idx.view(-1)]
        delta_tau = (0 - flatten_data_pairs[:, 2]) / flatten_data_pairs[:, 5]
        valid_tau_idx = delta_tau > 0
        current_taus = delta_tau[valid_tau_idx]
        network_positions = torch.ones((current_taus.size(0), 3), device=device)
        colors = torch.zeros((flatten_data_pairs.size(0), 3), device=device)
        if valid_tau_idx.any():
            colors = colors[valid_tau_idx]
            flatten_data_pairs = flatten_data_pairs[valid_tau_idx]
            outside_batch_indices = outside_batch_indices[valid_tau_idx]
            unqualified_idx = torch.arange(flatten_data_pairs.size(0), device=device)
            previous_positions = flatten_data_pairs[:, :3].clone()
            previous_taus = torch.zeros(flatten_data_pairs.size(0), device=device)
            positions_render = flatten_data_pairs[:, :3]
            directions_render = flatten_data_pairs[:, 3:6]
            while unqualified_idx.any():
                normalized_positions = (positions_render - min_radius) / (max_radius - min_radius)
                output_positions = []
                for idx in range(math.ceil(len(positions_render) / batch_size)):
                    start_batch = idx * batch_size
                    end_batch = (idx + 1) * batch_size
                    encoded_init_p = positional_encoding(normalized_positions[start_batch:end_batch].to(device))
                    encoded_direction = positional_encoding(directions_render[start_batch:end_batch].to(device))
                    batched_input_data = torch.cat([encoded_init_p, encoded_direction,
                                                    current_taus[start_batch:end_batch].unsqueeze(1).to(device)],
                                                   dim=1)
                    with torch.no_grad():
                        batched_output_positions = models[model_idx](batched_input_data)
                output_positions.append(batched_output_positions)
                output_positions = torch.cat(output_positions, dim=0)
                qualified_idx = output_positions[:, 2] < 0.01
                network_positions[unqualified_idx[qualified_idx]] = output_positions[qualified_idx]
                unqualified_idx = unqualified_idx[~qualified_idx]
                previous_positions = previous_positions[~qualified_idx]
                previous_taus = previous_taus[~qualified_idx]
                positions_render = positions_render[~qualified_idx]
                directions_render = directions_render[~qualified_idx]
                output_positions = output_positions[~qualified_idx]
                if unqualified_idx.size(0) != 0:
                    neighbor_direction = output_positions - previous_positions
                    neighbor_direction = neighbor_direction / neighbor_direction.norm(dim=-1).unsqueeze(-1)
                    delta_tau = (0 - output_positions[:, 2]) / neighbor_direction[:, 2]
                    current_taus = previous_taus + delta_tau
                    previous_positions = output_positions
                    previous_taus = current_taus
            distance2bhs = (pos_bh.unsqueeze(0) - network_positions.unsqueeze(1)).norm(dim=-1)
            distance2bh, nearest_bh_idx = distance2bhs.min(dim=-1)
            in_accretion_idx = (6 < distance2bh) & (distance2bh < 15)
            colors[in_accretion_idx] += update_color_tensor(distance2bh[in_accretion_idx], nearest_bh_idx[in_accretion_idx])
        return colors, outside_batch_indices



    # Vectorized creation of all taus
    taus_render = torch.zeros((flatten_data_pairs.size(0), args.render_coarse_max_num_step), device=device)
    flatten_size = flatten_data_pairs.size(0)
    @ti.kernel
    def set_taus_field(data_render: ti.types.ndarray(dtype=ti.float32, ndim=2),
                       sampled_data_taus: ti.types.ndarray(dtype=ti.float32, ndim=2)):
        for i in range(flatten_size):
            diff = data_render[i, 7]
            # limit the largest step of tau_step to prevent instable sampled points
            diff = ti.min(diff, args.render_max_tau_interval)
            tau_step = diff / (args.render_coarse_max_num_step - 1)
            # limit the smallest step of tau_step to prevent instable results
            for j in range(args.render_coarse_max_num_step):
                sampled_data_taus[i, j] = j * tau_step

    set_taus_field(flatten_data_pairs, taus_render)


    lengths = taus_render.size(1)
    taus_render = taus_render.flatten()

    inside_batch_indices = torch.repeat_interleave(inside_batch_indices, lengths).cpu()
    positions_render = torch.repeat_interleave(flatten_data_pairs[:, :3], lengths, dim=0).cpu()
    directions_render = torch.repeat_interleave(flatten_data_pairs[:, 3:6], lengths, dim=0).cpu()
    region_render = torch.repeat_interleave(flatten_data_pairs[:, 6], lengths).cpu()
    output_positions = torch.zeros(taus_render.size(0), 3, device="cpu")
    torch.cuda.synchronize()
    t2 = time.time()
    t_model = torch.zeros(3)
    # Normalize positions
    for model_idx in range(pos_bh.size(0) + 1):
        in_radius_idx = region_render == model_idx
        in_radius_positions_render = positions_render[in_radius_idx].to(device)
        in_radius_directions_render = directions_render[in_radius_idx].to(device)
        in_radius_taus_render = taus_render[in_radius_idx].to(device)
        if in_radius_positions_render.size(0) == 0: # No rays in the region
            continue
        if model_idx == 0:
            max_radius = args.max_radius
            min_radius = -args.max_radius
        else:
            max_radius = pos_bh[model_idx - 1].unsqueeze(0) + bh_render_radius
            min_radius = pos_bh[model_idx - 1].unsqueeze(0) - bh_render_radius
        in_radius_positions_render = (in_radius_positions_render - min_radius) / (max_radius - min_radius)
        inradius_output_positions = []
        for idx in range(math.ceil(len(in_radius_positions_render) / batch_size)):
            start_batch = idx * batch_size
            end_batch = (idx + 1) * batch_size
            torch.cuda.synchronize()
            t_model_1 = time.time()
            encoded_init_p = positional_encoding(in_radius_positions_render[start_batch:end_batch].to(device))
            torch.cuda.synchronize()
            t_model_11 = time.time()
            encoded_direction = positional_encoding(in_radius_directions_render[start_batch:end_batch].to(device))
            batched_input_data = torch.cat([encoded_init_p, encoded_direction, in_radius_taus_render[start_batch:end_batch].unsqueeze(1).to(device)], dim=1)
            torch.cuda.synchronize()
            t_model_2 = time.time()
            with torch.no_grad():
                batched_output_positions = models[model_idx](batched_input_data)
                if args.norm_output:
                    if model_idx == 0:
                        batched_output_positions = batched_output_positions
                    else:
                        batched_output_positions = batched_output_positions + pos_bh[
                            model_idx - 1].unsqueeze(0)
            inradius_output_positions.append(batched_output_positions)
            torch.cuda.synchronize()
            t_model_3 = time.time()
            t_model += torch.tensor([t_model_11 - t_model_1, t_model_2 - t_model_11, t_model_3 - t_model_2])
        inradius_output_positions = torch.cat(inradius_output_positions, dim=0)
        output_positions[in_radius_idx] = inradius_output_positions.cpu()
    # the indices indicating the start and end time of two rays should be ignored
    ignore_distance_idx = torch.cumsum(inside_batch_indices.unique(return_counts=True)[1], dim=0) - 1
    ignore_distance = torch.zeros(taus_render.size(0), dtype=torch.bool, device=device)
    ignore_distance[ignore_distance_idx] = True


    across_bh = []
    for idx in range(math.ceil(len(output_positions) / args.max_color_batch_size)):
        start_batch = idx * args.max_color_batch_size
        end_batch = (idx + 1) * args.max_color_batch_size
        across_bh_ = check_across_z(output_positions[start_batch:end_batch].to(device), inside_batch_indices[start_batch:end_batch].to(device), ignore_distance[start_batch:end_batch].to(device), region_render[start_batch:end_batch].to(device))
        across_bh.append(across_bh_)
    across_bh = torch.cat(across_bh, dim=0).cpu()
    across_taus_pre = taus_render[across_bh]
    across_bh_post = torch.cat([torch.tensor([False]), across_bh[:-1]], dim=0)
    across_taus_post = taus_render[across_bh_post]
    across_taus_both = torch.cat([across_taus_pre.unsqueeze(1), across_taus_post.unsqueeze(1)], dim=1)
    across_input_positions = positions_render[across_bh]
    across_inside_batch_indices = inside_batch_indices[across_bh]
    across_input_directions = directions_render[across_bh]
    across_size = across_input_positions.size(0)
    across_region_render = region_render[across_bh]

    @ti.kernel
    def set_finegrain_taus_field(across_taus_both: ti.types.ndarray(dtype=ti.float32, ndim=2),
                       sampled_data_taus: ti.types.ndarray(dtype=ti.float32, ndim=2)):
        for i in range(across_size):
            diff = across_taus_both[i, 1] - across_taus_both[i, 0]
            tau_step = diff / (args.render_finegrain_max_num_step - 1)
            for j in range(args.render_finegrain_max_num_step):
                sampled_data_taus[i, j] = across_taus_both[i, 0] + j * tau_step

    finegrain_taus_render = torch.zeros((across_size, args.render_finegrain_max_num_step), device=device)
    set_finegrain_taus_field(across_taus_both, finegrain_taus_render)
    nonzero_idx = finegrain_taus_render != 0
    lengths = nonzero_idx.sum(dim=1).cpu()
    finegrain_taus_render = finegrain_taus_render[nonzero_idx]
    finegrain_inside_batch_indices = torch.repeat_interleave(across_inside_batch_indices, lengths)
    finegrain_positions_render = torch.repeat_interleave(across_input_positions, lengths, dim=0)
    finegrain_directions_render = torch.repeat_interleave(across_input_directions, lengths, dim=0)
    finegrain_region_render = torch.repeat_interleave(across_region_render, lengths)
    finegrain_output_positions = torch.zeros(finegrain_taus_render.size(0), 3, device="cpu")
    torch.cuda.synchronize()
    # Normalize positions
    for model_idx in range(pos_bh.size(0) + 1):
        in_radius_idx = finegrain_region_render == model_idx
        in_radius_finegrain_positions_render = finegrain_positions_render[in_radius_idx].to(device)
        in_radius_finegrain_directions_render = finegrain_directions_render[in_radius_idx].to(device)
        in_radius_finegrain_taus_render = finegrain_taus_render[in_radius_idx].to(device)
        if in_radius_finegrain_positions_render.size(0) == 0:  # No rays in the region
            continue
        if model_idx == 0:
            max_radius = args.max_radius
            min_radius = -args.max_radius
        else:
            max_radius = pos_bh[model_idx - 1].unsqueeze(0) + bh_render_radius
            min_radius = pos_bh[model_idx - 1].unsqueeze(0) - bh_render_radius
        in_radius_finegrain_positions_render = (in_radius_finegrain_positions_render - min_radius) / (max_radius - min_radius)
        inradius_finegrain_output_positions = []
        for idx in range(math.ceil(len(in_radius_finegrain_positions_render) / batch_size)):
            start_batch = idx * batch_size
            end_batch = (idx + 1) * batch_size
            torch.cuda.synchronize()
            t_model_1 = time.time()
            encoded_init_p = positional_encoding(in_radius_finegrain_positions_render[start_batch:end_batch].to(device))
            torch.cuda.synchronize()
            t_model_11 = time.time()
            encoded_direction = positional_encoding(in_radius_finegrain_directions_render[start_batch:end_batch].to(device))
            batched_input_data = torch.cat([encoded_init_p, encoded_direction,
                                            in_radius_finegrain_taus_render[start_batch:end_batch].unsqueeze(1).to(device)],
                                           dim=1)
            torch.cuda.synchronize()
            t_model_2 = time.time()
            with torch.no_grad():
                batched_output_positions = models[model_idx](batched_input_data)
                if args.norm_output:
                    if model_idx == 0:
                        batched_output_positions = batched_output_positions
                    else:
                        batched_output_positions = batched_output_positions + pos_bh[
                            model_idx - 1].unsqueeze(0)
            inradius_finegrain_output_positions.append(batched_output_positions)
            torch.cuda.synchronize()
            t_model_3 = time.time()
            t_model += torch.tensor([t_model_11 - t_model_1, t_model_2 - t_model_11, t_model_3 - t_model_2])
        inradius_finegrain_output_positions = torch.cat(inradius_finegrain_output_positions, dim=0)
        finegrain_output_positions[in_radius_idx] = inradius_finegrain_output_positions.cpu()
    torch.cuda.synchronize()
    t3 = time.time()
    ignore_distance_idx = torch.cumsum(finegrain_inside_batch_indices.unique(return_counts=True)[1], dim=0) - 1
    ignore_distance = torch.zeros(finegrain_taus_render.size(0), dtype=torch.bool, device=device)
    ignore_distance[ignore_distance_idx] = True


    colors_list = []
    inside_batch_indices_list = []


    for idx in range(math.ceil(len(finegrain_output_positions) / args.max_color_batch_size)):
        start_batch = idx * args.max_color_batch_size
        end_batch = (idx + 1) * args.max_color_batch_size
        colors, across_batch_indices = color_calculating_torch(finegrain_output_positions[start_batch:end_batch].to(device), finegrain_inside_batch_indices[start_batch:end_batch].to(device),
                                                               ignore_distance[start_batch:end_batch].to(device), finegrain_region_render[start_batch:end_batch].to(device))
        colors_list.append(colors)
        inside_batch_indices_list.append(across_batch_indices)
    tensor_colors = torch.cat(colors_list, dim=0)
    tensor_inside_batch_indices = torch.cat(inside_batch_indices_list, dim=0)

    nonzero_tensor_indices = tensor_colors.norm(dim=1) != 0
    nonzero_tensor_colors = tensor_colors[nonzero_tensor_indices]
    nonzero_tensor_inside_batch_indices = tensor_inside_batch_indices[nonzero_tensor_indices]

    nonzero_densities = torch.ones_like(nonzero_tensor_inside_batch_indices).to(torch.float32)
    transmittance = torch.ones(data_pairs.shape[0], dtype=torch.float32, device=device)
    t_start = time.time()

    unused_indices = torch.ones_like(nonzero_tensor_inside_batch_indices, dtype=torch.bool)
    while unused_indices.any():
        color_batch_indices, inverse_indices, counts = nonzero_tensor_inside_batch_indices[unused_indices].unique(
            return_inverse=True, return_counts=True)
        num_groups = color_batch_indices.size(0)
        init = torch.full((num_groups,), fill_value=nonzero_tensor_inside_batch_indices[unused_indices].size(0), dtype=torch.long, device=device)
        first_indices = init.scatter_reduce(0, inverse_indices, torch.arange(len(nonzero_tensor_inside_batch_indices[unused_indices])).to(device), 'amin', include_self=True)
        # first_indices = len(inverse_indices) - 1 - torch.zeros_like(color_batch_indices).scatter_(0, inverse_indices.flip(0), torch.arange(len(nonzero_tensor_inside_batch_indices[unused_indices]), device=device))
        current_first_colors = nonzero_tensor_colors[unused_indices][first_indices]
        current_first_densities = current_first_colors.norm(dim=1)
        unused_nonzero_densities = nonzero_densities[unused_indices]
        unused_nonzero_densities[first_indices] = current_first_densities * transmittance[color_batch_indices] * args.fake_dtau
        nonzero_densities[unused_indices] = unused_nonzero_densities
        transmittance[color_batch_indices] *= torch.exp(-current_first_densities * args.fake_dtau)
        might_use_indices = torch.zeros(unused_indices.sum(), dtype=torch.bool)
        might_use_indices[first_indices] = True
        unused_indices[unused_indices.clone()] = ~might_use_indices.to(device)
    t_end = time.time()
    print(f"transmittance time: {t_end - t_start}")

    colors1.put_(nonzero_tensor_inside_batch_indices, nonzero_tensor_colors[:, 0] * nonzero_densities, accumulate=True)
    colors2.put_(nonzero_tensor_inside_batch_indices, nonzero_tensor_colors[:, 1] * nonzero_densities, accumulate=True)
    colors3.put_(nonzero_tensor_inside_batch_indices, nonzero_tensor_colors[:, 2] * nonzero_densities, accumulate=True)

    ## sphere color
    valid_indices = sky_sphere_direction.abs().sum(-1) != 0
    valid_sky_sphere_direction = sky_sphere_direction[valid_indices]
    valid_batch_indices = torch.arange(data_pairs.size(0), device=device)[valid_indices]
    valid_transmittance = transmittance[valid_indices]
    valid_sky_sphere_direction_position = dir_to_2d_coord(valid_sky_sphere_direction, Ws, Hs)
    flatten_sky_sphere_indices = valid_sky_sphere_direction_position[:, 0] + valid_sky_sphere_direction_position[:, 1] * Ws
    sky_sphere_colors = flatten_sky_sphere_texture[flatten_sky_sphere_indices]
    densities = valid_transmittance * args.fake_dtau
    colors1.put_(valid_batch_indices, sky_sphere_colors[:, 0] * densities, accumulate=True)
    colors2.put_(valid_batch_indices, sky_sphere_colors[:, 1] * densities, accumulate=True)
    colors3.put_(valid_batch_indices, sky_sphere_colors[:, 2] * densities, accumulate=True)
    tensor_global_colors = torch.stack([colors1, colors2, colors3], dim=1).view(image_width, image_height, 3)
    t4 = time.time()
    print(f"points generating step time: {t2 - t1}, model time: {t3 - t2}, obtain color time: {t4 - t3}")
    print(f"model time: {t_model}")
    return tensor_global_colors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_name", type=str, default="100_1e-3radius_3stacklayer_multiple")# 100-30radius_6stacklayer_multiple 100_1e-3radius_3stacklayer_multiple
    parser.add_argument("--aabb_batch_size", type=int, default=1036800, help="the batch size for intersection stage")
    parser.add_argument("--color_batch_size", type=int, default=1036800, help="the batch size for color_stage")
    parser.add_argument("--cos_distance_ratio", type=float, default=0.8, help="the ratio of the obtained distance to the cosine distance")
    parser.add_argument("--constant_tau", type=float, default=0.06, help="the delta tau for calculating the direction")
    parser.add_argument("--fall_in_bh_coefficient", type=float, default=7.0, help="obtain the radius coefficient * bh_mass for judging whether a ray would fall in black hole in near-field")
    parser.add_argument("--max_radius", type=int, default=100, help="the maximum radius for the space, should be the same to that used in the model training")
    parser.add_argument("--image_dir", type=str, default="images", help="the directory to store the images")
    parser.add_argument("--render_tau_min_step", type=float, default=0.4, help="the minimum step for the tau in the rendering process")
    parser.add_argument("--render_coarse_max_num_step", type=float, default=10, help="the maximum number of sampling coarse points in a black hole region for a single ray")
    parser.add_argument("--render_finegrain_max_num_step", type=float, default=10, help="the maximum number of sampling finegrain points in a black hole region for a single ray")
    parser.add_argument("--max_color_batch_size", type=int, default=100000000, help="the maximum batch size for color calculating")
    parser.add_argument("--near_bh_distance", type=float, default=1.8, help="the distance to the black hole to be considered as inside the black hole")
    parser.add_argument("--min_radius", type=int, default=1.6, help="the minimum radius for the ray in black holes")
    parser.add_argument("--bh_radius_tolerance", type=float, default=10.0, help="the tolerance for the radius of the black hole")
    parser.add_argument("--fake_dtau", type=float, default=1.0, help="the fake delta tau for calculating transmittance on the ray")
    parser.add_argument("--render_tau_finegrain_min_step", type=float, default=0.1, help="the minimum step for the tau in the finegrain rendering process")
    parser.add_argument("--render_max_tau_interval", type=float, default=45., help="the maximum interval for the tau in the near black hole region")
    parser.add_argument("--loose_boundary", type=float, default=0.1,
                        help="a value taken into consider when calculating the near-field boundary")
    parser.add_argument("--norm_output", type=bool, default=False, help="whether to normalize the output position")
    args = parser.parse_args()

    bh_num = 3

    textures = []
    texture_dir = "texture"
    for i in range(1, 4):
        texture_path = f"{texture_dir}/texture_real{i}.png"
        textures.append(np.array(Image.open(texture_path)))
    textures = np.stack(textures, axis=0)

    sky_path = f"{texture_dir}/texture_sky.png"
    skysphere_texture = np.array(Image.open(sky_path))
    fov = 60
    aspect_ratio = 16 / 9
    image_width = 1920
    image_height = int(image_width / aspect_ratio)
    ti.init(arch=ti.cuda)
    canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
    colors = ti.Vector.field(3, dtype=ti.f32, shape=(image_width * image_height))
    bloom_img = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
    direction_fields = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch_accretion_texture = torch.from_numpy(textures.astype(np.float32) / 255.0).to(torch.float32).to(
        device)
    Ha, Wa = torch_accretion_texture.size(1), torch_accretion_texture.size(2)
    flatten_accretion_texture = torch_accretion_texture.view(-1, 3)

    torch_sky_sphere_texture = torch.from_numpy(skysphere_texture.astype(np.float32) / 255.0).to(torch.float32).to(
        device)

    Hs, Ws = torch_sky_sphere_texture.size(0), torch_sky_sphere_texture.size(1)
    flatten_sky_sphere_texture = torch_sky_sphere_texture.view(-1, 3)
    bh_render_radius = 20
    if bh_num == 2:
        pos_bh = torch.tensor([[-30., 0., 0.], [30., 0., 0.]], device=device)
        mass_bh = torch.tensor([1., 1.], device=device)
    elif bh_num == 3:
        a = 60
        pos_bh = torch.tensor(
            [[a / 2, a * math.sqrt(3) / 6, 0.], [-a / 2, a * math.sqrt(3) / 6, 0.], [0, -a * math.sqrt(3) / 3, 0.]], device=device)
        mass_bh = torch.tensor([1., 1., 1.], device=device)
        args.near_bh_distance = 2.25
        args.min_radius = 2.2
    else:
        raise Exception("Invalid bh_num")
    bh_taichi_pos = ti.Vector.field(3, dtype=ti.f32, shape=(pos_bh.size(0)))
    bh_taichi_pos.from_torch(pos_bh)
    camera = Camera(fov)
    gui = ti.GUI("Black Hole", res=(image_width, image_height), show_gui=False)
    canvas.fill(0)
    save_dir = os.path.join("checkpoint", args.log_name)
    start_epoch = 0
    models = []
    for i in range(pos_bh.shape[0] + 1):
        if i == 0:
            if bh_num == 2:
                # 2bh
                save_path = os.path.join(os.path.join("checkpoint", "2bh"), "newest.pth")
                state_dict = torch.load(save_path, map_location=device)
                model = GeodesicNet(num_blocks=6, in_dim=199, hidden_dim=128, inter_dim_factor=1, activation=F.softplus)
                model.load_state_dict(state_dict["model"])
            elif bh_num == 3:
                # 3bh
                save_path = os.path.join(
                    os.path.join("checkpoint", "3bh"),
                    "newest.pth")
                state_dict = torch.load(save_path, map_location=device)
                model = GeodesicNet(num_blocks=6, in_dim=199, hidden_dim=200, inter_dim_factor=1, activation=F.softplus)
                model.load_state_dict(state_dict["model"])
            else:
                raise Exception("Invalid bh_num")
        else:
            if bh_num == 2:
                # 2bh
                save_path = os.path.join(os.path.join("checkpoint", "2bh"), f"inner{i}.pth")
                state_dict = torch.load(save_path, map_location=device)
                model = GeodesicNet(num_blocks=6, in_dim=199, hidden_dim=200, inter_dim_factor=1, activation=F.softplus)
                model.load_state_dict(state_dict["model"])
            elif bh_num == 3:
                # 3bh
                save_path = os.path.join(os.path.join("checkpoint",
                                                      "3bh"),
                                         f"inner{i}.pth")
                state_dict = torch.load(save_path, map_location=device)
                model = GeodesicNet(num_blocks=6, in_dim=199, hidden_dim=200, inter_dim_factor=1, activation=F.softplus)
                model.load_state_dict(state_dict["model"])
            else:
                raise Exception("Invalid bh_num")
        model.to(device)
        models.append(model)

    canvas.fill(0)
    bloom_img.fill(0)
    look_from = torch.tensor([0., 60, 5.])
    look_at = torch.tensor([0., 0., 0.])
    camera.set_pos_dir(look_from, look_at)
    update_direction_fields()
    direction_tensor = direction_fields.to_torch(device=device)
    direction_tensor = direction_tensor.view(-1, 3)
    position_tensor = camera.lookfrom.to_torch(device=device).unsqueeze(0).repeat(direction_tensor.shape[0], 1)
    data_pairs, sky_sphere_direction = aabb_intersection_calculate(models, position_tensor, direction_tensor, pos_bh, bh_render_radius, args)
    tensor_colors = neural_render(models, data_pairs, sky_sphere_direction, args)
    canvas.from_torch(tensor_colors)
    gui.set_image(canvas.to_numpy())
    save_path = args.image_dir + f"/{bh_num}bh.png"
    gui.show(save_path)
