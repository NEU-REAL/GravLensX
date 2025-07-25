import os
import math
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from model.pinn import GeodesicNet, calculate_speed, calculate_speed_acc, positional_encoding
from model.dataset import GeodesicDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
import random
import numpy as np
import yaml

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True


def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt)
    rt /= world_size
    return rt.item()

def xyz_to_spherical(batch_xyz, bh_position):
    x = batch_xyz[:, 0] - bh_position[0]
    y = batch_xyz[:, 1] - bh_position[1]
    z = batch_xyz[:, 2] - bh_position[2]
    r = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    epsilon = 1e-9
    theta = torch.acos(z / (r + epsilon))
    phi = torch.atan2(y, x)
    spherical_coords = torch.stack([r, theta, phi], dim=1)
    return spherical_coords


def spherical_to_xyz(batch_spherical, bh_position):
    r = batch_spherical[:, 0]
    theta = batch_spherical[:, 1]
    phi = batch_spherical[:, 2]
    x = r * torch.sin(theta) * torch.cos(phi) + bh_position[0]
    y = r * torch.sin(theta) * torch.sin(phi) + bh_position[1]
    z = r * torch.cos(theta) + bh_position[2]
    cartesian_coords = torch.stack([x, y, z], dim=1)
    return cartesian_coords

def main_worker(rank, world_size, args):
    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(rank)

    if args.use_wandb and rank == 0:
        wandb.init(project="BlackHole", name=args.log_name, config=args)
    # Initialize model and move it to the GPU
    model = GeodesicNet(num_blocks=args.num_blocks, in_dim=199, hidden_dim=args.hidden_dim, inter_dim_factor=args.inter_dim_factor, drop_ratio=args.drop_ratio).to(device)
    model = DDP(model, device_ids=[rank])

    # Load dataset and create distributed samplers
    with open('dataset.yaml', 'r', encoding='utf-8') as f:
        yaml_dir = yaml.load(f.read(), Loader=yaml.FullLoader)['dataset_dir']
    dataset_dir = os.path.join(yaml_dir, f"Kerr{args.num_bh}BH/Multiple", "{}radius".format(20), "Outer")
    if not os.path.exists(dataset_dir) and rank == 0:
        os.makedirs(dataset_dir)
    datasets = GeodesicDataset(dataset_dir)
    indices = list(range(len(datasets)))[:args.data_length]
    train_indices = indices[:math.ceil(args.train_ratio * len(indices))]
    test_indices = indices[int(args.train_ratio * len(indices)):]
    train_dataset = torch.utils.data.Subset(datasets, train_indices)
    test_dataset = torch.utils.data.Subset(datasets, test_indices)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, sampler=test_sampler, num_workers=2)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.scheduler_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                           max_lr=args.lr,  # your peak LR, e.g. 1e-3
                           total_steps=total_steps,
                           pct_start=0.3,  # 30% ramp-up
                           anneal_strategy='cos',
                           div_factor=5,
                           )
    # Load model and optimizer state if checkpoint exists
    save_dir = os.path.join("checkpoint", args.log_name)
    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, "newest.pth")
    start_epoch = 0
    if os.path.exists(save_path):
        state_dict = torch.load(save_path, map_location=device)
        start_epoch = state_dict['epoch']
        model.module.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])
        print("load checkpoint from epoch {}".format(start_epoch))
    best_test_loss = 1e10
    current_batch_size = args.batch_size // world_size
    for epoch in range(start_epoch, args.num_epoch):
        current_save_path = os.path.join(save_dir, "newest_{}.pth".format(epoch))
        train_sampler.set_epoch(epoch)
        training_loss = 0
        position_losses = 0
        speed_losses = 0

        xyz_min = -args.max_radius
        xyz_max = args.max_radius

        model.train()
        for data in train_loader:
            data = data.flatten(0, 2).to(device)
            for batch_idx in range(math.ceil(data.shape[0] / current_batch_size)):
                batch_data = data[batch_idx * current_batch_size:(batch_idx + 1) * current_batch_size]

                init_data = batch_data[:, 0:1]
                init_p, direction = init_data[:, :, :3].repeat(1, batch_data.size(1), 1), init_data[:, :, 3:6].repeat(1,
                                                                                                                      batch_data.size(
                                                                                                                          1),
                                                                                                                      1)
                target_p, target_v, target_a, tau = batch_data[:, :, :3], batch_data[:, :, 3:6], batch_data[:, :,
                                                                                                 6:9], batch_data[:, :,
                                                                                                       9:]
                init_p, direction, target_p, target_v, target_a, tau = (init_p.flatten(0, 1), direction.flatten(0, 1),
                                                                        target_p.flatten(0, 1), target_v.flatten(0, 1),
                                                                        target_a.flatten(0, 1), tau.flatten(0, 1))

                init_p = (init_p - xyz_min) / (xyz_max - xyz_min)
                assert target_p.all()
                encoded_init_p = positional_encoding(init_p)
                encoded_direction = positional_encoding(direction)


                predict_p, speed = calculate_speed(model, encoded_init_p, encoded_direction, tau)
                position_loss = torch.nn.functional.mse_loss(predict_p, target_p)
                speed_loss = torch.nn.functional.mse_loss(speed, target_v)
                speed_losses += speed_loss.detach() / math.ceil(data.shape[0] / current_batch_size)
                loss = position_loss + 800 * speed_loss

                assert not torch.isnan(loss).sum()
                position_losses += position_loss.detach() / math.ceil(data.shape[0] / current_batch_size)
                training_loss += loss.detach() / math.ceil(data.shape[0] / current_batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch < args.scheduler_epochs:
                scheduler.step()
        training_loss /= len(train_loader)
        position_losses /= len(train_loader)
        speed_losses /= len(train_loader)
        training_loss = reduce_mean(training_loss, world_size)
        position_losses = reduce_mean(position_losses, world_size)
        speed_losses = reduce_mean(speed_losses, world_size)

        model.eval()
        test_loss = 0

        for data in test_loader:
            data = data.flatten(0, 2).to(device)
            for batch_idx in range(math.ceil(data.shape[0] / current_batch_size)):
                batch_data = data[batch_idx * current_batch_size:(batch_idx + 1) * current_batch_size]
                init_data = batch_data[:, 0:1]
                init_p, direction = init_data[:, :, :3].repeat(1, batch_data.size(1), 1), init_data[:, :, 3:6].repeat(1,
                                                                                                                      batch_data.size(
                                                                                                                          1),
                                                                                                                      1)
                target_p, target_v, target_a, tau = batch_data[:, :, :3], batch_data[:, :, 3:6], batch_data[:, :,
                                                                                                 6:9], batch_data[:, :,
                                                                                                       9:]
                init_p, direction, target_p, target_v, target_a, tau = (init_p.flatten(0, 1), direction.flatten(0, 1),
                                                                        target_p.flatten(0, 1), target_v.flatten(0, 1),
                                                                        target_a.flatten(0, 1), tau.flatten(0, 1))
                init_p = (init_p - xyz_min) / (xyz_max - xyz_min)
                encoded_init_p = positional_encoding(init_p)
                encoded_direction = positional_encoding(direction)
                with torch.no_grad():
                    predict_p = model(torch.cat([encoded_init_p, encoded_direction, tau], dim=1))
                    position_loss = torch.nn.functional.mse_loss(predict_p, target_p)
                test_loss += position_loss / math.ceil(data.shape[0] / current_batch_size)
        test_loss /= len(test_loader)
        test_loss = reduce_mean(test_loss, world_size)
        print(f"Epoch {epoch}, training loss: {training_loss}, test loss: {test_loss}")
        save_dict = {}
        save_dict['epoch'] = epoch + 1
        if rank == 0:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_dict['model'] = model.module.state_dict()
            else:
                save_dict['model'] = torch.load(save_path)['model']
            save_dict['scheduler'] = scheduler.state_dict()
            save_dict['optimizer'] = optimizer.state_dict()
            torch.save(save_dict, save_path)
            torch.save(save_dict, current_save_path)
        if args.use_wandb and rank == 0:
            wandb.log({"training_loss": training_loss, "training position loss": position_losses,
                       "training speed loss": speed_losses, "test_loss": test_loss, "epoch": epoch})

    dist.destroy_process_group()



if __name__ == '__main__':
    data_length = 1000
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_epoch", type=int, default=25 * int(1000 / data_length))
    parser.add_argument("--batch_size", type=int, default=3000)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--save_dir", type=str, default="checkpoint")
    parser.add_argument("--log_name", type=str, default="far_field_training")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--speed_epoch", type=int, default=-1)
    parser.add_argument("--max_radius", type=int, default=100)
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    parser.add_argument("--milestones", default=[10 * int(1000 / data_length), 15 * int(1000 / data_length), 16 * int(1000 / data_length)])
    parser.add_argument("--master_port", default="12321")
    parser.add_argument("--data_length", type=int, default=data_length, help="The length of the data used for training")
    parser.add_argument("--inter_dim_factor", type=int, default=1, help="The factor of the intermediate dimension")
    parser.add_argument("--hidden_dim", type=int, default=200, help="The dimension of the hidden layer")
    parser.add_argument("--black_hole_radius", type=int, default=20)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--scheduler_epochs", type=int, default=20)
    parser.add_argument("--num_bh", type=int, default=3, help="The number of black holes in the dataset")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port
    seed_everything(args.seed)
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
