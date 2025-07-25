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
import torch.nn.functional as F
import random
import time
import yaml

def main_worker(rank, world_size, black_hole_positions, args):
    # Initialize process group
    torch.cuda.set_device(rank)
    device = torch.device(rank)
    model = GeodesicNet(num_blocks=args.num_blocks, in_dim=199, hidden_dim=args.hidden_dim, inter_dim_factor=args.inter_dim_factor, drop_ratio=args.drop_ratio, activation=F.softplus)
    model.to(device)
    black_hole_pos = black_hole_positions[rank].to(device)
    with open('dataset.yaml', 'r', encoding='utf-8') as f:
        yaml_dir = yaml.load(f.read(), Loader=yaml.FullLoader)['dataset_dir']
    dataset_dir = os.path.join(yaml_dir, f"Kerr{world_size}BH/Multiple", "{}radius".format(20), f"Inner{rank + 1}")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    datasets = GeodesicDataset(dataset_dir)
    indices = list(range(len(datasets)))[:args.data_length]
    train_indices = indices[:math.ceil(args.train_ratio * len(indices))]
    print("dataset_dir", dataset_dir)
    print("dataset length", len(datasets))
    print("train indices", len(train_indices))
    test_indices = indices[int(args.train_ratio * len(indices)):]
    train_dataset = torch.utils.data.Subset(datasets, train_indices)
    test_dataset = torch.utils.data.Subset(datasets, test_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    total_steps = args.scheduler_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                           max_lr=args.lr,
                           total_steps=total_steps,
                           pct_start=0.3,
                           anneal_strategy='cos',
                           div_factor=5,
                           )
    if args.use_wandb:
        wandb.init(project="BlackHole", name=args.log_name + f"_Inner{rank+1}", config=args)
    save_dir = os.path.join("checkpoint", args.log_name)
    if not os.path.exists(save_dir) and rank == 0:
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"inner{rank+1}.pth")
    start_epoch = 0
    if os.path.exists(save_path):
        state_dict = torch.load(save_path)
        start_epoch = state_dict['epoch']
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict["scheduler"])
        print("load checkpoint at epoch", start_epoch)
    best_test_loss = 1e10
    for epoch in range(start_epoch, args.num_epoch):
        current_save_path = os.path.join(save_dir, f"inner{rank + 1}_{epoch}.pth")
        training_loss = 0
        position_losses = 0
        speed_losses = 0
        acc_losses = 0
        if epoch > args.speed_epoch:
            current_batch_size = args.speed_batch_size
        else:
            current_batch_size = args.batch_size

        current_radius = 8

        xyz_min = black_hole_pos.unsqueeze(0) - args.black_hole_radius
        xyz_max = black_hole_pos.unsqueeze(0) + args.black_hole_radius

        model.train()
        for data in train_loader:
            data = data.flatten(0, 2).to(device)
            length_ratio = 1
            actual_batch_size = int(current_batch_size / length_ratio)
            for batch_idx in range(math.ceil(data.shape[0] / actual_batch_size)):
                batch_data = data[batch_idx * actual_batch_size:(batch_idx + 1) * actual_batch_size]
                if length_ratio == 1:
                    init_sample_ids = torch.zeros(batch_data.size(0), dtype=torch.int64)
                else:
                    init_sample_ids = torch.randint(0, int(batch_data.size(1) * (1 - length_ratio)), (batch_data.size(0),))
                init_data = batch_data[torch.arange(init_sample_ids.shape[0]), init_sample_ids].unsqueeze(1)
                num_per_ray = int(batch_data.size(1) * length_ratio)
                picked_indices = []
                first_dimension_indices = []
                for i, init_sample_idx in enumerate(init_sample_ids):
                    picked_idx = torch.randperm(batch_data.size(1) - init_sample_idx)[:num_per_ray] + init_sample_idx
                    picked_indices.append(picked_idx)
                    first_dimension_indices.append(torch.ones_like(picked_idx) * i)
                picked_indices = torch.cat(picked_indices, dim=0)
                first_dimension_indices = torch.cat(first_dimension_indices, dim=0)

                init_p, direction = init_data[:, :, :3].repeat(1, num_per_ray, 1), init_data[:, :, 3:6].repeat(1, num_per_ray, 1)
                target_p, target_v, target_a, tau = batch_data[first_dimension_indices, picked_indices, :3], batch_data[first_dimension_indices, picked_indices, 3:6], batch_data[first_dimension_indices, picked_indices,
                                                                                                 6:9], (batch_data[:, :, 9:] - batch_data[torch.arange(init_sample_ids.shape[0]), init_sample_ids, 9:].unsqueeze(1))[first_dimension_indices, picked_indices]
                init_p, direction = init_p.flatten(0, 1), direction.flatten(0, 1)
                init_p = (init_p - xyz_min) / (xyz_max - xyz_min)
                assert target_p.all()
                encoded_init_p = positional_encoding(init_p)
                encoded_direction = positional_encoding(direction)
                if epoch > args.acc_epoch:
                    predict_p, speed, acceleration = calculate_speed_acc(model, encoded_init_p, encoded_direction, tau)
                    position_loss = torch.nn.functional.mse_loss(predict_p, target_p)
                    speed_loss = torch.nn.functional.mse_loss(speed, target_v)
                    norm_acceleration = acceleration / (acceleration.detach().norm(dim=1, keepdim=True) + 1e-9)
                    norm_target_a = target_a / (target_a.norm(dim=1, keepdim=True) + 1e-9)
                    acc_loss = torch.nn.functional.mse_loss(norm_acceleration, norm_target_a)
                    speed_losses += speed_loss.item() / math.ceil(data.shape[0] / actual_batch_size)
                    acc_losses += acc_loss.item() / math.ceil(data.shape[0] / actual_batch_size)
                    loss = position_loss + 800 * speed_loss + acc_loss
                elif epoch > args.speed_epoch:
                    if args.norm_output:
                        predict_p, speed = calculate_speed(model, encoded_init_p, encoded_direction, tau)
                        predict_p = black_hole_pos.unsqueeze(0) + predict_p
                    else:
                        predict_p, speed = calculate_speed(model, encoded_init_p, encoded_direction, tau)
                    position_loss = torch.nn.functional.mse_loss(predict_p, target_p)
                    speed_loss = torch.nn.functional.mse_loss(speed, target_v)
                    speed_losses += speed_loss.item() / math.ceil(data.shape[0] / actual_batch_size)
                    loss = position_loss + 800 * speed_loss
                else:
                    predict_p = model(torch.cat([encoded_init_p, encoded_direction, tau], dim=1))
                    if args.norm_output:
                        predict_p = black_hole_pos.unsqueeze(0) + predict_p
                    position_loss = torch.nn.functional.mse_loss(predict_p, target_p)
                    loss = position_loss
                assert not torch.isnan(loss).sum()
                position_losses += position_loss.item() / math.ceil(data.shape[0] / actual_batch_size)
                training_loss += loss.item() / math.ceil(data.shape[0] / actual_batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch < args.scheduler_epochs:
                scheduler.step()
        training_loss /= len(train_loader)
        position_losses /= len(train_loader)
        speed_losses /= len(train_loader)
        acc_losses /= len(train_loader)
        print(training_loss)

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
                    if args.norm_output:
                        predict_p = black_hole_pos.unsqueeze(0) + predict_p
                    position_loss = torch.nn.functional.mse_loss(predict_p, target_p)

                test_loss += position_loss.item() / math.ceil(data.shape[0] / current_batch_size)
        test_loss /= len(test_loader)
        print(f"Epoch {epoch}, training loss: {training_loss}, test loss: {test_loss}")
        save_dict = {}
        save_dict['epoch'] = epoch + 1
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_dict['model'] = model.state_dict()
        else:
            save_dict['model'] = torch.load(save_path)['model']
        save_dict['scheduler'] = scheduler.state_dict()
        save_dict['optimizer'] = optimizer.state_dict()
        torch.save(save_dict, save_path)
        torch.save(save_dict, current_save_path)
        if args.use_wandb:
            wandb.log({"training_loss": training_loss, "training position loss": position_losses,
                       "training speed loss": speed_losses, "training acceleration loss": acc_losses,
                       "test_loss": test_loss, "epoch": epoch, "current_radius": current_radius})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--num_epoch", type=int, default=8)
    parser.add_argument("--speed_batch_size", type=int, default=900)
    parser.add_argument("--batch_size", type=int, default=3600)
    parser.add_argument("--use_wandb", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="checkpoint")
    parser.add_argument("--log_name", type=str, default="near_field_training")
    parser.add_argument("--speed_epoch", type=int, default=-1)
    parser.add_argument("--max_radius", type=int, default=100)
    parser.add_argument("--acc_epoch", type=int, default=10000)
    parser.add_argument("--master_port", default="12366")
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    parser.add_argument("--data_length", type=int, default=1000, help="The length of the data used for training")
    parser.add_argument("--inter_dim_factor", type=int, default=1, help="The factor of the intermediate dimension")
    parser.add_argument("--black_hole_radius", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=200)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--norm_output", type=bool, default=False, help="Whether to normalize the output")
    parser.add_argument("--scheduler_epochs", type=int, default=5)
    args = parser.parse_args()
    args.milestones = [int(1/3 * args.num_epoch), int(2/3 * args.num_epoch), int(5/6 * args.num_epoch)]
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.master_port
    # black_hole_positions = torch.tensor([[-30., 0., 0.], [30., 0., 0.]])
    a = 60
    black_hole_positions = torch.tensor(
        [[a / 2, a * math.sqrt(3) / 6, 0.], [-a / 2, a * math.sqrt(3) / 6, 0.], [0, -a * math.sqrt(3) / 3, 0.]])
    world_size = black_hole_positions.shape[0]
    mp.spawn(main_worker, nprocs=world_size, args=(world_size, black_hole_positions, args))
