import torch
from utils.dataset import CustomerDataset, CustomerCollate
from torch.utils.data import DataLoader
import torch.nn.parallel.data_parallel as parallel
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import sys
import time
import copy

from models.generator import Generator
from models.discriminator import Discriminator
from tensorboardX import SummaryWriter
from utils.optimizer import Optimizer
from utils.audio import hop_length
from utils.loss import MultiResolutionSTFTLoss

def create_model(args):

    generator = Generator(args.local_condition_dim, args.z_dim)
    #discriminator = Multiple_Random_Window_Discriminators(args.local_condition_dim)
    discriminator = Discriminator()

    return generator, discriminator

def save_checkpoint(args, generator, discriminator,
                    g_optimizer, d_optimizer, step, ema=None):
    checkpoint_path = os.path.join(args.checkpoint_dir, "model.ckpt-{}.pt".format(step))

    torch.save({"generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "g_optimizer": g_optimizer.state_dict(),
                "d_optimizer": d_optimizer.state_dict(),
                "global_step": step
                }, checkpoint_path)

    print("Saved checkpoint: {}".format(checkpoint_path))

    with open(os.path.join(args.checkpoint_dir, 'checkpoint'), 'w') as f:
        f.write("model.ckpt-{}.pt".format(step))

def attempt_to_restore(generator, discriminator, g_optimizer,
                       d_optimizer, checkpoint_dir, use_cuda, style_diff=False):
    checkpoint_list = os.path.join(checkpoint_dir, 'checkpoint')

    if os.path.exists(checkpoint_list):
        checkpoint_filename = open(checkpoint_list).readline().strip()
        checkpoint_path = os.path.join(
            checkpoint_dir, "{}".format(checkpoint_filename))
        print("Restore from {}".format(checkpoint_path))
        checkpoint = load_checkpoint(checkpoint_path, use_cuda)
        if style_diff:
            generator.load_state_dict(para_state_dict(generator, checkpoint["generator"]))
            discriminator.load_state_dict(para_state_dict(discriminator, checkpoint["discriminator"]))
        else:
            generator.load_state_dict(checkpoint["generator"])
            discriminator.load_state_dict(checkpoint["discriminator"])

        if not style_diff:
            g_optimizer.load_state_dict(checkpoint["g_optimizer"])
            d_optimizer.load_state_dict(checkpoint["d_optimizer"])

        global_step = checkpoint["global_step"]

    else:
        global_step = 0

    return global_step


def para_state_dict(model, loaded_paras):
    state_dict = copy.deepcopy(model.state_dict())
    for key in state_dict:  # 在新的网络模型中遍历对应参数
        if key in loaded_paras and state_dict[key].size() == loaded_paras[key].size():
            # print("成功初始化参数:", key)
            state_dict[key] = loaded_paras[key]
    return state_dict


def load_checkpoint(checkpoint_path, use_cuda):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)

    return checkpoint

def train(args):

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_dataset = CustomerDataset(
         args.input_train,
         local_condition=True)

    valid_dataset = CustomerDataset(
         args.input_valid,
         local_condition=True)

    device = torch.device("cuda" if args.use_cuda else "cpu")
    generator, discriminator = create_model(args)

    print(generator)
    print(discriminator)

    num_gpu = torch.cuda.device_count() if args.use_cuda else 1

    global_step = 0

    g_parameters = list(generator.parameters())
    d_parameters = list(discriminator.parameters())

    g_optimizer = optim.Adam(g_parameters, lr=args.g_learning_rate)

    # d_parameters = list(discriminator.discriminator.parameters())
    d_optimizer = optim.Adam(d_parameters, lr=args.d_learning_rate)

    train_writer_dir = os.path.join(args.checkpoint_dir, "train")
    valid_writer_dir = os.path.join(args.checkpoint_dir, "valid")

    train_writer = SummaryWriter( train_writer_dir )
    valid_writer = SummaryWriter( valid_writer_dir )

    generator.to(device)
    discriminator.to(device)

    if args.resume is not None:
        restore_step = attempt_to_restore(generator, discriminator, g_optimizer,
                                          d_optimizer, args.resume, args.use_cuda, style_diff=False)
        global_step = restore_step

    customer_g_optimizer = Optimizer(g_optimizer, args.g_learning_rate,
                global_step, args.warmup_steps, args.decay_learning_rate)
    customer_d_optimizer = Optimizer(d_optimizer, args.d_learning_rate,
                global_step, args.warmup_steps, args.decay_learning_rate)

    stft_criterion = MultiResolutionSTFTLoss().to(device)
    criterion = nn.MSELoss().to(device)

    for epoch in range(args.epochs):

        collate = CustomerCollate()

        train_data_loader = DataLoader(train_dataset, collate_fn=collate,
                batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=True, pin_memory=True)

        #train one epoch
        generator.train()
        discriminator.train()

        for batch, (samples, conditions) in enumerate(train_data_loader):

            start = time.time()
            batch_size = int(conditions.shape[0] // num_gpu * num_gpu)

            # samples.shape == [batch_size, 1, 67500]
            # conditons.shape == [batch_size, 768, 5]
            # z.shape == [batch_size, 128]
            samples = samples[:batch_size, :].to(device)
            conditions = conditions[:batch_size, :, :].to(device)
            # z = torch.randn(batch_size, args.z_dim).to(device)
            z = discriminator(samples, True).detach()

            losses = {}

            if num_gpu > 1:
                g_outputs = parallel(generator, (conditions, z))
            else:
                # g_outputs.shape == [batch_size, 1, 67500]
                g_outputs = generator(conditions, z)

            #train discriminator
            if global_step > args.discriminator_train_start_steps:
                if num_gpu > 1:
                    real_output = parallel(discriminator, (samples, ))
                    fake_output = parallel(discriminator, (g_outputs.detach(), ))
                else:
                    real_output = discriminator(samples, )
                    fake_output = discriminator(g_outputs.detach(), )

                fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
                real_loss = criterion(real_output, torch.ones_like(real_output))

                d_loss = fake_loss + real_loss

                customer_d_optimizer.zero_grad()
                d_loss.backward()
                nn.utils.clip_grad_norm_(d_parameters, max_norm=0.5)
                customer_d_optimizer.step_and_update_lr()
            else:
                d_loss = torch.Tensor([0])
                fake_loss = torch.Tensor([0])
                real_loss = torch.Tensor([0])

            losses['fake_loss'] = fake_loss.item()
            losses['real_loss'] = real_loss.item()
            losses['d_loss'] = d_loss.item()

            #train generator
            if num_gpu > 1:
                fake_output = parallel(discriminator, (g_outputs, ))
            else:
                fake_output = discriminator(g_outputs)

            adv_loss = criterion(fake_output, torch.ones_like(fake_output))

            sc_loss, mag_loss = stft_criterion(g_outputs.squeeze(1), samples.squeeze(1))

            if global_step > args.discriminator_train_start_steps:
               g_loss = adv_loss * args.lamda_adv + sc_loss + mag_loss
            else:
               g_loss = sc_loss + mag_loss

            losses['adv_loss'] = adv_loss.item()
            losses['sc_loss'] = sc_loss
            losses['mag_loss'] = mag_loss
            losses['g_loss'] = g_loss.item()

            customer_g_optimizer.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(g_parameters, max_norm=0.5)
            customer_g_optimizer.step_and_update_lr()

            time_used = time.time() - start
            if global_step > args.discriminator_train_start_steps:
                print("Step: {} --adv_loss: {:.3f} --real_loss: {:.3f} --fake_loss: {:.3f} --sc_loss: {:.3f} --mag_loss: {:.3f} --Time: {:.2f} seconds".format(
                   global_step, adv_loss, real_loss, fake_loss, sc_loss, mag_loss, time_used))
            else:
                print("Step: {} --sc_loss: {:.3f} --mag_loss: {:.3f} --Time: {:.2f} seconds".format(global_step, sc_loss, mag_loss, time_used))

            global_step += 1

            if global_step % args.checkpoint_step == 0:
                save_checkpoint(args, generator, discriminator,
                         g_optimizer, d_optimizer, global_step)

            if global_step % args.summary_step == 0:
                for key in losses:
                    train_writer.add_scalar('{}'.format(key), losses[key], global_step)

        # validation per epoch
        valid_data_loader = DataLoader(valid_dataset, collate_fn=collate,
               batch_size=args.batch_size, num_workers=args.num_workers,
               shuffle=True, pin_memory=True)

        # valid losses
        valid_losses = {}
        valid_losses['fake_loss'] = []
        valid_losses['real_loss'] = []
        valid_losses['d_loss'] = []
        valid_losses['adv_loss'] = []
        valid_losses['sc_loss'] = []
        valid_losses['mag_loss'] = []
        valid_losses['g_loss'] = []
        start = time.time()
        # 模型开启valid
        generator.eval()
        discriminator.eval()

        with torch.no_grad():
            for batch, (samples, conditions) in enumerate(valid_data_loader):
                batch_size = int(conditions.shape[0] // num_gpu * num_gpu)

                samples = samples[:batch_size, :].to(device)
                conditions = conditions[:batch_size, :, :].to(device)

                z = discriminator(samples, True).detach()

                if num_gpu > 1:
                    g_outputs = parallel(generator, (conditions, z))
                else:
                    g_outputs = generator(conditions, z)

                if global_step > args.discriminator_train_start_steps:
                    if num_gpu > 1:
                        real_output = parallel(discriminator, (samples, ))
                        fake_output = parallel(discriminator, (g_outputs.detach(), ))
                    else:
                        real_output = discriminator(samples, )
                        fake_output = discriminator(g_outputs.detach(), )

                    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
                    real_loss = criterion(real_output, torch.ones_like(real_output))

                    d_loss = fake_loss + real_loss

                else:
                    d_loss = torch.Tensor([0])
                    fake_loss = torch.Tensor([0])
                    real_loss = torch.Tensor([0])

                valid_losses['fake_loss'].apend(fake_loss.item())
                valid_losses['real_loss'].apend(real_loss.item())
                valid_losses['d_loss'].apend(d_loss.item())


                if num_gpu > 1:
                    fake_output = parallel(discriminator, (g_outputs, ))
                else:
                    fake_output = discriminator(g_outputs)

                adv_loss = criterion(fake_output, torch.ones_like(fake_output))

                sc_loss, mag_loss = stft_criterion(g_outputs.squeeze(1), samples.squeeze(1))

                if global_step > args.discriminator_train_start_steps:
                    g_loss = adv_loss * args.lamda_adv + sc_loss + mag_loss
                else:
                    g_loss = sc_loss + mag_loss

                    valid_losses['adv_loss'].apend(adv_loss.item())
                    valid_losses['sc_loss'].apend(sc_loss)
                    valid_losses['mag_loss'].apend(mag_loss)
                    valid_losses['g_loss'].apend(g_loss.item())


        # 对各项指标求平均
        for key, value in valid_losses:
            valid_losses[key] = sum(value)/(len(value))

        time_used = time.time() - start
        if global_step > args.discriminator_train_start_steps:
            print("Valid Epoch: {} --adv_loss: {:.3f} --real_loss: {:.3f} --fake_loss: {:.3f} --sc_loss: {:.3f} --mag_loss: {:.3f} --Time: {:.2f} seconds".format(
                epoch+1, valid_losses['adv_loss'], valid_losses['real_loss'], valid_losses['fake_loss'], valid_losses['sc_loss'], valid_losses['mag_loss'], time_used))
        else:
            print("Valid Epoch: {} --sc_loss: {:.3f} --mag_loss: {:.3f} --Time: {:.2f} seconds".format(epoch+1, valid_losses['sc_loss'], valid_losses['mag_loss'], time_used))


        for key in valid_losses:
            valid_writer.add_scalar('{}'.format(key), valid_losses[key], epoch+1)

def main():

    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train', type=str, default='data/train', help='Directory of training data')
    parser.add_argument('--input_valid', type=str, default='data/test', help='Directory of validation data')
    parser.add_argument('--num_workers',type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--checkpoint_dir', type=str, default="logdir", help="Directory to save model")
    parser.add_argument('--resume', type=str, default=None, help="The model name to restore")
    parser.add_argument('--checkpoint_step', type=int, default=5000)
    parser.add_argument('--summary_step', type=int, default=100)
    parser.add_argument('--use_cuda', type=_str_to_bool, default=True)
    parser.add_argument('--g_learning_rate', type=float, default=0.0001)
    parser.add_argument('--d_learning_rate', type=float, default=0.0001)
    parser.add_argument('--warmup_steps', type=int, default=20000)
    parser.add_argument('--decay_learning_rate', type=float, default=0.5)
    parser.add_argument('--local_condition_dim', type=int, default=768)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--condition_window', type=int, default=100)
    parser.add_argument('--lamda_adv', type=float, default=4.0)
    parser.add_argument('--discriminator_train_start_steps', type=int, default=10000)

    # next line for debug
    # sys.argv = ['train.py', '--input=data/train', '--use_cuda=false']

    # finetune
    sys.argv = ['train.py', '--use_cuda=false', '--batch_size=2', '--discriminator_train_start_step=10', '--checkpoint_step=20']
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
