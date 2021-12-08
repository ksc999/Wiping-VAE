from dataset import MyDataset
from models import BetaVAE
from PIL import Image
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import time
import yaml
import argparse
from loguru import logger
import os

from torch.utils.tensorboard import SummaryWriter


class beta_VAE_Pretrain:

    def __init__(self):
        self.parse_args()
        logger.add('./logs/{time}.log')
        logger.debug(self.args)
        time_index = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.writer = SummaryWriter(log_dir='./runs/' + time_index)

        self.transform = transforms.ToTensor()
        self.train_dataset = MyDataset(data_feat='train', transform=self.transform)
        self.val_dataset = MyDataset(data_feat='val', transform=self.transform)
        self.test_dataset = MyDataset(data_feat='test', transform=self.transform)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                           drop_last=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=True,
                                         drop_last=False)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True,
                                          drop_last=False)

        self.model = BetaVAE(in_channels=3, latent_dim=128)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.args.milestones)

        self.early_stop_queue = []
        self.early_stop_queue_cnt = 0
        self.early_stop_epoch_threshold = self.args.early_stop_threshold

        self.lowest_val_loss = float('inf')
        self.save_model_path = os.getcwd() + '/saved_models/' + time_index
        os.makedirs(self.save_model_path)

    def parse_args(self):
        parser = argparse.ArgumentParser(description='VAE in Wiping Project')
        parser.add_argument('--dataset_type', type=str, default='circle')
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--epochs', type=int, default=1000)
        parser.add_argument('--store_best_model_epoch_interval', type=int, default=1)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--milestones', type=int, nargs='+', default=[20, 40])
        parser.add_argument('--early_stop_tolerance', type=int, default=5)
        parser.add_argument('--early_stop_threshold', type=int, default=10)
        self.args = parser.parse_args()

    def train(self):
        for epoch in range(self.args.epochs):
            self.model.train()
            losses = []
            for data in self.train_dataloader:
                data = data.to(self.device)
                output = self.model(data)
                loss = self.model.loss_function(*output, M_N=0.005)['loss']
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
            train_loss_mean = np.mean(losses)
            self.writer.add_scalar('train/loss', train_loss_mean, epoch)
            self.scheduler.step()
            val_loss_mean = self.val(epoch)
            if epoch % self.args.store_best_model_epoch_interval == self.args.store_best_model_epoch_interval - 1:
                # if val_loss_mean <= self.lowest_val_loss:
                if True:
                    torch.save(self.model.state_dict(), self.save_model_path + '/best_model.pth')
                    # self.lowest_val_loss = val_loss_mean
                    logger.debug(f'epoch: {epoch} Best model!')
            self.writer.add_scalar('val/loss', val_loss_mean, epoch)
            logger.debug(f'epoch: {epoch}, train_loss: {train_loss_mean}, val_loss: {val_loss_mean}')

    def val(self, epoch):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for data in self.val_dataloader:
                data = data.to(self.device)
                output = self.model(data)
                loss = self.model.loss_function(*output, M_N=0.005)['loss']
                losses.append(loss.item())
        val_mean_loss = np.mean(losses)
        if epoch > self.early_stop_epoch_threshold:
            if self.early_stop_queue_cnt < self.args.early_stop_tolerance:
                self.early_stop_queue.append(val_mean_loss)
                self.early_stop_queue_cnt += 1
            else:
                self.early_stop_queue = self.early_stop_queue[1:]  # bug fixed
                self.early_stop_queue.append(val_mean_loss)
                dif = np.diff(np.array(self.early_stop_queue))
                if np.all(dif > 0):
                    logger.debug('Early stop!')
                    raise Exception('Early stop!')
        return val_mean_loss

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def visualized_res(self, pic_path=None):
        if pic_path is None:
            path = os.getcwd() + '/visual_res/'
        else:
            path = os.getcwd() + '/visual_res/' + pic_path
        # train
        train_list = self.train_dataset.sample_pathes
        idx = np.random.randint(0, len(train_list))
        to_tensor = transforms.ToTensor()
        train_image = Image.open(train_list[idx]).convert('RGB')

        train_input = to_tensor(train_image).unsqueeze(0)
        train_input = train_input.to(self.device)
        train_rec = self.model.generate(train_input)

        to_image = transforms.ToPILImage()
        train_rec_image = to_image(train_rec.squeeze(0))
        train_image.save(path + 'train.jpg')
        train_rec_image.save(path + 'train_rec.jpg')

        # test
        test_list = self.test_dataset.sample_pathes
        idx = np.random.randint(0, len(test_list))
        to_tensor = transforms.ToTensor()
        test_image = Image.open(test_list[idx]).convert('RGB')

        test_input = to_tensor(test_image).unsqueeze(0)
        test_input = test_input.to(self.device)
        test_rec = self.model.generate(test_input)

        to_image = transforms.ToPILImage()
        test_rec_image = to_image(test_rec.squeeze(0))
        test_image.save(path + 'test.jpg')
        test_rec_image.save(path + 'test_rec.jpg')


if __name__ == '__main__':
    wiping_vae_pretrain = beta_VAE_Pretrain()
    # wiping_vae_pretrain.train()
    time_step = '2021-12-08_14-29-12'
    wiping_vae_pretrain.load_model(os.getcwd() + '/saved_models/' + time_step + '/best_model.pth')
    # wiping_vae_pretrain.train()
    wiping_vae_pretrain.visualized_res()

    
    