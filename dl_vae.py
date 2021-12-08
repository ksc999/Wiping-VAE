# Copyright 2019 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import pickle
import random
import time
import os
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from tqdm import trange


class VAE(nn.Module):
    def __init__(self, zsize, layer_count=3, channels=3):
        super(VAE, self).__init__()

        d = 128
        self.d = d
        self.zsize = zsize

        self.layer_count = layer_count

        mul = 1
        inputs = channels
        for i in range(self.layer_count):
            setattr(self, "conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2

        self.d_max = inputs

        self.fc1 = nn.Linear(inputs * 4 * 4, zsize)
        self.fc2 = nn.Linear(inputs * 4 * 4, zsize)

        self.d1 = nn.Linear(zsize, inputs * 4 * 4)

        mul = inputs // d // 2

        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))

    def encode(self, x):
        for i in range(self.layer_count):
            x = F.relu(getattr(self, "conv%d_bn" % (i + 1))(getattr(self, "conv%d" % (i + 1))(x)))

        x = x.view(x.shape[0], self.d_max * 4 * 4)
        h1 = self.fc1(x)
        h2 = self.fc2(x)
        return h1, h2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, x):
        x = x.view(x.shape[0], self.zsize)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, 4, 4)
        #x = self.deconv1_bn(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)

        x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        mu = mu.squeeze()
        logvar = logvar.squeeze()
        z = self.reparameterize(mu, logvar)
        return self.decode(z.view(-1, self.zsize, 1, 1)), mu, logvar

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


im_size = 128
Z_SIZE = 64


def loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x) ** 2)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1


def process_batch(batch, device):
    data = [x.reshape(im_size, im_size, 3).transpose((2, 0, 1)) for x in batch]

    x = torch.from_numpy(np.asarray(data, dtype=np.float32)).to(device) / 127.5 - 1.
    x = x.view(-1, 3, im_size, im_size)
    return x

def vae_inference(input, device):
    with torch.no_grad():
        input = input.reshape(1, 128*128*3)
        # input = input.reshape(1, -1)
        batch = process_batch(input, device)
        z_size = Z_SIZE
        vae = VAE(zsize=z_size, layer_count=5)
        vae.load_state_dict(torch.load('best_vae_face.pt'))
        vae.to(device)
        vae.eval()
        encode_vec, _ = vae.encode(batch)
        '''
        debug code
        '''
        # rec, mu, logvar = vae(batch)
        # # assert mu == encode_vec[0]
        # resultsample = (torch.cat([batch[0], rec[0]], dim=1) * 0.5 + 0.5).cpu().numpy().transpose((1, 2, 0))
        # # resultsample = resultsample.cpu()
        # plt.imshow(resultsample)
        # plt.show()
        return encode_vec[0].cpu().numpy()

def train_vae(train_dataset, val_dataset, device):
    batch_size = 512
    eval_batch_size = batch_size
    z_size = Z_SIZE
    vae = VAE(zsize=z_size, layer_count=5)
    vae.to(device)
    vae.train()
    vae.weight_init(mean=0, std=0.02)

    lr = 0.001

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

    train_epoch = 200

    sample1 = torch.randn(128, z_size).view(-1, z_size, 1, 1)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True,
                                         drop_last=False)

    # dataset_len = len(dataset)
    # train_data = dataset[:int(0.9*dataset_len)]
    # valid_data = dataset[int(0.9*dataset_len):]
    best_valid_loss = 100000
    for epoch in range(train_epoch):
        vae.train()

        print("Train set size:", len(train_dataset))

        # random.shuffle(train_data)
        # random.shuffle(valid_data)
        losses = []

        epoch_start_time = time.time()
        if epoch <= 6:
            if (epoch + 1) % 2 == 0:
                vae_optimizer.param_groups[0]['lr'] /= 2
                print("learning rate change!")
        elif epoch <= 15:
            if (epoch + 1) % 3 == 0:
                vae_optimizer.param_groups[0]['lr'] /= 2
                print("learning rate change!")
        elif epoch <= 31:
            if (epoch + 1) % 4 ==0:
                vae_optimizer.param_groups[0]['lr'] /= 2
                print("learning rate change!")
        else:
            if (epoch + 1) % 5 == 0:
                vae_optimizer.param_groups[0]['lr'] /= 2
                print("learning rate change!")
        for idx, batch in enumerate(train_dataloader):
            vae.train()
            # batch = np.array(train_data[idx:min(idx+batch_size, len(train_data))])
            # batch = process_batch(batch, device)
            # batch = batch.to(device)
            # vae.train()
            vae.zero_grad()
            batch = batch.to(device)
            rec, mu, logvar = vae(batch)
            loss_re, loss_kl = loss_function(rec, batch, mu, logvar)
            loss = loss_re + loss_kl
            loss.backward()
            vae_optimizer.step()
            losses.append(loss.item())
            with torch.no_grad():
                vae.eval()
                if (idx//batch_size) % 30 == 0:
                    print("step {} train loss: {}".format(idx//batch_size, np.mean(losses)))
                    losses = []
                    resultsample = (torch.cat([batch[0], rec[0]], dim=1) * 0.5 + 0.5)
                    resultsample = resultsample.cpu().numpy().transpose(1, 2, 0)
                    plt.imshow(resultsample)
                    plt.imsave('./generated_images/train_{}.png'.format(idx//batch_size), resultsample)
                    # plt.show()

            #############################################

            # os.makedirs('results_rec', exist_ok=True)
            # os.makedirs('results_gen', exist_ok=True)

        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
        # print("epoch{} training loss: {}".format(epoch, np.mean(losses)))
        print("taining time:", per_epoch_ptime)
        val_losses = []
        with torch.no_grad():
            vae.eval()
            eval_epoch_l = []
            for batch in val_dataloader:
                # batch = np.array(valid_data[i:min(i + eval_batch_size, len(valid_data))])
                # batch = process_batch(batch, device)
                batch = batch.to(device)
                x_rec, mu, logvar = vae(batch)
                loss_re, loss_kl = loss_function(x_rec, batch, mu, logvar)
                loss = loss_re + loss_kl
                eval_epoch_l.append(loss.item())
            if np.mean(eval_epoch_l) < best_valid_loss:
                print("save best model")
                torch.save(vae.state_dict(), "./dl_saved_models/best.pt")
            resultsample = (torch.cat([batch[0], x_rec[0]], dim=1) * 0.5 + 0.5)
            resultsample = resultsample.cpu().numpy().transpose(1,2,0)
            # plt.imshow(resultsample)
            plt.imsave('./generated_images/val_{}.png'.format(epoch), resultsample)
            # plt.show()
            print("epoch{} eval loss: {}".format(epoch, np.mean(eval_epoch_l)))
            torch.save(vae.state_dict(), "./dl_saved_models/checkpoint{}.pt".format(epoch))

            # save_image(resultsample.view(-1, 3, im_size, im_size),
            #            'results_rec/sample_' + str(epoch) + "_" + str(i) + '.png')
            # x_rec = vae.decode(sample1)
            # resultsample = x_rec * 0.5 + 0.5
            # resultsample = resultsample.cpu()
            # save_image(resultsample.view(-1, 3, im_size, im_size),
            #            'results_gen/sample_' + str(epoch) + "_" + str(i) + '.png')

    print("Training finish!...")
    # torch.save(vae.state_dict(), "VAEmodel.pkl")
    
if __name__ == '__main__':
    from dataset import MyDataset
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    train_dataset = MyDataset(data_feat='train', transform=transforms.ToTensor())
    val_dataset = MyDataset(data_feat='val', transform=transforms.ToTensor())
    train_vae(train_dataset, val_dataset, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))