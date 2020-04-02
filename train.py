import os
import argparse

parser = argparse.ArgumentParser(description="choose dataset")
# parser.add_argument('-n', '--dataset_name', default='nbi')
parser.add_argument('-g', '--gpu', default='0')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import torch
from model import inceptionv4
import data
import typing
from typing import Tuple, List
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

print(torch.cuda.is_available())
pre_trained = r'/share/jqshen/checkpoints/inceptionv4-8e4777a0.pth'
# dataset_name = args.dataset_name
dataset_dir = '/share/jqshen/Dataset'
model_save_path = '/share/jqshen/checkpoints/inception'


class Trainer(object):
    def __init__(self, dataset_dir: str, subdir: str, batch_size: int, max_epoch: int, model_save_path: str,
                 pre_trained: str, is_continue: bool):
        '''
        :param dataset_dir: where the images are stored.
               *********** Contains folders with the classnames
        :param phase: 'train' or 'val'
        :param subdir: 'nbi' or 'white'
        :param batch_size: batch size
        :param subdir:
        :param max_epoch:
        :param model_save_path:
        '''
        self.model, self.input_shape = inceptionv4.inceptionv4(num_classes=2, pre_trained=pre_trained)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.dataloader = {'train': data.mydataloader(dataset_dir=dataset_dir, phase='train', subdir=subdir,
                                                      input_shape=self.input_shape, batch_size=batch_size),
                           'val': data.mydataloader(dataset_dir=dataset_dir, phase='val', subdir=subdir,
                                                    input_shape=self.input_shape, batch_size=batch_size)}
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.lr_scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.save_path = model_save_path + f'{subdir}.tar'
        if is_continue:
            if os.path.isfile(self.save_path):
                self.model.load(torch.load(self.save_path))
        self.store = {'train': {'Loss': [], 'Accuracy': []},
                      'val': {'Loss': [], 'Accuracy': []}}
        self.phase = 'train'
        self.max_epoch = max_epoch
        self.subdir = subdir

    def iterate(self):
        correct_num = 0
        for (images, labels, imgpaths) in tqdm(self.dataloader[self.phase], total=len(self.dataloader[self.phase])):
            images = images.to(self.device).float()
            labels = labels.long().squeeze().to(self.device)
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            if self.phase == 'train':
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            with torch.no_grad():
                pred = torch.argmax(torch.softmax(logits, dim=1)).squeeze()
                correct_num += torch.sum(pred * labels.squeeze()).item()
        acc = correct_num / len(self.dataloader[self.phase])
        self.store[self.phase]['Accuracy'].append(acc)
        self.store[self.phase]['Loss'].append(loss.cpu().item())

    def run(self):
        for epoch in range(self.max_epoch):
            self.phase = 'train'
            self.iterate()
            print(self.subdir, self.phase, epoch)
            if epoch % 10 == 0:
                self.phase = 'val'
                with torch.no_grad():
                    self.iterate()
                torch.save(self.model.state_dict(), self.save_path)
                print(self.subdir, self.phase, epoch)
        np.save(f'/home/jqshen/MyCode/MyModel/store_{self.subdir}.npy', self.store)


if __name__ == '__main__':
    trainer = Trainer(dataset_dir=dataset_dir, batch_size=30, model_save_path=model_save_path, is_continue=True,
                      pre_trained=pre_trained, max_epoch=50, subdir='white')
    trainer.run()
    trainer = Trainer(dataset_dir=dataset_dir, batch_size=30, model_save_path=model_save_path, is_continue=True,
                      pre_trained=pre_trained, max_epoch=50, subdir='nbi')
    trainer.run()
