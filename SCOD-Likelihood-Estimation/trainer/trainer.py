import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision.transforms as T
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from time import gmtime, strftime
import matplotlib.pyplot as plt
from PIL import Image
import cv2


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker(
            'loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _sanity_check(self, epoch, name, data, output, target):
        invTrans = T.Compose([T.Normalize(mean=[0., 0., 0.],
                                          std=[1/0.229, 1/0.224, 1/0.225]),
                              T.Normalize(mean=[-0.485, -0.456, -0.406],
                                          std=[1., 1., 1.]),
                              ])

        totensor = T.ToTensor()

        data, output, target = data.detach().cpu(
        ), output.detach().cpu(), target.detach().cpu()

        image = invTrans(data[:, :3].type(torch.FloatTensor))
        output = output >= 0.5

        nr_imgs = data.shape[0]

        
        side = int(np.ceil(np.sqrt(nr_imgs)))
        img_size = None
        cols = []
        for j in range(side):
            imgs = []
            for i in range(side):
                k = j*side + i
                if k < nr_imgs:
                    im = np.asarray((image[k, :, :, :]).permute(1, 2, 0))
                    if img_size is None:
                        img_size = np.shape(im)
                    im = np.ascontiguousarray(im*255, dtype=np.uint8)
                    im = cv2.putText(im,
                                    f"Predicted: {output[k]}, GT: {target[k]}",
                                    org=(20, 20),
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    fontScale=0.5,
                                    color=(0, 255, 0) if output[k] == target[k] else (
                                        255, 0, 0),
                                    thickness=2)
                else:
                    if len(imgs) == 0:
                        break
                    else:
                        im = np.zeros(img_size)
                imgs.append(im)

            if len(imgs) > 0:
                img = np.row_stack(imgs)
                cols.append(img)

        img = np.column_stack(cols)

        now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        Image.fromarray((img).astype(np.uint8)
                        ).save(f'Epoch_{epoch}_{name}_{now}.png')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """            
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx == 0:
                self._sanity_check(epoch, 'train', data, output, target)

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                
                if batch_idx == 0:
                    self._sanity_check(epoch, 'val', data, output, target)

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(
                        met.__name__, met(output, target))
                self.writer.add_image('input', make_grid(
                    data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
