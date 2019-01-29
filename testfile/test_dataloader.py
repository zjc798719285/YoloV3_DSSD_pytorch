from DataLoader import dataset
from torch.utils.data import DataLoader

import time
import numpy as np
import torch

batch_size = 24
trainlist = '../cfg/trainlist_7.txt'
label = 'E:\Person_detection\Dataset\Yolov3_labels\labels'

dataset = dataset(trainlist=trainlist, label=label, batch_size=batch_size)
dataloader = DataLoader(shuffle=False, dataset=dataset, batch_size=batch_size, num_workers=2)


def decodebbox(targets, anchors):

    for tar, anchor in zip(targets, anchors):
        nG = tar.shape[2]
        tcoord, conf_mask, obj_mask = tar[:, 0:12, ...], tar[:, 12:15, ...], tar[:, 15:, ...]
        tcoord = tcoord.view(1, 4, 3, nG, nG).permute(0, 2, 3, 4, 1).contiguous().view(-1, 4).numpy()
        grid_x, grid_y = np.meshgrid(np.linspace(start=0, stop=nG-1, num=nG),
                                     np.transpose(np.linspace(start=0, stop=nG-1, num=nG)))
        grid_x = np.repeat(grid_x, repeats=3, axis=2)
        print()





if __name__ =='__main__':

 for epoch in range(10):
    t1 = time.time()
    for step, (image, tcoord) in enumerate(dataloader):
        tcoord = tcoord.numpy()
        print(epoch, step)
    t2 = time.time()
    print('epoch time:', (t2 - t1))




