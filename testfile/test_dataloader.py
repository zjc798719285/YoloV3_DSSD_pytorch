from DataLoader import dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch

batch_size = 1
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

 for step, (image, t13, t26, t52, tcls13, tcls26, tcls52, target) in enumerate(dataloader):
     print(step)
     Image = image.numpy()[0, ...]*255
     if Image.shape[1] != 416 or Image.shape[2] !=416:
         break
     Image = np.transpose(Image, [1, 2, 0]).astype(np.uint8).copy()
     anchor13 = np.array([[116, 90], [156, 198], [373, 326]])
     anchor26 = np.array([[30, 61], [62, 45], [59, 119]])
     anchor52 = np.array([[10, 13], [16, 30], [33, 23]])
     boxes = decodebbox(targets=[t13, t26, t52], anchors=[anchor13, anchor26, anchor52])


     for ti in target:
         x, y, w, h = ti[1]*416, ti[2]*416, ti[3]*416, ti[4]*416
         xmin, ymin, xmax, ymax = int(x - w/2), int(y-h/2), int(x+w/2), int(y+h/2)
         cv2.rectangle(Image, (xmin, ymin), (xmax, ymax), 2)
     cv2.imshow('frame', Image)
     while True:
         if cv2.waitKey(1) & 0xFF == ord(' '):  # 按q停止
             break




