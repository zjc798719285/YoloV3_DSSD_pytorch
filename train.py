from DataLoader import dataset, DataLoader
from models.Resnet import MnasNet
import torch.optim as optim
import Loss
import torch
import time
import torch.nn as nn


trainlist = './cfg/trainlist_7.txt'
label = 'E:\Person_detection\Dataset\Yolov3_labels\labels'
epoch = 1000
batch_size = 12


models = MnasNet().to('cuda')
models.train()
models.load_state_dict(torch.load('./checkpoints/pretrain/yolo3_32.pt'))
dataset = dataset(trainlist=trainlist, label=label, batch_size=batch_size)
dataloader = DataLoader(shuffle=True, dataset=dataset, batch_size=batch_size, num_workers=2)
optimizer = optim.Adam(models.parameters(), lr=0.0001)



def main():
    for i in range(epoch):
        t1 = time.time()
        for step, (image, tcoord) in enumerate(dataloader):
           outputs = models(image.to('cuda'))
           print('epoch:', i, 'step:', step, 'loss:')
        t2 = time.time()
        print('epoch time:', (t2 - t1))
        print('*************save models**********************')
        torch.save(models.state_dict(), './checkpoints/yolo3_{}.pt'.format(i))



if __name__== '__main__':
    main()