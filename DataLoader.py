from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import time
import copy


class dataset(Dataset):
    def __init__(self, trainlist, label, batch_size):
        super(dataset, self).__init__()
        self.trainlist = open(trainlist).readlines()
        length = len(self.trainlist)//batch_size
        self.trainlist = self.trainlist[0: length * batch_size]
        self.label = label



    def augmentImage(self, image, target):
        net_size = 416
        h, w = image.shape[0], image.shape[1]
        res_scale = net_size / max(h, w)
        res_h, res_w = min(np.ceil(h * res_scale).astype(np.int16), net_size), \
                       min(np.ceil(w * res_scale).astype(np.int16), net_size)

        res_image = cv2.resize(image, (res_w, res_h))
        if min(res_h, res_w) == net_size:
            image = np.transpose(res_image, [2, 0, 1])/255
            return image, target
        pad_1 = np.random.randint(low=0, high=416 - min(res_h, res_w), size=1)[0]
        if res_w > res_h:
            # print('1', res_image.shape)
            pad_2 = net_size - pad_1 - res_h
            image = np.transpose(np.pad(res_image, ((pad_1, pad_2), (0, 0), (0, 0)), 'constant', constant_values=127), [2, 0, 1])/255
            # print('1', pad_1, pad_2, res_h, image.shape)
            target[:, 2] = (target[:, 2] * res_h + pad_1)/net_size
            target[:, 4] = target[:, 4] * res_h/net_size
        else:
            pad_2 = net_size - pad_1 - res_w
            # print('2', res_image.shape)
            image = np.transpose(np.pad(res_image, ((0, 0), (pad_1, pad_2), (0, 0)), 'constant', constant_values=127), [2, 0, 1])/255
            # print('2', pad_1, pad_2, res_w, image.shape)
            target[:, 1] = (target[:, 1] * res_w + pad_1) / net_size
            target[:, 3] = target[:, 3] * res_w / net_size

        return image, target


    def __len__(self):
        return len(self.trainlist)


    def bbox_iou(self, gt, anchors):
        intra = np.minimum(gt[:, 0], anchors[:, 0]) * np.minimum(gt[:, 1], anchors[:, 1])
        union = gt[:, 0] * gt[:, 1] + anchors[:, 0] * anchors[:, 1] - intra
        iou = intra/union
        return iou

    def build_target(self, anchors, nG, target_, nA, nC, net_size=416, ignore_thresh=0.5):
        obj_mask = np.zeros(shape=(nA, nG, nG)).astype(np.float32)
        conf_mask = np.ones(shape=(nA, nG, nG)).astype(np.float32)
        tx = np.zeros(shape=(nA, nG, nG)).astype(np.float32)
        ty = np.zeros(shape=(nA, nG, nG)).astype(np.float32)
        tw = np.zeros(shape=(nA, nG, nG)).astype(np.float32)
        th = np.zeros(shape=(nA, nG, nG)).astype(np.float32)
        tcls = np.zeros(shape=(nA, nG, nG, nC))
        if target_.shape[0] == 0:
            return 0
        for t in range(target_.shape[0]):
            cls_label = int(target_[t, 0])
            gx = target_[t, 1] * nG
            gy = target_[t, 2] * nG
            gw = target_[t, 3] * nG
            gh = target_[t, 4] * nG
            gi = int(gx)
            gj = int(gy)
            gt_box = np.array([[target_[t, 1] * net_size, target_[t, 2] * net_size]])
            anch_ious = self.bbox_iou(gt_box, anchors)
            conf_mask[anch_ious > ignore_thresh, gj, gi] = 0
            best_anch = np.argmax(anch_ious)
            obj_mask[best_anch, gj, gi] = 1
            conf_mask[best_anch, gj, gi] = 1

            tx[best_anch, gj, gi] = gx - gi
            ty[best_anch, gj, gi] = gy - gj
            # Width and height
            tw[best_anch, gj, gi] = np.log(gw / anchors[best_anch][0] + 1e-16)
            th[best_anch, gj, gi] = np.log(gh / anchors[best_anch][1] + 1e-16)
            tcls[best_anch, gj, gi, cls_label] = 1

        return tx, ty, tw, th, conf_mask, obj_mask, tcls

    def __getitem__(self, index):
        targets = np.zeros(shape=(50, 5))
        imagePath = self.trainlist[index]
        labelPath = os.path.join(self.label, imagePath.split('\\', len(imagePath))[-1][0:-4] + 'txt')
        targets_ = np.loadtxt(labelPath).reshape(-1, 5)
        image, targets_ = self.augmentImage(cv2.imread(imagePath[0:-1]), targets_)
        tx13, ty13, tw13, th13, conf_mask13, obj_mask13, tcls13 = self.build_target(target_=targets_,
                                         anchors=np.array([[116, 90], [156, 198], [373, 326]]), nC=7, nG=13, nA=3)
        tx26, ty26, tw26, th26, conf_mask26, obj_mask26, tcls26 = self.build_target(target_=targets_,
                                         anchors=np.array([[30, 61], [62, 45], [59, 119]]), nC=7, nG=26, nA=3)
        tx52, ty52, tw52, th52, conf_mask52, obj_mask52, tcls52 = self.build_target(target_=targets_,
                                         anchors=np.array([[10, 13], [16, 30], [33, 23]]), nC=7, nG=52, nA=3)
        tcoord_13 = np.concatenate((tx13.reshape(-1, 1), ty13.reshape(-1, 1), tw13.reshape(-1, 1),
                                    th13.reshape(-1, 1), conf_mask13.reshape(-1, 1), obj_mask13.reshape(-1, 1),
                                    tcls13.reshape(-1, 7)), 1)
        tcoord_26 = np.concatenate((tx26.reshape(-1, 1), ty26.reshape(-1, 1), tw26.reshape(-1, 1),
                                    th26.reshape(-1, 1), conf_mask26.reshape(-1, 1), obj_mask26.reshape(-1, 1),
                                    tcls26.reshape(-1, 7)), 1)
        tcoord_52 = np.concatenate((tx52.reshape(-1, 1), ty52.reshape(-1, 1), tw52.reshape(-1, 1),
                                    th52.reshape(-1, 1), conf_mask52.reshape(-1, 1), obj_mask52.reshape(-1, 1),
                                    tcls52.reshape(-1, 7)), 1)
        tcoord = np.concatenate((tcoord_13, tcoord_26, tcoord_52), axis=0)
        targets[0:targets_.shape[0], :] = targets_

        return image.astype(np.float32), tcoord.astype(np.float32)










if __name__ == '__main__':


    dataset = dataset(trainlist='./cfg/trainlist_7.txt', label='E:\Person_detection\Dataset\Yolov3_labels\labels')
    dataloader = DataLoader(shuffle=True, dataset=dataset, batch_size=64, num_workers=4)
    t2 = 0
    for e in range(10):
        sum_time = 0
        for idx, (image, t13, t26, t52, tcls13, tcls26, tcls52) in enumerate(dataloader):
            t1 = time.time()
            print(idx, t1 - t2)
            if idx > 0:
                sum_time += (t1 - t2)
            t2 = copy.deepcopy(t1)
        print('sum_time:', sum_time)