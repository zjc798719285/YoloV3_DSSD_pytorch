# from models.Resnet_mask_org import *
from models.Resnet import MnasNet
import cv2, os
import numpy as np
import torch


def demo(weightbox, testlist, labdir, anchors, iou_thresh, nC=7):
    boxer = MnasNet().eval()
    boxer.load_state_dict(torch.load(weightbox))
    boxer.cuda()
    sum_recall, sum_acc = 0, 0
    f = open(testlist, 'r')
    files = f.readlines()
    for num, i in enumerate(files):

        lab = os.path.join(labdir, i.split('\\', len(i))[-1][0:-4] + 'txt')
        gt = np.loadtxt(lab).reshape((-1, 5))

        img = cv2.imread(i[0:-1])

        sized = cv2.resize(img, (416, 416))/255
        sized = np.expand_dims(np.transpose(sized, (2, 0, 1)), 0)

        outputs = boxer(torch.cuda.FloatTensor(sized))
        detections = decodebbox(outputs, anchors, nC)
        boxes = nms(detections, iou_thresh=0.4, conf_thresh=0.3)
        boxes = np.array([i[0] for i in boxes if i[1] == 1])
        TP, numGT, numPRE = compute_tp(gt, boxes, iou_thresh)
        sum_recall += (TP/numGT)
        sum_acc += (TP/numPRE)
        print('id:', num, 'TP:', TP, 'nGT:', numGT, 'nPRE:', numPRE, 'recall:', TP/numGT, 'acc:', TP/numPRE,
              'mRecall:', sum_recall / (num+1), 'mACC:', sum_acc/(num+1))
    print('mRecall:', sum_recall/len(files))
    print('mACC:', sum_acc/len(files))


def compute_tp(gt, box, iou_thresh):
    TP=0
    if len(box) == 0:
        return 0, gt.shape[0], 1e-16
    for gi in gt:
       iou = bbox_iou2(gi, box)
       if len(iou) == 1:
           best_iou = iou
       else:
            best_iou = np.max(iou)
       if best_iou > iou_thresh:
           TP += 1
    return TP, gt.shape[0], box.shape[0]



def decodebbox(output, anchors, nC):

    boxes, confs, classes,  = [], [], []
    for out, an in zip(output, anchors):
        nB = out.shape[0]  # batch size
        nA = an.shape[0]  # num_anchors
        nH = out.shape[2]
        nW = out.shape[3]
        cls_anchor_dim = nB * nA * nH * nW
        out = out.view(nB, nA, (5 + nC), nH, nW)
        ix = torch.LongTensor(range(0, 5)).to('cuda')

        cls_grid = torch.linspace(5, 5 + nC - 1, nC).long().to('cuda')
        pred_boxes = torch.FloatTensor(4, cls_anchor_dim).to('cuda')
        coord = out.index_select(2, ix[0:4]).view(nB * nA, -1, nH * nW).\
                         transpose(0, 1).contiguous().view(-1, cls_anchor_dim)
        coord[0:2] = coord[0:2].sigmoid()
        conf = out.index_select(2, ix[4]).view(cls_anchor_dim).sigmoid()

        cls = out.index_select(2, cls_grid)
        cls = cls.view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(cls_anchor_dim, nC)
        cls = cls.sigmoid()
        grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim).to('cuda')
        grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(cls_anchor_dim).to('cuda')
        anchor_w = an.index_select(1, ix[0]).repeat(1, nB * nH * nW).view(cls_anchor_dim)
        anchor_h = an.index_select(1, ix[1]).repeat(1, nB * nH * nW).view(cls_anchor_dim)

        pred_boxes[0] = (coord[0] + grid_x) / nW
        pred_boxes[1] = (coord[1] + grid_y) / nH
        pred_boxes[2] = (coord[2].exp() * anchor_w)/416
        pred_boxes[3] = (coord[3].exp() * anchor_h)/416

        pred_boxes = pred_boxes.detach().cpu().numpy()
        conf = conf.detach().cpu().numpy()
        cls = cls.detach().cpu().numpy()
        boxes.append(pred_boxes)
        confs.append(conf)
        classes.append(cls)

    boxes = np.concatenate((boxes), 1)
    confs = np.concatenate((confs), 0)
    classes = np.concatenate((classes), 0)

    return {'boxes': boxes, 'confs': confs, 'classes': classes}




def nms(detections, iou_thresh, conf_thresh):
    if len(detections) < 1:
        return []
    coord = np.transpose(detections['boxes'], [1, 0])
    confs = np.expand_dims(detections['confs'], 1)
    classes = detections['classes']
    classes = classes / np.sum(classes, axis=1, keepdims=True)
    boxes = np.concatenate((coord, confs, classes), axis=1)
    boxes = boxes[boxes[:, 4] > conf_thresh]
    a = boxes[np.argsort(-boxes[:, 4])]
    box_list = []
    maxsteps = len(a)
    for i in range(maxsteps):
        if len(a) <= 0:
            break
        bestbox = a[0:1, :]
        iou = bbox_iou(a, bestbox)
        # idx = [iou > iou_thresh] or [np.argmax(a[:, 5:], axis=1) == np.argmax(bestbox[:, 5:], axis=1)]
        idx_ = [iou <= iou_thresh] or [np.argmax(a[:, 5:], axis=1) != np.argmax(bestbox[:, 5:], axis=1)]
        getbox = a[0]   # np.mean(a[idx], axis=0)
        box_list.append([getbox[0:4], np.argmax(getbox[4:])])
        a = a[idx_]
    return box_list


def bbox_iou2(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
        # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
    b1_y1, b1_y2 = box1[2] - box1[4] / 2, box1[2] + box1[4] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the corrdinates of the intersection rectangle

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16) + 1e-16
    iou = np.reshape(iou, newshape=(-1))
    return iou


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
        # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the corrdinates of the intersection rectangle

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * np.maximum(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou





############################################
if __name__ == '__main__':

    weightbox = './checkpoints/yolo3_32.pt'
    videofile = './cfg/voc_test.txt'
    labdir = 'G:\DataSet\VOCtrainval_11-May-2012\VOCdevkit\VOC2012\labels_ped_car'

    anchor13 = torch.cuda.FloatTensor(np.array([[116, 90], [156, 198], [373, 326]]))
    anchor26 = torch.cuda.FloatTensor(np.array([[30, 61], [62, 45], [59, 119]]))
    anchor52 = torch.cuda.FloatTensor(np.array([[10, 13], [16, 30], [33, 23]]))
    anchors = [anchor13, anchor26, anchor52]
    demo(weightbox, videofile, labdir, anchors, iou_thresh=0.5)



