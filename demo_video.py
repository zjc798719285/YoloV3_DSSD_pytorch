import torch
from models.Resnet import MnasNet
import cv2
import numpy as np



def demo(weightbox, videofile, anchors, nC=7):

    boxer = MnasNet().eval()
    boxer.load_state_dict(torch.load(weightbox))
    use_cuda = True
    if use_cuda:
        boxer.cuda()
        # masker.cuda()
    cap = cv2.VideoCapture(videofile)
    while True:
        res, img = cap.read()
        img_show = cv2.resize(img, (416, 416))
        sized = np.expand_dims(np.transpose(img_show / 255, [2, 0, 1]), 0)
        outputs = boxer(torch.cuda.FloatTensor(sized))
        # output_mask = masker(x1, x2, x3, x4)
        detections = decodebbox(outputs, anchors, nC)
        boxes = nms2(detections, iou_thresh=0.4, conf_thresh=0.4)
        # boxes = nms(detections, iou_thresh=0.4, conf_thresh=0.4)
        img_show = drawbox(boxes, img_show)
        # img_show = drawmask(output=output_mask, img_show=img_show)

        cv2.imshow('frame', img_show)
        if cv2.waitKey(1) & 0xFF == ord(' '):  # 按q停止
            while True:
                if cv2.waitKey(1) & 0xFF == ord(' '):  # 按q停止
                    break





def drawbox(boxes, img_show):
    names = ['person', 'car', 'bicycle', 'motobike', 'bus', 'train', 'truck']
    for box in boxes:
        if box[5] > 0.5:
            box[0], box[1], box[2], box[3] = box[0] - box[2] / 2, box[0] \
                 + box[2] / 2, box[1] - box[3] / 2, box[1] + box[3] / 2

            cv2.rectangle(img_show, (int(box[0]), int(box[2])),
                      (int(box[1]), int(box[3])), (30, 0, 100), 1)
            # cv2.putText(img_show, names[box[1]-1], (int(box[0][0] * 416), int(box[0][2] * 416)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #         (30 * int(box[1]), 255 - 30 * int(box[1]), 255-10 * int(box[1])), 1)
    return img_show

def decodebbox(target, anchor, nC):
    boxes, confs, classes,  = [], [], []

    nB = target.shape[0]  # batch size
    nA = anchor.shape[0]  # num_anchors
    nH = target.shape[2]
    nW = target.shape[3]
    cls_anchor_dim = nB * nA * nH * nW
    ix = torch.LongTensor(range(0, 5)).to('cuda')
    cls_grid = torch.linspace(5, 5 + nC - 1, nC).long().to('cuda')
    pred_boxes = torch.FloatTensor(cls_anchor_dim, 4).to('cuda')
    coord, conf, cls = target[:, 0:12, ...], target[:, 12:15, ...], target[:, 15:, ...]
    coord = coord.view(-1, 4)
    conf = conf.view(-1, 1)
    cls = cls.contiguous().view(nB * nA, nC, nH * nW).transpose(1, 2).contiguous().view(-1, nC)
    grid_x = torch.linspace(0, nW - 1, nW).repeat(nB * nA, nH, 1).view(cls_anchor_dim).to('cuda')
    grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(cls_anchor_dim).to('cuda')
    anchor_w = anchor.index_select(1, ix[0]).repeat(1, nB * nH * nW).view(cls_anchor_dim)
    anchor_h = anchor.index_select(1, ix[1]).repeat(1, nB * nH * nW).view(cls_anchor_dim)

    pred_boxes[:, 0] = (coord[:, 0]*nH + grid_x)*(416/nH)
    pred_boxes[:, 1] = (coord[:, 1]*nH + grid_y)*(416/nH)
    pred_boxes[:, 2] = (coord[:, 2].exp() * anchor_w)
    pred_boxes[:, 3] = (coord[:, 3].exp() * anchor_h)

    pred_boxes = pred_boxes.detach().cpu().numpy()
    conf = conf.detach().cpu().numpy()
    cls = cls.detach().cpu().numpy()
    boxes.append(pred_boxes)
    confs.append(conf)
    classes.append(cls)

    boxes = np.concatenate((boxes), 0)
    confs = np.concatenate((confs), 0)
    classes = np.concatenate((classes), 0)

    return {'boxes': boxes, 'confs': confs, 'classes': classes}




def nms(detections, iou_thresh, conf_thresh):
    if len(detections) < 1:
        return []
    coord = detections['boxes']
    confs = detections['confs']
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
        idx = [iou > iou_thresh] #and [np.argmax(a[:, 5:], axis=1) == np.argmax(bestbox[:, 5:], axis=1)]
        idx_ = [iou <= iou_thresh] #and [np.argmax(a[:, 5:], axis=1) != np.argmax(bestbox[:, 5:], axis=1)]
        getbox = np.mean(a[idx], axis=0)
        box_list.append([getbox[0:4], np.argmax(getbox[5:])])
        a = a[idx_]
    return box_list



def nms2(detections, iou_thresh, conf_thresh):
    if len(detections) < 1:
        return []
    coord = detections['boxes']
    confs = detections['confs']
    classes = detections['classes']
    classes = classes / np.sum(classes, axis=1, keepdims=True)
    boxes = np.concatenate((coord, confs, classes), axis=1)
    boxes = boxes[boxes[:, 4] > conf_thresh]
    return boxes


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

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)




def drawmask(img_show, output):

    mask = np.repeat(np.transpose(output.sigmoid().detach().cpu().numpy()[0, ...], [1, 2, 0]), 3, axis=2)
    mask = np.where(cv2.resize(mask, (416, 416)) > 0.5, 1, 0).astype(np.uint8)
    mask_color = np.zeros_like(mask)
    mask_color[..., 0] = mask[..., 0] * 10
    mask_color[..., 1] = mask[..., 1] * 50
    mask_color[..., 2] = mask[..., 2] * 0
    mask_color.astype(np.uint8)
    img_show = img_show + mask_color

    return img_show


############################################
if __name__ == '__main__':

    weightbox = './checkpoints/yolo3_118.pt'

    videofile = 'E:\Person_detection\Dataset\\video\\test2.mp4'

    anchor13 = torch.cuda.FloatTensor(np.array([[116, 90], [156, 198], [373, 326]]))
    anchor26 = torch.cuda.FloatTensor(np.array([[30, 61], [62, 45], [59, 119]]))
    anchor52 = torch.cuda.FloatTensor(np.array([[10, 13], [16, 30], [33, 23]]))
    anchors = [anchor13, anchor26, anchor52]
    demo(weightbox,  videofile, anchors)




