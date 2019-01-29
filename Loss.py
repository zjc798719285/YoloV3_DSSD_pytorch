import torch
import torch.nn as nn




def loss(outputs, t13, t26, t52, tcls13, tcls26, tcls52, nB, epoch, step):

    loss_13 = loss_base(outputs[0], t13, tcls13, nG=13, nA=3, nB=nB, nC=7, epoch=epoch, step=step)
    loss_26 = loss_base(outputs[1], t26, tcls26, nG=26, nA=3, nB=nB, nC=7, epoch=epoch, step=step)
    loss_52 = loss_base(outputs[2], t52, tcls52, nG=52, nA=3, nB=nB, nC=7, epoch=epoch, step=step)
    loss = loss_13 + loss_26 + loss_52

    return loss



def loss_base(output, target, tcls, nG, nA, nB, nC, epoch, step):
    '''
    :param output:
    :param target:
    :param tcls:
    :param nG:
    :param nA:
    :param nB:
    :return:
    '''

    num_anchors = nB * nA * nG * nG

    coord, conf, cls = output[:, 0:12, ...], output[:, 12:15, ...], output[:, 15:, ...]
    tcoord, conf_mask, obj_mask = target[:, 0:12, ...], target[:, 12:15, ...], target[:, 15:18, ...]
    coord = coord.contiguous().view(4, num_anchors)
    tcoord = tcoord.contiguous().view(4, num_anchors)
    cls = cls.contiguous().view(nB*nA, nC, nG*nG).transpose(1, 2).contiguous().view(-1, nC)

    conf = conf.contiguous().view(1, num_anchors)
    conf_mask = conf_mask.contiguous().view(1, num_anchors)
    obj_mask = obj_mask.contiguous().view(num_anchors)


    tcls = tcls.contiguous().view(-1, nC)


    nProposals = int((conf >0.5).sum())

    loss_coord = nn.SmoothL1Loss(size_average=False)(coord*obj_mask, tcoord*obj_mask)/nB
    loss_conf = nn.BCELoss(size_average=False)(conf*conf_mask, conf_mask * obj_mask)/nB
    loss_cls = nn.BCELoss(size_average=False)(cls, tcls)/nB
    loss = loss_coord + loss_conf + loss_cls
    print('epoch:%3d, step:%4d, nPP:%3d, Loss: box %6.3f, conf %6.3f, class %6.3f, total %7.3f' %
          (epoch, step, nProposals, loss_coord, loss_conf, loss_cls, loss))
    return loss