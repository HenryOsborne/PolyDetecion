import argparse
import torch
import yaml

from utils.torch_utils import intersect_dicts
from config.yolov3 import cfg
from models.yolov3 import yolov3
from models.yolo import Model

device = torch.device(cfg.device)


def build_yolo_v5():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='checkpoint/yolov5s.pt', help='initial weights path')  #
    parser.add_argument('--cfg', type=str, default='models/yolov5s.yaml', help='model.yaml path')  #
    parser.add_argument('--hyp', type=str, default='config/hyp.scratch.yaml', help='hyperparameters path')  #
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    opt = parser.parse_args()

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps

    nc = cfg.num_classes
    ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
    state_dict = ckpt['model'].float().state_dict()  # to FP32
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(state_dict, strict=False)  # load

    return model


def build_yolo_v3():
    model = yolov3().to(device)
    model.load_darknet_weights("./checkpoint/darknet53_448.weights")
    return model


def build_model(model=None):
    assert model is not None
    if model == 'yolo_v3':
        return build_yolo_v3()
    elif model == 'yolo_v5':
        return build_yolo_v5()
    else:
        raise ValueError('model type not inplemetion')
