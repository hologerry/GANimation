# import argparse
# import datetime
# import glob
import os
# import pickle
# import re
# import time
from os.path import join as ospj

import cv2
# import face_recognition
import numpy as np
import torch
# import torchvision.transforms as transforms
# from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataloader import get_dataloader
from eval_attr.models.resnet import resnet18
from eval_attr.utils.eval import accuracy as attr_accuracy
from eval_attr.utils.misc import AverageMeter
from eval_fid.fid_score import calculate_frechet_distance
from eval_fid.inception import InceptionV3
from models.models import ModelsFactory
from options.test_options import TestOptions
from test_attribute_utils import prepare_attr_eval_batch
# from utils import cv_utils, face_utils


def predict_attr_score(opts, eval_loader, attr_pred_model, gan_model):
    print("Evaluating the attribute performance...")
    top1 = [AverageMeter() for _ in range(len(opts.selected_attrs))]

    # gan_model.eval()
    # switch to evaluate mode
    attr_pred_model.eval()
    all_avg = 0
    each_top1_avg = []

    with torch.no_grad():
        for i, data_batch in tqdm(enumerate(eval_loader), total=len(eval_loader)):
            # measure data loading time
            modified_batch = prepare_attr_eval_batch(data_batch, opts.selected_attrs)

            target = modified_batch['attr_target'].cuda()
            target_idx = modified_batch['target_idx'].cuda()
            # compute output
            gan_model.set_input(modified_batch)
            fake_img = gan_model.forward(keep_data_for_visuals=False, return_estimates=False, return_for_eval=True)
            output = attr_pred_model(fake_img)
            # measure accuracy
            prec1 = []
            count = 0
            assert len(output) == len(opts.selected_attrs)
            for j in range(len(output)):
                for batch_idx in range(fake_img.size(0)):
                    b_target_idx = target_idx[batch_idx]
                    if j == b_target_idx.item():
                        count += 1
                        cur_output = output[j][batch_idx]
                        cur_output = cur_output.reshape((1, 2))
                        cur_target = target[batch_idx, j]
                        cur_target = cur_target.reshape((1,))

                        cur_prec1 = attr_accuracy(cur_output, cur_target, topk=(1,))
                        prec1.append(cur_prec1)
                        top1[j].update(prec1[-1][0].item(), 1)

            assert count == fake_img.size(0)  # since for each sample, we only calculate its target attribute
            each_top1_avg = [top1[k].avg for k in range(len(top1))]
            all_avg = sum(each_top1_avg) / len(each_top1_avg)

    # gan_model.train()
    return all_avg, each_top1_avg



def predict_fid_score(opts, eval_part1_loader, eval_part2_loader, fid_model, gan_model):
    print("Evaluating the FID performance...")
    fid_model.eval()
    # gan_model.eval()

    pred_act_arr1 = np.empty((len(eval_part1_loader.dataset), opts.dims))
    for i, data_batch in tqdm(enumerate(eval_part1_loader), total=len(eval_part1_loader)):
        img = data_batch['real_img'].cuda()

        start = i
        end = i + img.size(0)

        pred = fid_model(img)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_act_arr1[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu1 = np.mean(pred_act_arr1, axis=0)
    sigma1 = np.cov(pred_act_arr1, rowvar=False)

    pred_act_arr2 = np.empty((len(eval_part2_loader.dataset), opts.dims))
    for i, data_batch in tqdm(enumerate(eval_part2_loader), total=len(eval_part2_loader)):
        modified_batch = prepare_attr_eval_batch(data_batch, opts.selected_attrs)
        gan_model.set_input(modified_batch)
        fake_img = gan_model.forward(keep_data_for_visuals=False, return_estimates=False, return_for_eval=True)

        start = i
        end = i + fake_img.size(0)

        pred = fid_model(fake_img)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_act_arr2[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu2 = np.mean(pred_act_arr2, axis=0)
    sigma2 = np.cov(pred_act_arr2, rowvar=False)

    try:
        fid_score = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    except Exception:
        fid_score = 0.0
    # gan_model.train()
    return fid_score

def main():
    opts = TestOptions().parse()
    if not os.path.isdir(opts.output_dir):
        os.makedirs(opts.output_dir)

    eval_part1_loader = get_dataloader(opts.data_dir, 'celebahq_ffhq_fake', 'eval_part1',
                                       opts.image_size, opts.selected_attrs, len(opts.selected_attrs))
    eval_part2_loader = get_dataloader(opts.data_dir, 'celebahq_ffhq_fake', 'eval_part2', 
                                       opts.image_size, opts.selected_attrs, len(opts.selected_attrs))

    model = ModelsFactory.get_by_name(opts.model, opts)
    model.set_eval()

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[opts.dims]
    fid_model = InceptionV3([block_idx])
    fid_model = torch.nn.DataParallel(fid_model).cuda()
    fid_model = fid_model.eval()

    # NOTE: here we hard code to resnet18, we construct the resnet with selected attributes
    attr_pred_model = resnet18(pretrained=True, num_attributes=len(opts.selected_attrs))
    attr_model_ckpt = ospj('eval_attr/checkpoints_select_no_extra/model_best.pth.tar')  # for local
    assert os.path.isfile(attr_model_ckpt), f"checkpoint file {attr_model_ckpt} for attribute prediction not found!"
    print(f"=> loading attribute checkpoint '{attr_model_ckpt}'")
    checkpoint = torch.load(attr_model_ckpt, map_location=torch.device("cpu"))
    attr_pred_model.load_state_dict(checkpoint['state_dict'])
    attr_pred_model = torch.nn.DataParallel(attr_pred_model).cuda()
    attr_pred_model = attr_pred_model.eval()
    # attr_pred_model_cpu = attr_pred_model.module
    print(f"=> loaded attribute checkpoint '{attr_model_ckpt}' (epoch {checkpoint['epoch']})")

    fid_score = predict_fid_score(opts, eval_part1_loader, eval_part2_loader, fid_model, model)
    all_attrs_avg, each_attr_avg = predict_attr_score(opts, eval_part2_loader, attr_pred_model, model)

    eval_dict = {}
    eval_dict["FID"] = fid_score
    eval_dict["Attribute_Average"] = all_attrs_avg
    for k, v in eval_dict.items():
        # writer.add_scalar(f"Eval/{k}", v, epoch)
        print(f"Eval {k}: {v}")

    all_attr_eval_dict = {}
    for attr_name, attr_pred in zip(opts.selected_attrs, each_attr_avg):
        all_attr_eval_dict[attr_name] = attr_pred
    # for k, v in all_attr_eval_dict.items():
    #     writer.add_scalar(f"Eval_Each_Attr/{k}", v, epoch)


if __name__ == '__main__':
    main()
