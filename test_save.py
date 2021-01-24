import os
# import pickle
# import re
# import time
import tqdm
from os.path import join as ospj
from models.models import ModelsFactory
import torch.nn.functional as F
from options.test_options import TestOptions
from test_attribute_utils import change_beard_target, prepare_attr_eval_batch
from torchvision.utils import save_image
from data.dataloader import get_dataloader
from test_attribute_utils import (change_beard, change_hair_type,
                                  change_hair_color_target, change_skin_color_target,
                                  change_one_attr_target,
                                  change_multi_attr,
                                  interpolate_hair_type, interpolate_hair_color,
                                  interpolate_beard, interpolate_skin_color,
                                  interpolate_other_attr)

def test_and_save_single(model, data, save_path):
    model.set_input(data)
    fake_img = model.forward(keep_data_for_visuals=False, return_estimates=False, return_for_eval=True)
    fake_img = fake_img.clamp(-1.0, 1.0)

    for idx, one in enumerate(fake_img):
        one_save = save_path.replace(f'.png', f'_{idx}.png')
        save_image(one, one_save, padding=0, normalize=True)

def self_rec(model, data):
    org_img = data['real_img'].cuda()
    model.set_input(data)
    fake_img = model.forward(keep_data_for_visuals=False, return_estimates=False, return_for_eval=True)
    return F.l1_loss(fake_img, org_img)


def main():
    opts = TestOptions().parse()
    if not os.path.isdir(opts.output_dir):
        os.makedirs(opts.output_dir)

    result_dir = ospj(opts.output_dir, "results_evaluation_ganimation")
    os.makedirs(result_dir, exist_ok=True)

    model = ModelsFactory.get_by_name(opts.model, opts)
    model.set_eval()

    test_loader = get_dataloader(opts.data_dir, 'celebahq_ffhq_fake', 'ffhq_test',
                                 opts.image_size, opts.selected_attrs, 1)

    test_epoch = 200
    all_sre_val = 0.0
    for test_batch_idx, test_data_batch in enumerate(test_loader):

        org_img, org_attr = test_data_batch['real_img'].clone().detach(), test_data_batch['real_cond'].clone().detach()

        img, attr, _ = change_hair_color_target(org_img.clone().detach(), org_attr.clone().detach(), opts.selected_attrs)
        test_data_batch['real_img'] = img
        test_data_batch['desired_cond'] = attr
        test_data_batch['real_cond'] = attr
        save_file = ospj(result_dir, f"test_epoch_{test_epoch}_ffhq_batch_{test_batch_idx}_test_hair_color.png")
        test_and_save_single(model, test_data_batch, save_file)


        # skin color
        img, attr, _ = change_skin_color_target(org_img.clone().detach(), org_attr.clone().detach(), opts.selected_attrs)
        test_data_batch['real_img'] = img
        test_data_batch['desired_cond'] = attr
        test_data_batch['real_cond'] = attr
        save_file = ospj(result_dir, f"test_epoch_{test_epoch}_ffhq_batch_{test_batch_idx}_test_skin_color.png")
        test_and_save_single(model, test_data_batch, save_file)

        # beard
        img, attr, _ = change_beard_target(org_img.clone().detach(), org_attr.clone().detach(), opts.selected_attrs)
        test_data_batch['real_img'] = img
        test_data_batch['desired_cond'] = attr
        test_data_batch['real_cond'] = attr
        save_file = ospj(result_dir, f"test_epoch_{test_epoch}_ffhq_batch_{test_batch_idx}_test_beard.png")
        test_and_save_single(model, test_data_batch, save_file)

        # one attribute
        img, attr, _ = change_one_attr_target(org_img.clone().detach(), org_attr.clone().detach(), opts.selected_attrs)
        test_data_batch['real_img'] = img
        test_data_batch['desired_cond'] = attr
        test_data_batch['real_cond'] = attr
        save_file = ospj(result_dir, f"test_epoch_{test_epoch}_ffhq_batch_{test_batch_idx}_test_one.png")
        test_and_save_single(model, test_data_batch, save_file)

        # test_data_batch['real_img'] = org_img
        # test_data_batch['desired_cond'] = org_attr
        # test_data_batch['real_cond'] = org_attr
        # sre = self_rec(model, test_data_batch)
        # print(test_batch_idx, sre)
        # all_sre_val += sre.item()

    # print("sre:", all_sre_val / len(test_loader))


if __name__ == '__main__':
    main()
