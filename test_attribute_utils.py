import copy
import random

import torch


# def get_test_attr_batch(selected_attrs, original_attrs_batch):
#     attr_channel = len(selected_attrs)
#     attrs = []

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Black_Hair')] = -1
#     attr[selected_attrs.index('Blond_Hair')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Black_Hair')] = -1
#     attr[selected_attrs.index('Brown_Hair')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Black_Hair')] = -1
#     attr[selected_attrs.index('Blond_Hair')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     male_idx = selected_attrs.index('Male')
#     if original_attrs_batch[3][male_idx] == 1:
#         attr[male_idx] = -1
#     else:
#         attr[male_idx] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Male')] = 1
#     # attr[selected_attrs.index('Goatee')] = 1
#     attr[selected_attrs.index('Mustache')] = 1
#     # attr[selected_attrs.index('5_o_Clock_Shadow')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('No_Beard')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     # attr = torch.zeros((attr_channel))
#     # attr[selected_attrs.index('Pale_Skin')] = 1
#     # attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Arched_Eyebrows')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Mouth_Slightly_Open')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Smiling')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     # attr = torch.zeros((attr_channel))
#     # attr[selected_attrs.index('Bangs')] = 1
#     # attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Heavy_Makeup')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Eyeglasses')] = 1
#     attrs.append(attr.reshape(1, attr_channel))

#     attr = torch.zeros((attr_channel))
#     attr[selected_attrs.index('Gray_Hair')] = 1
#     attr[selected_attrs.index('Young')] = -1
#     attrs.append(attr.reshape(1, attr_channel))

#     test_attr = torch.cat(attrs, 0)
#     return test_attr


def change_hair_type(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    hair_types = ['Bald', 'Straight_Hair', 'Wavy_Hair']
    hair_types_in_select = [a for a in hair_types if a in selected_attrs]
    new_img_batch = img_batch.repeat(len(hair_types_in_select), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    hair_type_idxes = []
    for type in hair_types_in_select:
        hair_type_idxes.append(selected_attrs.index(type))
    for type in hair_types_in_select:
        attr = torch.zeros((attr_channel))
        for idx in hair_type_idxes:
            if attr_batch[0, idx] == 1:
                attr[idx] = -1
        attr[selected_attrs.index(type)] = 1
        if attr_batch[0, selected_attrs.index(type)] == 1:
            attr[selected_attrs.index(type)] = 0  # 0 means keep current
        attrs.append(attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, hair_type_idxes


def change_hair_color(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    hair_colors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    hair_colors_in_select = [a for a in hair_colors if a in selected_attrs]
    new_img_batch = img_batch.repeat(len(hair_colors_in_select), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    hair_color_idxes = []
    for color in hair_colors_in_select:
        hair_color_idxes.append(selected_attrs.index(color))
    for color in hair_colors_in_select:
        attr = torch.zeros((attr_channel))
        for idx in hair_color_idxes:
            if attr_batch[0, idx] == 1:
                attr[idx] = -1
        attr[selected_attrs.index(color)] = 1
        if attr_batch[0, selected_attrs.index(color)] == 1:
            attr[selected_attrs.index(color)] = 0  # 0 means keep current
        attrs.append(attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, hair_color_idxes


def change_skin_color(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    skin_colors = ['Skin_0', 'Skin_1', 'Skin_2', 'Skin_3']
    skin_colors_in_select = [a for a in skin_colors if a in selected_attrs]
    new_img_batch = img_batch.repeat(len(skin_colors_in_select), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    skin_color_idxes = []
    for color in skin_colors_in_select:
        skin_color_idxes.append(selected_attrs.index(color))
    for color in skin_colors_in_select:
        attr = torch.zeros((attr_channel))
        for idx in skin_color_idxes:
            if attr_batch[0, idx] == 1:
                attr[idx] = -1
        attr[selected_attrs.index(color)] = 1
        if attr_batch[0, selected_attrs.index(color)] == 1:
            attr[selected_attrs.index(color)] = 0  # 0 means keep current
        attrs.append(attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, skin_color_idxes


def change_beard(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    beards = ['Goatee', 'Mustache', 'No_Beard', 'Sideburns']  # 两鬓胡须
    beards_in_select = [a for a in beards if a in selected_attrs]
    new_img_batch = img_batch.repeat(len(beards_in_select), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    beard_idxes = []
    for color in beards_in_select:
        beard_idxes.append(selected_attrs.index(color))
    for color in beards_in_select:
        attr = torch.zeros((attr_channel))
        for idx in beard_idxes:
            if attr_batch[0, idx] == 1:
                attr[idx] = -1
        attr[selected_attrs.index(color)] = 1
        if attr_batch[0, selected_attrs.index(color)] == 1:
            attr[selected_attrs.index(color)] = 0  # 0 means keep current
        attrs.append(attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, beard_idxes


def change_other_attr(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    other_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                   'Bags_Under_Eyes', 'Bangs', 'Big_Lips', 'Big_Nose',
                   'Blurry', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                   'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                   'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                   'Rosy_Cheeks', 'Smiling', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                   'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    change_in_select = [a for a in other_attrs if a in selected_attrs]

    # change_attr = ['Arched_Eyebrows', 'Eyeglasses', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Young']
    # change_attr = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Eyeglasses', 'Goatee', 'Male', 'Mustache', 'Pale_Skin',
    #                'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young']
    new_img_batch = img_batch.repeat(len(change_in_select), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []
    change_one_idxes = []
    for change in change_in_select:
        attr = torch.zeros((attr_channel))
        idx = selected_attrs.index(change)
        if attr_batch[0, idx] == 1:
            attr[idx] = -1
        else:
            attr[idx] = 1
        attrs.append(attr.reshape(1, attr_channel))
        change_one_idxes.append(idx)
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, change_one_idxes


def all_zero_attr(img_batch, selected_attrs):
    assert img_batch.size(0) == 1
    attr_channel = len(selected_attrs)
    attrs = []
    attr = torch.zeros((attr_channel))
    attrs.append(attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return img_batch, test_attr, None


# def all_one_one_attr(img_batch, selected_attrs):
#     assert img_batch.size(0) == 1
#     change_attr = ['Arched_Eyebrows', 'Eyeglasses', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Young']
#     # change_attr = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Eyeglasses', 'Goatee', 'Male', 'Mustache', 'Pale_Skin',
#     #                'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young']
#     new_img_batch = img_batch.repeat(len(change_attr), 1, 1, 1)
#     attr_channel = len(selected_attrs)
#     attrs = []
#     change_one_idxes = []
#     for change in change_attr:
#         attr = torch.zeros((attr_channel))
#         idx = selected_attrs.index(change)
#         attr[idx] = 1
#         attrs.append(attr.reshape(1, attr_channel))
#         change_one_idxes.append(idx)
#     test_attr = torch.cat(attrs, 0)
#     return new_img_batch, test_attr, change_one_idxes

def interpolate_hair_type(img_batch, attr_batch, selected_attrs, interp_num, start, end, step, div, out=False):
    assert img_batch.size(0) == 1
    hair_types = ['Bald', 'Straight_Hair', 'Wavy_Hair']
    hair_types_in_select = [a for a in hair_types if a in selected_attrs]
    new_img_batch = img_batch.repeat(len(hair_types_in_select), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    hair_type_idxes = []
    for type in hair_types:
        hair_type_idxes.append(selected_attrs.index(type))
    for type in hair_types_in_select:
        for alpha in range(start, end, step):
            alpha /= div
            attr = torch.zeros((attr_channel))
            for idx in hair_type_idxes:
                if attr_batch[0, idx] == 1:
                    attr[idx] = -1
            attr[selected_attrs.index(type)] = 1
            if attr_batch[0, selected_attrs.index(type)] == 1:
                attr[selected_attrs.index(type)] = 0  # 0 means keep current
            attrs.append(alpha * attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, hair_type_idxes


def interpolate_hair_color(img_batch, attr_batch, selected_attrs, interp_num, start, end, step, div, out=False):
    assert img_batch.size(0) == 1
    hair_colors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    hair_colors_in_select = [a for a in hair_colors if a in selected_attrs]
    new_img_batch = img_batch.repeat(len(hair_colors_in_select) * interp_num, 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    hair_color_idxes = []
    for color in hair_colors:
        hair_color_idxes.append(selected_attrs.index(color))

    for color in hair_colors_in_select:
        for alpha in range(start, end, step):
            alpha /= div
            attr = torch.zeros((attr_channel))
            for idx in hair_color_idxes:
                if attr_batch[0, idx] == 1:
                    attr[idx] = -1
            attr[selected_attrs.index(color)] = 1
            if attr_batch[0, selected_attrs.index(color)] == 1:
                attr[selected_attrs.index(color)] = 0  # 0 means keep current
            attrs.append(alpha * attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, hair_color_idxes


def interpolate_skin_color(img_batch, attr_batch, selected_attrs, interp_num, start, end, step, div, out=False):
    assert img_batch.size(0) == 1
    skin_colors = ['Skin_0', 'Skin_1', 'Skin_2', 'Skin_3']
    skin_colors_in_select = [a for a in skin_colors if a in selected_attrs]
    new_img_batch = img_batch.repeat(len(skin_colors_in_select) * interp_num, 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    skin_color_idxes = []
    for color in skin_colors:
        skin_color_idxes.append(selected_attrs.index(color))
    for color in skin_colors_in_select:
        for alpha in range(start, end, step):
            alpha /= div
            attr = torch.zeros((attr_channel))
            for idx in skin_color_idxes:
                if attr_batch[0, idx] == 1:
                    attr[idx] = -1
            attr[selected_attrs.index(color)] = 1
            if attr_batch[0, selected_attrs.index(color)] == 1:
                attr[selected_attrs.index(color)] = 0  # 0 means keep current
            attrs.append(alpha * attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, skin_color_idxes


def interpolate_beard(img_batch, attr_batch, selected_attrs, interp_num, start, end, step, div, out=False):
    assert img_batch.size(0) == 1
    beards = ['Goatee', 'Mustache', 'No_Beard', 'Sideburns']  # 两鬓胡须
    beards_in_select = [a for a in beards if a in selected_attrs]
    new_img_batch = img_batch.repeat(len(beards_in_select) * interp_num, 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    beard_idxes = []
    for color in beards_in_select:
        beard_idxes.append(selected_attrs.index(color))
    for color in beards_in_select:
        for alpha in range(start, end, step):
            alpha /= div
            attr = torch.zeros((attr_channel))
            for idx in beard_idxes:
                if attr_batch[0, idx] == 1:
                    attr[idx] = -1
            attr[selected_attrs.index(color)] = 1
            if attr_batch[0, selected_attrs.index(color)] == 1:
                attr[selected_attrs.index(color)] = 0  # 0 means keep current
            attrs.append(alpha * attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, beard_idxes


def interpolate_other_attr(img_batch, attr_batch, selected_attrs, interp_num, start, end, step, div, out=False):
    assert img_batch.size(0) == 1
    other_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                   'Bags_Under_Eyes', 'Bangs', 'Big_Lips', 'Big_Nose',
                   'Blurry', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                   'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                   'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                   'Rosy_Cheeks', 'Smiling', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                   'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    change_in_select = [a for a in other_attrs if a in selected_attrs]

    # change_attr = ['Arched_Eyebrows', 'Eyeglasses', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Young']
    # change_attr = ['5_o_Clock_Shadow', 'Bald', 'Bangs', 'Eyeglasses', 'Goatee', 'Male', 'Mustache', 'Pale_Skin',
    #                'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Hat', 'Young']
    new_img_batch = img_batch.repeat(len(change_in_select) * interp_num, 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []
    change_one_idxes = []
    for one in change_in_select:
        change_one_idxes.append(selected_attrs.index(one))
    for change in change_in_select:
        for alpha in range(start, end, step):
            alpha /= div
            attr = torch.zeros((attr_channel))
            idx = selected_attrs.index(change)
            if attr_batch[0, idx] == 1:
                attr[idx] = -1
            else:
                attr[idx] = 1
            attrs.append(alpha * attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, change_one_idxes


def change_hair_color_target(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    hair_colors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    new_img_batch = img_batch.repeat(len(hair_colors), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    hair_color_idxes = []
    for color in hair_colors:
        hair_color_idxes.append(selected_attrs.index(color))
    for color_idx in hair_color_idxes:
        attr = attr_batch.clone().detach()[0]
        for idx in hair_color_idxes:
            attr[idx] = 0
        attr[color_idx] = 1
        attrs.append(attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, hair_color_idxes


def change_skin_color_target(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    skin_colors = ['Skin_0', 'Skin_1', 'Skin_2', 'Skin_3']
    new_img_batch = img_batch.repeat(len(skin_colors), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    skin_color_idxes = []
    for color in skin_colors:
        skin_color_idxes.append(selected_attrs.index(color))
    for color_idx in skin_color_idxes:
        attr = attr_batch.clone().detach()[0]
        for idx in skin_color_idxes:
            attr[idx] = 0
        attr[color_idx] = 1
        attrs.append(attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, skin_color_idxes


def change_beard_target(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    beards = ['Mustache', 'No_Beard']
    new_img_batch = img_batch.repeat(len(beards), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []

    beard_idxes = []
    for color in beards:
        beard_idxes.append(selected_attrs.index(color))
    for color_idx in beard_idxes:
        attr = attr_batch.clone().detach()[0]
        for idx in beard_idxes:
            attr[idx] = 0
        attr[color_idx] = 1
        attrs.append(attr.reshape(1, attr_channel))
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, beard_idxes


def change_one_attr_target(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    change_attr = ['Arched_Eyebrows', 'Eyeglasses', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Young']
    new_img_batch = img_batch.repeat(len(change_attr), 1, 1, 1)
    attr_channel = len(selected_attrs)
    attrs = []
    change_one_idxes = []
    for change in change_attr:
        attr = attr_batch.clone().detach()[0]
        idx = selected_attrs.index(change)
        attr[idx] = 1.0 - attr[idx]
        attrs.append(attr.reshape(1, attr_channel))
        change_one_idxes.append(idx)
    test_attr = torch.cat(attrs, 0)
    return new_img_batch, test_attr, change_one_idxes

def get_attribute_indexs(selected_attrs):
    if len(selected_attrs) == 44:
        return get_attribute_indexs_all(selected_attrs)
    elif len(selected_attrs) == 17:
        return get_attribute_indexs_17(selected_attrs)
    else:
        raise NotImplementedError


def get_attribute_indexs_17(selected_attrs):
    hair_colors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    skin_colors = ['Skin_0', 'Skin_1', 'Skin_2', 'Skin_3']
    beards = ['Mustache', 'No_Beard']
    other_attrs = ['Arched_Eyebrows', 'Eyeglasses', 'Heavy_Makeup', 'Male', 'Mouth_Slightly_Open', 'Smiling', 'Young']

    hair_color_idxes = []
    for c in hair_colors:
        if c in selected_attrs:
            hair_color_idxes.append(selected_attrs.index(c))
    skin_color_idxes = []
    for c in skin_colors:
        if c in selected_attrs:
            skin_color_idxes.append(selected_attrs.index(c))
    beard_idxes = []
    for b in beards:
        if b in selected_attrs:
            beard_idxes.append(selected_attrs.index(b))
    other_idxes = []
    for o in other_attrs:
        if o in selected_attrs:
            other_idxes.append(selected_attrs.index(o))
    return [], hair_color_idxes, skin_color_idxes, beard_idxes, other_idxes


'''
default=['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Young', 'Skin_0', 'Skin_1', 'Skin_2', 'Skin_3'])
'''


def get_attribute_indexs_all(selected_attrs):
    hair_types = ['Bald', 'Straight_Hair', 'Wavy_Hair']
    hair_colors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
    skin_colors = ['Skin_0', 'Skin_1', 'Skin_2', 'Skin_3']
    beards = ['Goatee', 'Mustache', 'No_Beard', 'Sideburns']  # 两鬓胡须
    other_attrs = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                   'Bags_Under_Eyes', 'Bangs', 'Big_Lips', 'Big_Nose',
                   'Blurry', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses',
                   'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                   'Narrow_Eyes', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                   'Rosy_Cheeks', 'Smiling', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                   'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    hair_type_indexs = []
    for t in hair_types:
        if t in selected_attrs:
            hair_type_indexs.append(selected_attrs.index(t))

    hair_color_idxes = []
    for c in hair_colors:
        if c in selected_attrs:
            hair_color_idxes.append(selected_attrs.index(c))
    skin_color_idxes = []
    for c in skin_colors:
        if c in selected_attrs:
            skin_color_idxes.append(selected_attrs.index(c))
    beard_idxes = []
    for b in beards:
        if b in selected_attrs:
            beard_idxes.append(selected_attrs.index(b))
    other_idxes = []
    for o in other_attrs:
        if o in selected_attrs:
            other_idxes.append(selected_attrs.index(o))
    return hair_type_indexs, hair_color_idxes, skin_color_idxes, beard_idxes, other_idxes


def prepare_attr_eval_batch(data_batch, selected_attrs):
    attr_channel = len(selected_attrs)
    hair_type_indexs, hair_color_idxes, skin_color_idxes, beard_idxes, other_idxes = get_attribute_indexs(selected_attrs)
    img_batch, attr_batch = data_batch['real_img'], data_batch['real_cond']
    data_batch['target_idx'] = torch.zeros((img_batch.size(0), 1), dtype=torch.long)

    attrs = []
    for i in range(img_batch.size(0)):
        cur_attr_idx = i % attr_channel
        data_batch['target_idx'][i] = cur_attr_idx
        attr = torch.zeros((attr_channel))
        if cur_attr_idx in hair_type_indexs:
            for idx in hair_type_indexs:
                if attr_batch[i, idx] == 1:
                    attr[idx] = -1
            attr[cur_attr_idx] = 1
            if attr_batch[i, cur_attr_idx] == 1:
                attr[cur_attr_idx] = 0  # 0 means keep current
            attrs.append(attr.reshape(1, attr_channel))

        elif cur_attr_idx in hair_color_idxes:
            for idx in hair_color_idxes:
                if attr_batch[i, idx] == 1:
                    attr[idx] = -1
            attr[cur_attr_idx] = 1
            if attr_batch[i, cur_attr_idx] == 1:
                attr[cur_attr_idx] = 0  # 0 means keep current
            attrs.append(attr.reshape(1, attr_channel))

        elif cur_attr_idx in skin_color_idxes:
            for idx in skin_color_idxes:
                if attr_batch[i, idx] == 1:
                    attr[idx] = -1
            attr[cur_attr_idx] = 1
            if attr_batch[i, cur_attr_idx] == 1:
                attr[cur_attr_idx] = 0  # 0 means keep current
            attrs.append(attr.reshape(1, attr_channel))

        elif cur_attr_idx in beard_idxes:
            for idx in beard_idxes:
                if attr_batch[i, idx] == 1:
                    attr[idx] = -1
            attr[cur_attr_idx] = 1
            if attr_batch[i, cur_attr_idx] == 1:
                attr[cur_attr_idx] = 0  # 0 means keep current
            attrs.append(attr.reshape(1, attr_channel))

        elif cur_attr_idx in other_idxes:
            if attr_batch[i, cur_attr_idx] == 1:
                attr[cur_attr_idx] = -1
            else:
                attr[cur_attr_idx] = 1
            attrs.append(attr.reshape(1, attr_channel))

        else:
            raise ValueError

    data_batch['attr_delta'] = torch.cat(attrs, 0)
    data_batch['attr_target'] = data_batch['real_cond'] + data_batch['attr_delta']
    data_batch['desired_cond'] = data_batch['attr_target']
    data_batch['attr_target'] = data_batch['attr_target'].type(torch.LongTensor)
    return data_batch


def create_random_target_label(selected_attrs):
    # this dataset use random target label, each attribute are randomly set
    label = [0.0 for _ in range(len(selected_attrs))]
    hair_type_indexs, hair_color_idxes, skin_color_idxes, beard_idxes, other_idxes = get_attribute_indexs(selected_attrs)
    for other_idx in other_idxes:
        r = random.random()
        if r >= 0.5:
            label[other_idx] = 1.0
    if len(hair_type_indexs) > 0:
        hair_t_idx = random.choices(hair_type_indexs, k=1)[0]
        label[hair_t_idx] = 1.0
    if len(hair_color_idxes) > 0:  # prevent from raising error from choices
        hair_idx = random.choices(hair_color_idxes, k=1)[0]
        label[hair_idx] = 1.0
    if len(skin_color_idxes) > 0:  # prevent from raising error from choices
        skin_idx = random.choices(skin_color_idxes, k=1)[0]
        label[skin_idx] = 1.0
    if len(beard_idxes) > 0:  # prevent from raising error from choices
        beard_idx = random.choices(beard_idxes, k=1)[0]
        label[beard_idx] = 1.0
    return label


def create_random_label(selected_attrs, label):
    # this dataset is used for random change the label, apart from the other one, random set 0 or 1
    randomize_num = random.randint(1, 3)
    # Since for the each value of hair color, skin color and beard type is mutal exclusion
    # we compress to one label to stand all the coresponding attribute for random choose
    # this label is for real sample, we have to make it mutal exclusion
    hair_types_idxes, hair_color_idxes, skin_color_idxes, beard_idxes, other_idxes = get_attribute_indexs(selected_attrs)

    optimized_attr_idxs_num = 1 + 1 + 1 + 1 + len(other_idxes)
    optimized_attr_idxs = [i for i in range(optimized_attr_idxs_num)]
    randomize_idxes = random.choices(optimized_attr_idxs, k=randomize_num)
    randomized_label = copy.deepcopy(label)

    for chg_idx in randomize_idxes:
        if chg_idx == 0:  # change hair type
            cur_hair_type_idx = -1
            for t_idx in hair_types_idxes:
                if label[t_idx] == 1.0:
                    cur_hair_type_idx = t_idx
            change_hair_type_idx = cur_hair_type_idx  # if no color is 1, this will be correct too
            while change_hair_type_idx == cur_hair_type_idx:
                change_hair_type_idx = random.choices(hair_types_idxes, k=1)[0]
            randomized_label[cur_hair_type_idx] = 0.0
            randomized_label[change_hair_type_idx] = 1.0

        elif chg_idx == 1:  # change hair color
            cur_hair_color_idx = -1
            for h_idx in hair_color_idxes:
                if label[h_idx] == 1.0:
                    cur_hair_color_idx = h_idx
            change_hair_color_idx = cur_hair_color_idx  # if no color is 1, this will be correct too
            while change_hair_color_idx == cur_hair_color_idx:
                change_hair_color_idx = random.choices(hair_color_idxes, k=1)[0]
            randomized_label[cur_hair_color_idx] = 0.0
            randomized_label[change_hair_color_idx] = 1.0

        elif chg_idx == 2:  # change skin color
            cur_skin_color_idx = -1
            for s_idx in skin_color_idxes:
                if label[s_idx] == 1.0:
                    cur_skin_color_idx = s_idx
            change_skin_color_idx = cur_skin_color_idx
            while change_skin_color_idx == cur_skin_color_idx:
                change_skin_color_idx = random.choices(skin_color_idxes, k=1)[0]
            randomized_label[cur_skin_color_idx] = 0.0
            randomized_label[change_skin_color_idx] = 1.0

        elif chg_idx == 3:  # change beard type:
            cur_beard_color_idx = -1
            for b_idx in beard_idxes:
                if label[b_idx] == 1.0:
                    cur_beard_color_idx = b_idx
            change_beard_color_idx = cur_beard_color_idx
            while change_beard_color_idx == cur_beard_color_idx:
                change_beard_color_idx = random.choices(beard_idxes, k=1)[0]
            randomized_label[cur_beard_color_idx] = 0.0
            randomized_label[change_beard_color_idx] = 1.0

        else:  # for other attributes we can simply invert it
            other_idx = other_idxes[chg_idx-4]
            randomized_label[other_idx] = 1.0 - randomized_label[other_idx]

    return randomized_label


def change_multi_attr(img_batch, attr_batch, selected_attrs):
    assert img_batch.size(0) == 1
    attr_batch_label = attr_batch.squeeze(0).tolist()
    target_attr_label = create_random_label(selected_attrs, attr_batch_label)
    target_attr = torch.tensor(target_attr_label, device=img_batch.device)
    target_attr = target_attr.unsqueeze(0)
    delta_attr = target_attr - attr_batch
    return delta_attr


def get_selected_attr_from_all(output, selected_attrs):
    assert isinstance(output, list)  # the resnet18 output is a list of all attribute
    assert len(output) == len(selected_attrs)
    pred_attr = [0.0] * len(selected_attrs)
    for attr_i, attr_v in enumerate(output):
        assert attr_v.size(0) == 1, "For testing we set batch size as 1"
        pred_attr[attr_i] = torch.argmax(attr_v, dim=1).float().item()
    pred_attr_tensor = torch.tensor(pred_attr).unsqueeze(0)  # will be on cpu, just as dataloader
    return pred_attr_tensor
