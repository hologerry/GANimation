import random
import torch.utils.data as data
import torchvision.transforms as T
from data.celeba_hq import CelebAHQ
from data.celebahq_ffhq_fake import CelebAHQFFHQFake
from data.wild_images import Wild


def get_image_transform(img_size, mode='train', crop='random', crop_prob=0.5, random_flip_ratio=1.0):
    transforms = []
    # if crop == 'random' and mode == 'train':
    #     crop = T.RandomResizedCrop(img_size, scale=[0.8, 1.0], ratio=[0.9, 1.1])
    #     rand_crop = T.Lambda(lambda x: crop(x) if random.random() < crop_prob else x)
    #     transforms.append(rand_crop)
    # elif crop == 'center' and mode == 'train':
    #     crop = T.CenterCrop(img_size)
    #     transforms.append(crop)

    transforms.append(T.Resize([img_size, img_size]))
    if random_flip_ratio > 0.0 and mode == 'train':
        transforms.append(T.RandomHorizontalFlip(random_flip_ratio))
    transforms += [T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]

    return T.Compose(transforms)



def get_dataloader(data_root='data_root', dataset_name='celebahq_ffhq_fake', mode='train', img_size=256, selected_attrs=None,
                   batch_size=2, random_flip_ratio=1.0):
    print(f'Preparing DataLoader to fetch {dataset_name} images in {mode}...')
    dataset = None
    transform_img = get_image_transform(img_size, mode=mode, random_flip_ratio=random_flip_ratio, crop=None)

    if dataset_name == 'celebahq':
        dataset = CelebAHQ(data_root, dataset_name, mode, transform_img, selected_attrs)
    elif dataset_name == 'celebahq_ffhq_fake':
        dataset = CelebAHQFFHQFake(data_root, dataset_name, mode, transform_img, selected_attrs)
    elif dataset_name == 'wild_images':
        dataset = Wild(data_root, dataset_name, mode, transform_img, selected_attrs)
    else:
        raise NotImplementedError
    print(f'Prepared DataLoader to fetch {dataset_name} images in {mode}')

    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           num_workers=batch_size,
                           shuffle=mode == 'train')
