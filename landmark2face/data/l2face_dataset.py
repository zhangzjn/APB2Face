import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class L2FaceDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        img_size = opt.img_size
        root = '../AnnVI/feature/{}'.format(opt.name.split('_')[0])
        image_dir = '{}/{}_image_crop'.format(root, img_size)
        label_dir = '{}/{}_landmark_crop_thin'.format(root, img_size)
        # label_dir = '{}/512_landmark_crop'.format(root)
        self.labels = []

        imgs = os.listdir(image_dir)
        # if 'man' in opt.name:
        #     imgs.sort(key=lambda x:int(x.split('.')[0]))
        # else:
        #     imgs.sort(key=lambda x: (int(x.split('.')[0].split('-')[0]), int(x.split('.')[0].split('-')[1])))
        for img in imgs:
            img_path = os.path.join(image_dir, img)
            lab_path = os.path.join(label_dir, img)
            if os.path.exists(lab_path):
                self.labels.append([img_path, lab_path])
        # transforms.Resize([img_size, img_size], Image.BICUBIC),
        self.transforms_image = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transforms.Resize([img_size, img_size], Image.BICUBIC),
        self.transforms_label = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.shuffle()

    def shuffle(self):
        random.shuffle(self.labels)


    def __getitem__(self, index):
        img_path, lab_path = self.labels[index]
        img = Image.open(img_path).convert('RGB')
        lab = Image.open(lab_path).convert('RGB')
        img = self.transforms_image(img)
        lab = self.transforms_label(lab)

        imgA_path, labA_path = random.sample(self.labels, 1)[0]
        imgA = Image.open(imgA_path).convert('RGB')
        imgA = self.transforms_image(imgA)


        return {'A': imgA, 'A_label': lab, 'B': img, 'B_label': lab}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.labels)


if __name__ == '__main__':
    from options.train_options import TrainOptions
    opt = TrainOptions().parse()
    dataset = L2FaceDataset(opt)
    dataset_size = len(dataset)
    print(dataset_size)
    for i, data in enumerate(dataset):
        print(data)
