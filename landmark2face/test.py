import os
from options.test_options import TestOptions
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import cv2
import torchvision.transforms as transforms
from PIL import Image
from models.l2face_model import *
import numpy as np
import sys
from util.util import *
from APB.APBDataset import *
from APB.APBNet import *
import torch


def tuple_shape(shape):
    r_data = []
    for p in shape:
        r_data.append([p.x, p.y])
    return r_data


def drawCircle(img, shape, radius=1, color=(255, 255, 255), thickness=1):
    for p in shape:
        img = cv2.circle(img, (int(p[0]), int(p[1])), radius, color, thickness)
    return img


def vector2points(landmark):
    shape = []
    for i in range(len(landmark) // 2):
        shape.append([landmark[2 * i], landmark[2 * i + 1]])
    return shape


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.isTrain = False
    opt.name = 'man1_Res9'
    opt.model = 'l2face'
    opt.netG = 'resnet_9blocks_l2face'
    opt.dataset_mode = 'l2face'
    model = L2FaceModel(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()
    transforms_label = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # audio2landmark
    audio_net = APBNet()
    checkpoint = torch.load('APB/man1_best.pth')
    audio_net.load_state_dict(checkpoint['net_G'])
    audio_net.cuda()
    audio_net.eval()
    # dataset
    feature_path = '../AnnVI/feature'
    idt_name = 'man1'
    testset = APBDataset(feature_path, idt_name=idt_name, mode='test', img_size=256)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    out_path = 'result'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(out_path, '{}.avi'.format(idt_name)), fourcc, 25.0, (256 * 2, 256))

    for idx, data in enumerate(dataloader):
        audio_feature_A1, pose_A1, eye_A1 = data[0][0].cuda(), \
                                            data[0][1].cuda(), \
                                            data[0][2].cuda()
        landmark_A1, landmark_A2 = data[1][0].cuda(),\
                                   data[1][1].cuda()

        image_path_A1 = data[2][0][0][0]
        print('\r{}/{}'.format(idx+1, len(dataloader)), end='')

        landmark = audio_net(audio_feature_A1, pose_A1, eye_A1)
        landmark = landmark.cpu().data.numpy().tolist()[0]
        lab_template = np.zeros((256, 256, 3)).astype(np.uint8)
        lab = drawCircle(lab_template.copy(), vector2points(landmark), radius=1, color=(255, 255, 255), thickness=4)
        lab = Image.fromarray(lab).convert('RGB')
        lab = transforms_label(lab).unsqueeze(0)

        input_data = {'A': lab, 'A_label': lab, 'B': lab, 'B_label': lab}
        model.set_input(input_data)
        model.test()
        visuals = model.get_current_visuals()
        B_img_f = tensor2im(visuals['fake_B'])
        B_img = cv2.imread(image_path_A1)
        B_img = cv2.cvtColor(B_img, cv2.COLOR_BGR2RGB)
        B_img = cv2.resize(B_img, (256, 256))

        img_out = np.concatenate([B_img_f, B_img], axis=1)
        for _ in range(5):  # five times slower
            out.write(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
        if idx == 100:
            break
    out.release()
