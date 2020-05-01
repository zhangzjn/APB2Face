import torch
import torch.nn as nn


class APBNet(nn.Module):

    def __init__(self, num_landmark=212):
        super(APBNet, self).__init__()
        self.num_landmark = num_landmark
        # audio
        self.audio1 = nn.Sequential(
            nn.Conv2d(1, 72, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(72, 108, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(108, 162, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(162, 243, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
            nn.Conv2d(243, 256, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)), nn.ReLU(),
        )
        self.audio2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(4, 1), stride=(4, 1)), nn.ReLU()
        )
        self.trans_audio = nn.Sequential(nn.Linear(256 * 2, 256))
        # pose
        self.trans_pose = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        # eye
        self.trans_eye = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64)
        )
        # cat
        self.trans_cat = nn.Sequential(
            nn.Linear(256 + 64 * 2, 240), nn.ReLU(),
            nn.Linear(240, self.num_landmark)
        )

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, audio, pose, eye):
        x_a = self.audio1(audio)
        x_a = self.audio2(x_a)
        x_a = x_a.view(-1, self.num_flat_features(x_a))
        x_a = self.trans_audio(x_a)
        x_p = self.trans_pose(pose)
        x_e = self.trans_eye(eye)
        x_cat = torch.cat([x_a, x_p, x_e], dim=1)
        output = self.trans_cat(x_cat)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layers1 = [nn.Linear(106 * 2, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                  nn.Linear(512, 64)]

        layers2 = [nn.Linear(106 * 2, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 512), nn.LeakyReLU(0.2, True),
                   nn.Linear(512, 64)]

        layers3 = [nn.Linear(128, 128), nn.LeakyReLU(0.2, True),
                   nn.Linear(128, 32), nn.LeakyReLU(0.2, True),
                   nn.Linear(32, 1)]

        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)

    def forward(self, input1, input2):
        x1 = self.layers1(input1)
        x2 = self.layers2(input2)
        x_cat = torch.cat([x1, x2], dim=1)
        out = self.layers3(x_cat)
        return out


if __name__ == "__main__":
    torch.cuda.set_device(0)
    from APBDataset import *
    import time
    landmark_paths = '/media/datasets/zhangzjn/AnnVI/feature'
    testset = APBDataset(landmark_paths, 'man1')
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, shuffle=True, num_workers=1)
    net = Generator()
    for batch_idx, training_data in enumerate(testloader):
        audio_feature_A1, pose_A1, eye_A1 = training_data[0][0], training_data[0][1],\
                                            training_data[0][2]
        landmark = training_data[2][0]
        t_start = time.time()
        for i in range(10000):
            net(audio_feature_A1, landmark, pose_A1, eye_A1)
        print(time.time() - t_start)
        break