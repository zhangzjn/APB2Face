import random
from torch.utils.data import dataset
import torch
import os
import torchvision.transforms as transforms
import torch
import numpy as np


class APBDataset(dataset.Dataset):
	def __init__(self, root, idt_name='man1', mode='train', img_size=256):
		self.root = root
		self.idt_name = idt_name
		if not isinstance(mode, list):
			mode = [mode]

		self.data_all = list()
		for m in mode:
			training_data_path = os.path.join(self.root, self.idt_name, '{}_{}.t7'.format(img_size, m))
			training_data = torch.load(training_data_path)
			img_paths = training_data['img_paths']
			audio_features = training_data['audio_features']
			lands = training_data['lands']
			poses = training_data['poses']
			eyes = training_data['eyes']
			for i in range(len(img_paths)):
				img_path = [os.path.join(self.root, self.idt_name, p) for p in img_paths[i]]  # [image, landmark]
				audio_feature = audio_features[i]
				land = lands[i]
				pose = poses[i]
				eye = eyes[i]
				self.data_all.append([img_path, audio_feature, land, pose, eye])
		self.data_all.sort(key=lambda x: int(x[0][0].split('/')[-1].split('.')[0]))
		if 'train' in mode and len(mode) == 1:
			self.shuffle()

	def shuffle(self):
		random.shuffle(self.data_all)

	def __len__(self):
		return len(self.data_all)

	def __getitem__(self, index):
		img_path_A1, audio_feature_A1, land_A1, pose_A1, eye_A1 = self.data_all[index]
		img_path_A2, audio_feature_A2, land_A2, pose_A2, eye_A2 = random.sample(self.data_all, 1)[0]
		# audio
		audio_feature_A1 = torch.tensor(audio_feature_A1).unsqueeze(dim=0)
		# pose
		pose_A1 = torch.tensor(pose_A1)
		# eye
		eye_A1 = torch.tensor(eye_A1)
		# landmark
		land_A1 = torch.tensor(land_A1)
		land_A2 = torch.tensor(land_A2)


		return [audio_feature_A1, pose_A1, eye_A1], [land_A1, land_A2], [img_path_A1, img_path_A2]


if __name__ == '__main__':
	root = '/media/datasets/zhangzjn/AnnVI/feature'
	idt_name = 'man1'
	trainset = APBDataset(root, idt_name)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
	for batch_idx, _ in enumerate(trainloader):
		print(batch_idx)
