## APB2Face &mdash; Official PyTorch Implementation

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic) ![PyTorch 1.3.1](https://img.shields.io/badge/pytorch-1.3.1-green.svg?style=plastic) ![License MIT](https://img.shields.io/github/license/zhangzjn/APB2Face)

Official pytorch implementation of the paper "[APB2FACE: AUDIO-GUIDED FACE REENACTMENT WITH AUXILIARY POSE AND BLINK SIGNALS, ICASSP'20](https://arxiv.org/pdf/2004.14569.pdf)".

For any inquiries, please contact Jiangning Zhang at [186368@zju.edu.cn](mailto:186368@zju.edu.cn)

## Using the Code

### Requirements

This code has been developed under `Python3.7`, `PyTorch 1.3.1` and `CUDA 10.1` on `Ubuntu 16.04`. 


```shell
# Install python3 packages
pip3 install -r requirements.txt
```

### Inference

- Download pretraind [Audio-to-Landmark model](https://drive.google.com/file/d/159jQ27M_dqKmQ3ZacYZu6woXQ1f8Yc_H/view?usp=sharing) for the person **man1** to the path `landmark2face/APB/man1_best.pth`.
- Download pretraind [Landmark-to-Face model](https://drive.google.com/file/d/1UqjxWG2kNVfG3G65SxdEsTrlGg9KqBRU/view?usp=sharing) for the person **man1** to the path `landmark2face/checkpoints/man1_Res9/latest_net_G.pth`

```shell
python3 test.py 
```

You can view the result in `result/man1.avi`

### Training

1. Train **Audio-to-Landmark** model.

   ```shell
   python3 audio2landmark/main.py
   ```

2. Train **Landmark-to-Face** model.

   ```shell
   cd landmark2face
   sh experiments/train.sh
   ```
   you can watch the checkpoint in `checkpoints/man1_Res9`

3. Do following operations before the test.

   ```shell
   copy audio2landmark/APBNet.py landmark2face/APB/APBNet.py  # if you modify APBNet.py
   copy audio2landmark/APBDataset.py landmark2face/APB/APBDataset.py  # if you modify APBDataset.py
   copy audio2landmark/checkpoints/man1-xxx/man1_best.pth landmark2face/APB/man1_best.pth
   ```

## Datasets in the paper

We propose a new **AnnVI** dataset, you can download it from 
[Google Drive](https://drive.google.com/file/d/1xEnZwNLU4SmgFFh4WGV4KEOdegfFrOdp/view?usp=sharing) 
or 
[Baidu Cloud](https://pan.baidu.com/s/1oydpePBQieRoDmaENg3kfQ) (Key:str3).
### Citation

If you think this work is useful for your research, please consider citing:

```
@inproceedings{zhang2020apb2face,
  title={APB2FACE: Audio-Guided Face Reenactment with Auxiliary Pose and Blink Signals},
  author={Zhang, Jiangning and Liu, Liang and Xue, Zhucun and Liu, Yong},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={4402--4406},
  year={2020},
  organization={IEEE}
}
```

### Acknowledgements

We thank for the source code from the great work [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).