import argparse
import logging
import sys
from APBDataset import *
from APBGAN import *
from utils import *


parser = argparse.ArgumentParser(description='APBNet')
parser.add_argument('--isTrain', default=True, type=bool, help='running mode')
parser.add_argument('--lr', default=0.0003, type=float, help='learning rate')
parser.add_argument('--every', default=300, type=float, help='learning rate decay')
parser.add_argument('--gpus', default='1', type=str, help='gpus')
parser.add_argument('--landmark_path', default='../AnnVI/feature', type=str, help='landmark path that contains several persons')
parser.add_argument('--checkpoints', default='checkpoints', type=str, help='checkpoint path')
parser.add_argument('--epochs', default=800, type=int, help='epochs')
parser.add_argument('--resume', '-r', default=False, type=bool, help='resume')
parser.add_argument('--resume_epoch', default=None, type=int, help='resume epoch')
parser.add_argument('--resume_name', default='man1-20200428-134038', type=str, help='resume epoch')
parser.add_argument('--idt_name', default='man1', type=str, help='identity name')

opt = parser.parse_args()
opt.gpus = [int(dev) for dev in opt.gpus.split(',')]
torch.cuda.set_device(opt.gpus[0])

# logging
if not os.path.exists(opt.checkpoints):
    os.mkdir(opt.checkpoints)
if opt.resume:
    opt.logdir = '{}/{}'.format(opt.checkpoints, opt.resume_name)
else:
    opt.logdir = '{}/{}-{}'.format(opt.checkpoints, opt.idt_name, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(opt.logdir):
    os.mkdir(opt.logdir)

log_format = '%(asctime)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(opt.logdir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logger = logging.getLogger()
logger.addHandler(fh)

for key, val in vars(opt).items():
    if isinstance(val, list):
        val = [str(v) for v in val]
        val = ','.join(val)
    if val is None:
        val = 'None'
    logger.info('{:>20} : {:<50}'.format(key, val))

logger.info('==> Preparing data..')
trainset = APBDataset(opt.landmark_path, opt.idt_name, mode='train')
testset = APBDataset(opt.landmark_path, opt.idt_name, mode='test')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=1)


logger.info('==> Building model..')
net = GANModel(opt, logger)

def train(epoch):
    net.train()
    net.run_train(trainloader)

def test(epoch):
    net.eval()
    net.run_test(testloader)

# drawloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
# net.test_draw(drawloader)

for epoch in range(1, opt.epochs):
    train(epoch)
    test(epoch)
    logger.info('-' * 50)
