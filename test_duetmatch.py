import os
import argparse
import torch
import pdb

from networks.net_factory import net_factory
from utils.test_3d_patch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default="../data/brats2019/3d", help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='test', help='exp_name')
parser.add_argument('--model', type=str,  default='VNet', help='model_name')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--detail', type=int,  default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=1, help='apply NMS post-processing?')
parser.add_argument('--labelnum', type=int, default=10, help='labeled data')
parser.add_argument('--stage_name',type=str, default='self_train', help='self_train or pre_train')

FLAGS = parser.parse_args()
dataset = 'test'
if 'brats2019' in FLAGS.root_path:
    dataset = 'brats2019'
elif 'brats2018' in FLAGS.root_path:
    dataset = 'brats2018'
elif 'brats2017' in FLAGS.root_path:
    dataset = 'brats2017'
elif 'isles2022' in FLAGS.root_path:
    dataset = 'isles2022'
print('Testing on', dataset)

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "/content/drive/MyDrive/Research/models/DualMatch/{}/ours/{}_{}_labeled/{}".format(dataset, FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
test_save_path = "/content/drive/MyDrive/Research/models/DualMatch/{}/ours/{}_{}_labeled/{}_predictions/".format(dataset, FLAGS.exp, FLAGS.labelnum, FLAGS.model)
num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/data/" + item.replace('\n', '') for item in image_list]

def test_calculate_metric():
    model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode="test")
    save_model_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(FLAGS.model))
    
    model.load_state_dict(torch.load(save_model_path))
    print("init weight from {}".format(save_model_path))

    model.eval()

    avg_metric = test_all_case(model, image_list, num_classes=num_classes,
                           patch_size=(96, 96, 96), stride_xy=64, stride_z=64,
                           save_result=True, test_save_path=test_save_path,
                           metric_detail=FLAGS.detail, nms=FLAGS.nms)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)

# python test_LA.py --model 0214_re01 --gpu 0