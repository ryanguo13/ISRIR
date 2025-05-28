from tqdm import tqdm
import torch
import torch.nn as nn
import data as Data
import traceback
import model as Model
import argparse
import core.logger as Logger
import os

# 重定向标准输出和标准错误到文件
if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # 启用确定性模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/517lc_optimized.json',
                        help='配置文件路径')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true', default=False)
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'], default='val')
    parser.add_argument('--enable_wandb', action='store_true', default=False)
    parser.add_argument('--log_wandb_ckpt', action='store_true', default=False)
    parser.add_argument('--log_eval', action='store_true', default=False)

    #os environs
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # 启用CUDA优化
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    diffusion = Model.create_model(opt)

    # 获取原始模型
    model = diffusion.netG
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    else:
        model.eval()
    
    # 创建包装模型

    # 创建示例输入
    dummy_input1 = torch.randn(1, 3, 128, 128, dtype=torch.float32).cuda()
    dummy_input2 = torch.randn(1, 3, 128, 128, dtype=torch.float32).cuda()
    