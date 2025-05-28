#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
优化训练启动脚本
包含自动环境检测、wandb配置和训练启动
"""

import os
import sys
import subprocess
import json
import argparse


def check_environment():
    """检查训练环境"""
    print("🔍 检查训练环境...")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory / 1024**3
            print(f"✅ GPU可用: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"   可用GPU数量: {gpu_count}")
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
        else:
            print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查必要的依赖
    required_packages = ['wandb', 'numpy', 'matplotlib', 'seaborn', 'tqdm', 'tensorboardX']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} 未安装")
    
    if missing_packages:
        print(f"\n请安装缺失的包: pip install {' '.join(missing_packages)}")
        return False
    
    return True


def setup_wandb():
    """设置wandb"""
    print("\n🔧 设置Wandb...")
    
    try:
        import wandb
        
        # 检查是否已登录
        if wandb.api.api_key is None:
            print("请先登录wandb:")
            print("1. 访问 https://wandb.ai/")
            print("2. 注册/登录账户")
            print("3. 复制API key")
            print("4. 运行: wandb login")
            
            api_key = input("或者直接在这里输入API key (留空跳过): ").strip()
            if api_key:
                wandb.login(key=api_key)
                print("✅ Wandb登录成功")
            else:
                print("⚠️  跳过wandb登录，将在离线模式运行")
                os.environ['WANDB_MODE'] = 'offline'
        else:
            print("✅ Wandb已登录")
            
    except ImportError:
        print("❌ Wandb未安装")
        return False
    
    return True


def optimize_config(config_path):
    """根据系统配置优化训练参数"""
    print(f"\n⚙️  优化配置文件: {config_path}")
    
    try:
        import torch
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            # config['datasets']['train']['batch_size'] = 8
            # config['datasets']['train']['num_workers'] = 12

        # # 根据GPU内存调整batch size
        # if torch.cuda.is_available():
        #     gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
        #     if gpu_memory >= 16:
        #         config['datasets']['train']['batch_size'] = 12
        #         config['datasets']['train']['num_workers'] = 16
        #         print(f"✅ 高端GPU检测 ({gpu_memory:.1f}GB)，batch_size设置为12")
        #     elif gpu_memory >= 8:
        #         print(f"✅ 中端GPU检测 ({gpu_memory:.1f}GB)，batch_size设置为8")
        #     else:
        #         config['datasets']['train']['batch_size'] = 4
        #         config['datasets']['train']['num_workers'] = 8
        #         print(f"✅ 低端GPU检测 ({gpu_memory:.1f}GB)，batch_size设置为4")
        # else:
        #     config['datasets']['train']['batch_size'] = 2
        #     config['datasets']['train']['num_workers'] = 4
        #     print("⚠️  CPU模式，batch_size设置为2")
        
        # 检查数据集路径
        train_dataroot = config['datasets']['train']['dataroot']
        val_dataroot = config['datasets']['val']['dataroot']
        
        if not os.path.exists(train_dataroot):
            print(f"⚠️  训练数据集路径不存在: {train_dataroot}")
            print("请确保数据集已正确下载和解压")
        
        if not os.path.exists(val_dataroot):
            print(f"⚠️  验证数据集路径不存在: {val_dataroot}")
            print("请确保数据集已正确下载和解压")
        
        # 保存优化后的配置
        # optimized_config_path = config_path.replace('.json', '_auto_optimized.json')
        # with open(optimized_config_path, 'w') as f:
        #     json.dump(config, f, indent=2)
        
        # print(f"✅ 优化配置保存到: {optimized_config_path}")
        return config_path
        
    except Exception as e:
        print(f"❌ 配置优化失败: {e}")
        return config_path


def create_directories():
    """创建必要的目录"""
    base_dir = "F:/SR3_training_result"
    directories = [
        'logs', 'tb_logger', 'results', 'checkpoint', 'param_outputs'
    ]
    
    # 创建基础目录
    os.makedirs(base_dir, exist_ok=True)
    
    # 创建子目录
    for dir_name in directories:
        full_path = os.path.join(base_dir, dir_name)
        os.makedirs(full_path, exist_ok=True)
    
    print("✅ 创建输出目录完成")


def main():
    parser = argparse.ArgumentParser(description='优化训练启动脚本')
    parser.add_argument('--config', '-c', type=str, 
                       default='config/517lc_optimized.json',
                       help='配置文件路径')
    parser.add_argument('--resume', '-r', type=str, default=None,
                       help='恢复训练的checkpoint路径')
    parser.add_argument('--no-wandb', action='store_true',
                       help='禁用wandb日志')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅检查环境，不开始训练')
    
    args = parser.parse_args()
    
    print("🚀 开始优化训练配置...")
    print("=" * 60)
    
    # 环境检查
    if not check_environment():
        print("\n❌ 环境检查失败，请安装必要依赖后重试")
        sys.exit(1)
    
    # 设置wandb
    if not args.no_wandb:
        if not setup_wandb():
            print("\n⚠️  Wandb设置失败，将禁用wandb日志")
            args.no_wandb = True
    
    # 创建目录
    create_directories()
    
    # 优化配置
    optimized_config = optimize_config(args.config)
    
    if args.dry_run:
        print("\n✅ 环境检查完成 (dry-run模式)")
        return
    
    # 构建训练命令
    cmd = [
        sys.executable, 'sr_optimized.py',
        '--config', optimized_config,
        '--phase', 'train'
    ]
    
    if not args.no_wandb:
        cmd.extend(['--enable_wandb', '--log_wandb_ckpt', '--log_eval'])
    
    if args.resume:
        # 修改配置文件中的resume路径
        with open(optimized_config, 'r') as f:
            config = json.load(f)
        config['path']['resume_state'] = args.resume
        with open(optimized_config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ 设置恢复训练路径: {args.resume}")
    
    print("\n🚀 开始训练...")
    print("命令:", ' '.join(cmd))
    print("=" * 60)
    
    # 启动训练
    try:
        subprocess.run(cmd, check=True)
        print("\n🎉 训练完成!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        sys.exit(0)


if __name__ == "__main__":
    main() 