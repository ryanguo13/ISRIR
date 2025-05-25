import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from core.param_logger import ParameterLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from tqdm import tqdm
import time
import torch.cuda.amp as amp


def export_onnx_model(model, checkpoint_dir, step):
    """导出ONNX模型"""
    try:
        # 创建ONNX输出目录
        onnx_dir = os.path.join(checkpoint_dir, 'onnx')
        os.makedirs(onnx_dir, exist_ok=True)
        
        # 设置模型为评估模式
        model.eval()
        
        # 创建示例输入 (低分辨率图像16x16 + 条件噪声)
        # 批次大小=1, 通道=6 (3通道LR + 3通道噪声), 16x16
        dummy_input = torch.randn(1, 6, 16, 16).cuda()
        
        # 如果模型被DataParallel包装，获取原始模型
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        
        # ONNX导出路径
        onnx_path = os.path.join(onnx_dir, f'sr3_model_step_{step}.onnx')
        
        # 导出ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['lr_image'],
            output_names=['sr_image'],
            dynamic_axes={
                'lr_image': {0: 'batch_size'},
                'sr_image': {0: 'batch_size'}
            }
        )
        
        return onnx_path
        
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}")
    finally:
        # 恢复训练模式
        model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/517lc_optimized.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--log_wandb_ckpt', action='store_true')
    parser.add_argument('--log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 启用混合精度训练以提高速度
    use_amp = True
    scaler = amp.GradScaler(
        init_scale=2.**16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True
    )

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        wandb.define_metric("training/*", step_metric="step")
        val_step = 0
    else:
        wandb_logger = None

    # Initialize Parameter Logger
    param_log_dir = opt['path'].get('param_output', 'param_outputs')
    param_logger = ParameterLogger(opt, param_log_dir)
    logger.info(f'Parameter logging enabled: {param_logger.enabled}')

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    
    # Register activation hooks for parameter logging
    if param_logger.enabled:
        param_logger.register_activation_hooks(diffusion.netG)

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    
    # 获取参数保存频率
    save_param_freq = opt['train'].get('save_param_freq', 100)
    
    if opt['phase'] == 'train':
        start_time = time.time()
        
        while current_step < n_iter:
            current_epoch += 1
            epoch_start_time = time.time()
            
            for batch_idx, train_data in tqdm(enumerate(train_loader), 
                                            desc=f"Epoch {current_epoch}", 
                                            total=len(train_loader)):
                current_step += 1
                if current_step > n_iter:
                    break
                
                step_start_time = time.time()
                
                # 使用混合精度训练
                if use_amp:
                    with amp.autocast():
                        diffusion.feed_data(train_data)
                        loss = diffusion.optimize_parameters_amp(scaler)
                else:
                    diffusion.feed_data(train_data)
                    diffusion.optimize_parameters()
                
                step_time = time.time() - step_start_time
                
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    logs['step_time'] = step_time
                    logs['lr'] = diffusion.optG.param_groups[0]['lr']
                    
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger and current_step % (opt['train']['print_freq'] * 2) == 0:
                        wandb_logs = {'training/' + k: v for k, v in logs.items()}
                        wandb_logs['step'] = current_step
                        wandb_logger.log_metrics(wandb_logs)
                    
                    # 记录损失组件
                    if param_logger.enabled:
                        param_logger.log_loss_components(logs, current_step)

                # 参数记录
                if param_logger.enabled and current_step % save_param_freq == 0:
                    logger.info(f'Saving parameters at step {current_step}')
                    
                    # 获取扩散模型用于噪声调度记录
                    diffusion_model = diffusion.netG.module if isinstance(diffusion.netG, torch.nn.DataParallel) else diffusion.netG
                    
                    param_logger.log_model_weights(diffusion.netG, current_step)
                    param_logger.log_gradients(diffusion.netG, current_step)
                    param_logger.log_activations(current_step)
                    param_logger.log_noise_schedule(diffusion_model, current_step)
                    param_logger.create_visualizations(current_step)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    val_start_time = time.time()
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    
                    # 减少验证样本数量以提高速度
                    max_val_samples = min(len(val_loader), 3)
                    
                    for val_idx, val_data in enumerate(val_loader):
                        if val_idx >= max_val_samples:
                            break
                            
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                        lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
                        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        
                        # 只在第一个样本上添加tensorboard图像
                        if idx == 1:
                            tb_logger.add_image(
                                'Iter_{}'.format(current_step),
                                np.transpose(np.concatenate(
                                    (fake_img, sr_img, hr_img), axis=1), [2, 0, 1]),
                                idx)
                        
                        avg_psnr += Metrics.calculate_psnr(sr_img, hr_img)

                        if wandb_logger and idx <= 2:  # 只记录前2个样本到wandb
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    val_time = time.time() - val_start_time
                    
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    
                    # log
                    logger.info('# Validation # PSNR: {:.4e}, Time: {:.2f}s'.format(avg_psnr, val_time))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('val_time', val_time, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_time': val_time,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)
                    
                    # 导出ONNX模型 (每5000步导出一次，减少频率)
                    if current_step % (opt['train']['save_checkpoint_freq'] * 5) == 0:
                        try:
                            logger.info('Exporting ONNX model...')
                            export_onnx_model(diffusion.netG, opt['path']['checkpoint'], current_step)
                            logger.info(f'ONNX model exported at step {current_step}')
                        except Exception as e:
                            logger.warning(f'ONNX export failed: {e}')

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            epoch_time = time.time() - epoch_start_time
            logger.info(f'Epoch {current_epoch} completed in {epoch_time:.2f}s')
            
            if wandb_logger:
                wandb_logger.log_metrics({
                    'epoch': current_epoch-1,
                    'epoch_time': epoch_time
                })

        # 训练结束后保存最终统计信息
        if param_logger.enabled:
            param_logger.save_summary_statistics()
            logger.info(f'Parameter logs saved to: {param_log_dir}')

        total_time = time.time() - start_time
        logger.info(f'Training completed in {total_time:.2f}s ({total_time/3600:.2f}h)')
        tb_logger.close()
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
            lr_img = Metrics.tensor2img(visuals['LR'])  # uint8
            fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            }) 