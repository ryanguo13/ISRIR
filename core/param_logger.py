import torch
import numpy as np
import os
import json
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class ParameterLogger:
    def __init__(self, config, log_dir):
        self.config = config
        self.log_dir = log_dir
        self.enabled = config.get('param_logging', {}).get('enabled', False)
        
        if not self.enabled:
            return
            
        # 创建输出目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'weights'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'gradients'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'activations'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'noise_schedule'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'loss_components'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'statistics'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'visualizations'), exist_ok=True)
        
        self.param_config = config.get('param_logging', {})
        self.save_gradients = self.param_config.get('save_gradients', True)
        self.save_weights = self.param_config.get('save_weights', True)
        self.save_activations = self.param_config.get('save_activations', True)
        self.save_noise_schedule = self.param_config.get('save_noise_schedule', True)
        self.save_loss_components = self.param_config.get('save_loss_components', True)
        self.layers_to_monitor = self.param_config.get('layers_to_monitor', [])
        
        # 存储统计信息
        self.stats = defaultdict(list)
        self.activations = {}
        self.gradients = {}
        
    def should_monitor_layer(self, name):
        """判断是否应该监控某个层"""
        if not self.layers_to_monitor:
            return True
        
        for layer_type in self.layers_to_monitor:
            if layer_type.lower() in name.lower():
                return True
        return False
    
    def log_model_weights(self, model, step):
        """保存模型权重"""
        if not self.enabled or not self.save_weights:
            return
            
        weights_data = {}
        for name, param in model.named_parameters():
            if self.should_monitor_layer(name) and param.requires_grad:
                weights_data[name] = {
                    'weight': param.data.cpu().numpy(),
                    'shape': list(param.shape),
                    'mean': float(param.data.mean()),
                    'std': float(param.data.std()),
                    'min': float(param.data.min()),
                    'max': float(param.data.max()),
                    'norm': float(param.data.norm())
                }
        
        # 保存权重数据
        save_path = os.path.join(self.log_dir, 'weights', f'weights_step_{step}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(weights_data, f)
        
        # 保存统计信息
        stats_path = os.path.join(self.log_dir, 'statistics', f'weight_stats_step_{step}.json')
        weight_stats = {name: {k: v for k, v in data.items() if k != 'weight'} 
                       for name, data in weights_data.items()}
        with open(stats_path, 'w') as f:
            json.dump(weight_stats, f, indent=2)
    
    def log_gradients(self, model, step):
        """保存梯度信息"""
        if not self.enabled or not self.save_gradients:
            return
            
        gradient_data = {}
        for name, param in model.named_parameters():
            if self.should_monitor_layer(name) and param.grad is not None:
                grad = param.grad.data
                gradient_data[name] = {
                    'gradient': grad.cpu().numpy(),
                    'shape': list(grad.shape),
                    'mean': float(grad.mean()),
                    'std': float(grad.std()),
                    'min': float(grad.min()),
                    'max': float(grad.max()),
                    'norm': float(grad.norm()),
                    'zero_fraction': float((grad == 0).float().mean())
                }
        
        # 保存梯度数据
        save_path = os.path.join(self.log_dir, 'gradients', f'gradients_step_{step}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(gradient_data, f)
        
        # 保存统计信息
        stats_path = os.path.join(self.log_dir, 'statistics', f'gradient_stats_step_{step}.json')
        grad_stats = {name: {k: v for k, v in data.items() if k != 'gradient'} 
                     for name, data in gradient_data.items()}
        with open(stats_path, 'w') as f:
            json.dump(grad_stats, f, indent=2)
    
    def register_activation_hooks(self, model):
        """注册激活值钩子"""
        if not self.enabled or not self.save_activations:
            return
            
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = {
                        'output': output.detach().cpu().numpy(),
                        'shape': list(output.shape),
                        'mean': float(output.mean()),
                        'std': float(output.std()),
                        'min': float(output.min()),
                        'max': float(output.max())
                    }
            return hook
        
        for name, module in model.named_modules():
            if self.should_monitor_layer(name):
                module.register_forward_hook(hook_fn(name))
    
    def log_activations(self, step):
        """保存激活值"""
        if not self.enabled or not self.save_activations or not self.activations:
            return
            
        # 保存激活值数据
        save_path = os.path.join(self.log_dir, 'activations', f'activations_step_{step}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.activations, f)
        
        # 保存统计信息
        stats_path = os.path.join(self.log_dir, 'statistics', f'activation_stats_step_{step}.json')
        activation_stats = {name: {k: v for k, v in data.items() if k != 'output'} 
                           for name, data in self.activations.items()}
        with open(stats_path, 'w') as f:
            json.dump(activation_stats, f, indent=2)
        
        # 清空激活值缓存
        self.activations.clear()
    
    def log_noise_schedule(self, diffusion_model, step):
        """保存噪声调度信息"""
        if not self.enabled or not self.save_noise_schedule:
            return
            
        noise_data = {}
        
        # 提取噪声调度参数
        if hasattr(diffusion_model, 'betas'):
            noise_data['betas'] = diffusion_model.betas.cpu().numpy()
        if hasattr(diffusion_model, 'alphas_cumprod'):
            noise_data['alphas_cumprod'] = diffusion_model.alphas_cumprod.cpu().numpy()
        if hasattr(diffusion_model, 'sqrt_alphas_cumprod'):
            noise_data['sqrt_alphas_cumprod'] = diffusion_model.sqrt_alphas_cumprod.cpu().numpy()
        if hasattr(diffusion_model, 'sqrt_one_minus_alphas_cumprod'):
            noise_data['sqrt_one_minus_alphas_cumprod'] = diffusion_model.sqrt_one_minus_alphas_cumprod.cpu().numpy()
        
        noise_data['num_timesteps'] = getattr(diffusion_model, 'num_timesteps', 0)
        
        # 保存噪声调度数据
        save_path = os.path.join(self.log_dir, 'noise_schedule', f'noise_schedule_step_{step}.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(noise_data, f)
    
    def log_loss_components(self, loss_dict, step):
        """保存损失函数各组件"""
        if not self.enabled or not self.save_loss_components:
            return
            
        # 保存损失组件
        save_path = os.path.join(self.log_dir, 'loss_components', f'loss_step_{step}.json')
        with open(save_path, 'w') as f:
            json.dump(loss_dict, f, indent=2)
        
        # 更新统计信息
        for key, value in loss_dict.items():
            self.stats[key].append(value)
    
    def create_visualizations(self, step):
        """创建可视化图表"""
        if not self.enabled:
            return
            
        try:
            # 绘制损失曲线
            if self.stats:
                plt.figure(figsize=(12, 8))
                for key, values in self.stats.items():
                    if len(values) > 1:
                        plt.plot(values, label=key)
                plt.xlabel('Step')
                plt.ylabel('Loss')
                plt.title('Loss Components Over Time')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.log_dir, 'visualizations', f'loss_curves_step_{step}.png'))
                plt.close()
            
            # 如果有权重统计信息，创建权重分布图
            weight_stats_file = os.path.join(self.log_dir, 'statistics', f'weight_stats_step_{step}.json')
            if os.path.exists(weight_stats_file):
                with open(weight_stats_file, 'r') as f:
                    weight_stats = json.load(f)
                
                # 绘制权重分布
                plt.figure(figsize=(15, 10))
                layer_names = list(weight_stats.keys())
                means = [weight_stats[name]['mean'] for name in layer_names]
                stds = [weight_stats[name]['std'] for name in layer_names]
                
                plt.subplot(2, 2, 1)
                plt.bar(range(len(means)), means)
                plt.title('Weight Means by Layer')
                plt.xticks(range(len(layer_names)), [name.split('.')[-1] for name in layer_names], rotation=45)
                
                plt.subplot(2, 2, 2)
                plt.bar(range(len(stds)), stds)
                plt.title('Weight Standard Deviations by Layer')
                plt.xticks(range(len(layer_names)), [name.split('.')[-1] for name in layer_names], rotation=45)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.log_dir, 'visualizations', f'weight_distributions_step_{step}.png'))
                plt.close()
                
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def save_summary_statistics(self):
        """保存总结统计信息"""
        if not self.enabled:
            return
            
        summary = {
            'total_steps': len(self.stats.get('l_pix', [])),
            'loss_statistics': {}
        }
        
        for key, values in self.stats.items():
            if values:
                summary['loss_statistics'][key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'final_value': float(values[-1]) if values else 0
                }
        
        with open(os.path.join(self.log_dir, 'summary_statistics.json'), 'w') as f:
            json.dump(summary, f, indent=2) 