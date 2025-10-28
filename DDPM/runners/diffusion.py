import copy
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as tvu
import tqdm
from datasets import (
    all_but_one_class_path_dataset,
    data_transform,
    get_dataset,
    get_forget_dataset,
    inverse_data_transform,
)
from functions import create_class_labels, cycle, get_optimizer
from functions.denoising import generalized_steps_conditional
from functions.losses import loss_registry_conditional
from models.diffusion import Conditional_Model
from models.ema import EMAHelper
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance optimizations
        self.mixed_precision = getattr(args, 'mixed_precision', False) and torch.cuda.is_available()
        self.grad_accumulation = getattr(args, 'grad_accumulation', 1)
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logging.info("Mixed precision training enabled")
        else:
            self.scaler = None

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def _fix_state_dict(self, state_dict):
        """Fix state dict by removing _orig_mod and module prefixes"""
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove _orig_mod. prefix if present
            if key.startswith('_orig_mod.'):
                new_key = key.replace('_orig_mod.', '')
            # Remove module. prefix if present  
            elif key.startswith('module.'):
                new_key = key.replace('module.', '')
            else:
                new_key = key
            new_state_dict[new_key] = value
        return new_state_dict

    def _load_model_checkpoint(self, model, checkpoint_path):
        """Optimized model loading with state dict fixes"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Use map_location for faster loading
        map_location = self.device if torch.cuda.is_available() else 'cpu'
        states = torch.load(checkpoint_path, map_location=map_location)
        
        if isinstance(states, (list, tuple)) and len(states) > 0:
            state_dict = states[0]
        else:
            state_dict = states
            
        # Fix state dict
        state_dict = self._fix_state_dict(state_dict)
        
        # Load with error handling
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logging.warning(f"Missing keys in state dict: {len(missing_keys)}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys in state dict: {len(unexpected_keys)}")
            
        return states

    def save_fim(self):
        args, config = self.args, self.config
        bs = min(torch.cuda.device_count(), 4)  # Limit to 4 GPUs for stability
        
        fim_dataset = ImageFolder(
            os.path.join(args.ckpt_folder, "class_samples"),
            transform=transforms.ToTensor(),
        )
        fim_loader = DataLoader(
            fim_dataset,
            batch_size=bs,
            num_workers=min(config.data.num_workers, 8),  # Limit workers
            shuffle=True,
            pin_memory=True  # Faster data transfer
        )

        logging.info("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(self.config)
        
        checkpoint_path = os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth")
        states = self._load_model_checkpoint(model, checkpoint_path)
        
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.eval()

        # calculate FIM with memory optimizations
        fisher_dict = {}
        fisher_dict_temp_list = [{} for _ in range(bs)]

        # Only track parameters that require gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = param.data.clone().zero_()
                for i in range(bs):
                    fisher_dict_temp_list[i][name] = param.data.clone().zero_()

        # calculate Fisher information diagonals with tqdm and memory optimizations
        for step, data in enumerate(
            tqdm.tqdm(fim_loader, desc="Calculating Fisher information matrix")
        ):
            x, c = data
            x, c = x.to(self.device, non_blocking=True), c.to(self.device, non_blocking=True)

            b = self.betas
            ts = torch.chunk(torch.arange(0, self.num_timesteps), args.n_chunks)

            for _t in ts:
                for i in range(len(_t)):
                    e = torch.randn_like(x)
                    t = torch.tensor([_t[i]]).expand(bs).to(self.device)

                    # Use mixed precision if available
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            if i == 0:
                                loss = loss_registry_conditional[config.model.type](
                                    model, x, t, c, e, b, keepdim=True
                                )
                            else:
                                loss += loss_registry_conditional[config.model.type](
                                    model, x, t, c, e, b, keepdim=True
                                )
                    else:
                        if i == 0:
                            loss = loss_registry_conditional[config.model.type](
                                model, x, t, c, e, b, keepdim=True
                            )
                        else:
                            loss += loss_registry_conditional[config.model.type](
                                model, x, t, c, e, b, keepdim=True
                            )

                # store first-order gradients for each sample
                for i in range(bs):
                    model.zero_grad()
                    if i != len(loss) - 1:
                        loss[i].backward(retain_graph=True)
                    else:
                        loss[i].backward()
                    for name, param in model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            fisher_dict_temp_list[i][name] += param.grad.data
                del loss

            # Aggregate gradients
            for name in fisher_dict:
                for i in range(bs):
                    fisher_dict[name].data += (
                        fisher_dict_temp_list[i][name].data ** 2
                    ) / len(fim_loader.dataset)
                    fisher_dict_temp_list[i][name].zero_()

            # Save periodically to avoid memory issues
            if (step + 1) % max(1, config.training.save_freq // 10) == 0:
                with open(os.path.join(args.ckpt_folder, "fisher_dict_temp.pkl"), "wb") as f:
                    pickle.dump(fisher_dict, f)

        # Final save
        with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "wb") as f:
            pickle.dump(fisher_dict, f)

    def train(self):
        args, config = self.args, self.config
        D_train_loader = get_dataset(args, config)
        D_train_iter = cycle(D_train_loader)
        
        model = Conditional_Model(config)
        optimizer = get_optimizer(self.config, model.parameters())
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
        
        model.train()
        
        start = time.time()
        accumulation_steps = 0
        
        for step in range(0, self.config.training.n_iters):
            model.train()
            x, c = next(D_train_iter)
            n = x.size(0)
            x = x.to(self.device, non_blocking=True)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(
                low=0, high=self.num_timesteps, size=(n // 2 + 1,)
            ).to(self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            
            # Mixed precision forward
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = loss_registry_conditional[config.model.type](model, x, t, c, e, b)
                loss = loss / self.grad_accumulation
                self.scaler.scale(loss).backward()
            else:
                loss = loss_registry_conditional[config.model.type](model, x, t, c, e, b)
                loss = loss / self.grad_accumulation
                loss.backward()
            
            accumulation_steps += 1
            
            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                actual_loss = loss.item() * self.grad_accumulation
                logging.info(
                    f"step: {step}, loss: {actual_loss:.6f}, time: {end-start:.2f}s"
                )
                start = time.time()
            
            # Gradient accumulation and update
            if accumulation_steps % self.grad_accumulation == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                
                optimizer.zero_grad()
                accumulation_steps = 0

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0 or (step + 1) == self.config.training.n_iters:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )
                
                # Only sample every few checkpoints to save time
                if (step + 1) % (self.config.training.snapshot_freq * 2) == 0:
                    test_model = ema_helper.ema_copy(model) if self.config.model.ema else copy.deepcopy(model)
                    test_model.eval()
                    self.sample_visualization(test_model, step, args.cond_scale)
                    del test_model
                    torch.cuda.empty_cache()  # Clear cache after sampling

    def train_forget(self):
        args, config = self.args, self.config
        logging.info(
            f"Training diffusion forget with contrastive and EWC. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        )
        D_train_loader = all_but_one_class_path_dataset(
            config,
            os.path.join(args.ckpt_folder, "class_samples"),
            args.label_to_forget,
        )
        D_train_iter = cycle(D_train_loader)

        logging.info("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        checkpoint_path = os.path.join(args.ckpt_folder, "ckpts/ckpt.pth")
        states = self._load_model_checkpoint(model, checkpoint_path)
        
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
        else:
            ema_helper = None

        with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "rb") as f:
            fisher_dict = pickle.load(f)

        params_mle_dict = {}
        for name, param in model.named_parameters():
            params_mle_dict[name] = param.data.clone()

        label_choices = list(range(config.data.n_classes))
        label_choices.remove(args.label_to_forget)

        accumulation_steps = 0
        
        for step in range(0, config.training.n_iters):
            model.train()
            x_remember, c_remember = next(D_train_iter)
            x_remember, c_remember = x_remember.to(self.device, non_blocking=True), c_remember.to(
                self.device
            )
            x_remember = data_transform(config, x_remember)

            n = x_remember.size(0)
            channels = config.data.channels
            img_size = config.data.image_size
            c_forget = (torch.ones(n, dtype=int) * args.label_to_forget).to(self.device)
            x_forget = (
                torch.rand((n, channels, img_size, img_size), device=self.device) - 0.5
            ) * 2.0
            e_remember = torch.randn_like(x_remember)
            e_forget = torch.randn_like(x_forget)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = loss_registry_conditional[config.model.type](
                        model, x_forget, t, c_forget, e_forget, b, cond_drop_prob=0.0
                    ) + config.training.gamma * loss_registry_conditional[config.model.type](
                        model, x_remember, t, c_remember, e_remember, b, cond_drop_prob=0.0
                    )
                    forgetting_loss = loss.item()

                    ewc_loss = 0.0
                    for name, param in model.named_parameters():
                        _loss = (
                            fisher_dict[name].to(self.device)
                            * (param - params_mle_dict[name].to(self.device)) ** 2
                        )
                        loss += config.training.lmbda * _loss.sum()
                        ewc_loss += config.training.lmbda * _loss.sum()
                
                loss = loss / self.grad_accumulation
                self.scaler.scale(loss).backward()
            else:
                loss = loss_registry_conditional[config.model.type](
                    model, x_forget, t, c_forget, e_forget, b, cond_drop_prob=0.0
                ) + config.training.gamma * loss_registry_conditional[config.model.type](
                    model, x_remember, t, c_remember, e_remember, b, cond_drop_prob=0.0
                )
                forgetting_loss = loss.item()

                ewc_loss = 0.0
                for name, param in model.named_parameters():
                    _loss = (
                        fisher_dict[name].to(self.device)
                        * (param - params_mle_dict[name].to(self.device)) ** 2
                    )
                    loss += config.training.lmbda * _loss.sum()
                    ewc_loss += config.training.lmbda * _loss.sum()
                
                loss = loss / self.grad_accumulation
                loss.backward()

            accumulation_steps += 1

            if (step + 1) % config.training.log_freq == 0:
                logging.info(
                    f"step: {step}, loss: {loss.item() * self.grad_accumulation:.6f}, forgetting loss: {forgetting_loss:.6f}, ewc loss: {ewc_loss:.6f}"
                )

            # Gradient accumulation update
            if accumulation_steps % self.grad_accumulation == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                
                optimizer.zero_grad()
                accumulation_steps = 0

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model
                torch.cuda.empty_cache()

    def retrain(self):
        args, config = self.args, self.config

        D_remain_loader, _ = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)

        model = Conditional_Model(config)

        optimizer = get_optimizer(self.config, model.parameters())
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        model.train()

        start = time.time()
        accumulation_steps = 0
        
        for step in range(0, self.config.training.n_iters):
            model.train()
            x, c = next(D_remain_iter)

            n = x.size(0)
            x = x.to(self.device, non_blocking=True)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss = loss_registry_conditional[config.model.type](model, x, t, c, e, b)
                loss = loss / self.grad_accumulation
                self.scaler.scale(loss).backward()
            else:
                loss = loss_registry_conditional[config.model.type](model, x, t, c, e, b)
                loss = loss / self.grad_accumulation
                loss.backward()
            
            accumulation_steps += 1

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                actual_loss = loss.item() * self.grad_accumulation
                logging.info(f"step: {step}, loss: {actual_loss:.6f}, time: {end-start:.2f}s")
                start = time.time()

            # Gradient accumulation update
            if accumulation_steps % self.grad_accumulation == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                    optimizer.step()
                
                optimizer.zero_grad()
                accumulation_steps = 0

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model
                torch.cuda.empty_cache()

    def saliency_unlearn(self):
        args, config = self.args, self.config

        D_remain_loader, D_forget_loader = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)
        D_forget_iter = cycle(D_forget_loader)

        if args.mask_path:
            mask = torch.load(args.mask_path)
        else:
            mask = None

        logging.info("Loading checkpoints {}".format(args.ckpt_folder))

        model = Conditional_Model(config)
        checkpoint_path = os.path.join(args.ckpt_folder, "ckpts/ckpt.pth")
        states = self._load_model_checkpoint(model, checkpoint_path)
        
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        optimizer = get_optimizer(config, model.parameters())
        criteria = torch.nn.MSELoss()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
        else:
            ema_helper = None

        model.train()
        start = time.time()
        accumulation_steps = 0
        
        for step in range(0, self.config.training.n_iters):
            model.train()

            # remain stage
            remain_x, remain_c = next(D_remain_iter)
            n = remain_x.size(0)
            remain_x = remain_x.to(self.device, non_blocking=True)
            remain_x = data_transform(self.config, remain_x)
            e = torch.randn_like(remain_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    remain_loss = loss_registry_conditional[config.model.type](
                        model, remain_x, t, remain_c, e, b
                    )

                    # forget stage
                    forget_x, forget_c = next(D_forget_iter)

                    n = forget_x.size(0)
                    forget_x = forget_x.to(self.device, non_blocking=True)
                    forget_x = data_transform(self.config, forget_x)
                    e = torch.randn_like(forget_x)

                    t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                        self.device
                    )
                    t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                    if args.method == "ga":
                        forget_loss = -loss_registry_conditional[config.model.type](
                            model, forget_x, t, forget_c, e, b
                        )
                    else:
                        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                        forget_x = forget_x * a.sqrt() + e * (1.0 - a).sqrt()

                        output = model(forget_x, t.float(), forget_c, mode="train")

                        if args.method == "rl":
                            pseudo_c = torch.full(
                                forget_c.shape,
                                (args.label_to_forget + 1) % 10,
                                device=forget_c.device,
                            )
                            pseudo = model(forget_x, t.float(), pseudo_c, mode="train").detach()
                            forget_loss = criteria(pseudo, output)

                    loss = forget_loss + args.alpha * remain_loss
                
                loss = loss / self.grad_accumulation
                self.scaler.scale(loss).backward()
            else:
                remain_loss = loss_registry_conditional[config.model.type](
                    model, remain_x, t, remain_c, e, b
                )

                # forget stage
                forget_x, forget_c = next(D_forget_iter)

                n = forget_x.size(0)
                forget_x = forget_x.to(self.device, non_blocking=True)
                forget_x = data_transform(self.config, forget_x)
                e = torch.randn_like(forget_x)

                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                    self.device
                )
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                if args.method == "ga":
                    forget_loss = -loss_registry_conditional[config.model.type](
                        model, forget_x, t, forget_c, e, b
                    )
                else:
                    a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                    forget_x = forget_x * a.sqrt() + e * (1.0 - a).sqrt()

                    output = model(forget_x, t.float(), forget_c, mode="train")

                    if args.method == "rl":
                        pseudo_c = torch.full(
                            forget_c.shape,
                            (args.label_to_forget + 1) % 10,
                            device=forget_c.device,
                        )
                        pseudo = model(forget_x, t.float(), pseudo_c, mode="train").detach()
                        forget_loss = criteria(pseudo, output)

                loss = forget_loss + args.alpha * remain_loss
                loss = loss / self.grad_accumulation
                loss.backward()

            accumulation_steps += 1

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                actual_loss = loss.item() * self.grad_accumulation
                logging.info(f"step: {step}, loss: {actual_loss:.6f}, time: {end-start:.2f}s")
                start = time.time()

            # Gradient accumulation update
            if accumulation_steps % self.grad_accumulation == 0:
                if self.mixed_precision:
                    self.scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass
                else:
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), config.optim.grad_clip
                        )
                    except Exception:
                        pass

                if mask:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask[name].to(param.grad.device)
                
                if self.mixed_precision:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                accumulation_steps = 0

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model
                torch.cuda.empty_cache()

    def load_ema_model(self):
        model = Conditional_Model(self.config)
        checkpoint_path = os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth")
        states = self._load_model_checkpoint(model, checkpoint_path)
        
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            test_model = ema_helper.ema_copy(model)
        else:
            ema_helper = None

        model.eval()
        return model

    def sample(self):
            model = Conditional_Model(self.config)
            checkpoint_path = os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth")
            states = self._load_model_checkpoint(model, checkpoint_path)

            model = model.to(self.device)
            # Don't use DataParallel here to avoid module prefix issues
            # model = torch.nn.DataParallel(model)

            if self.config.model.ema and len(states) > 3:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[3])
                test_model = ema_helper.ema_copy(model)
            else:
                test_model = model

            model.eval()
            test_model.eval()

            if self.args.mode == "sample_fid":
                self.sample_fid(test_model, self.args.cond_scale)
            elif self.args.mode == "sample_classes":
                self.sample_classes(test_model, self.args.cond_scale)
            elif self.args.mode == "visualization":
                self.sample_visualization(
                    test_model, str(self.args.cond_scale), self.args.cond_scale
                )
                
    def sample_classes(self, model, cond_scale):
        """
        Samples each class from the model. Can be used to calculate FIM, for generative replay
        or for classifier evaluation. Stores samples in "./class_samples/<class_label>".
        """
        config = self.config
        args = self.args
        sample_dir = os.path.join(args.ckpt_folder, "class_samples")
        os.makedirs(sample_dir, exist_ok=True)
        img_id = 0
        classes, _ = create_class_labels(
            args.classes_to_generate, n_classes=config.data.n_classes
        )
        n_samples_per_class = args.n_samples_per_class

        for i in classes:
            os.makedirs(os.path.join(sample_dir, str(i)), exist_ok=True)
            if n_samples_per_class % config.sampling.batch_size == 0:
                n_rounds = n_samples_per_class // config.sampling.batch_size
            else:
                n_rounds = n_samples_per_class // config.sampling.batch_size + 1
            n_left = n_samples_per_class  # tracker on how many samples left to generate

            with torch.no_grad():
                for j in tqdm.tqdm(
                    range(n_rounds),
                    desc=f"Generating image samples for class {i} to use as dataset",
                ):
                    if n_left >= config.sampling.batch_size:
                        n = config.sampling.batch_size
                    else:
                        n = n_left

                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                    c = torch.ones(x.size(0), device=self.device, dtype=int) * int(i)
                    
                    # Use mixed precision for faster sampling
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            x = self.sample_image(x, model, c, cond_scale)
                    else:
                        x = self.sample_image(x, model, c, cond_scale)
                        
                    x = inverse_data_transform(config, x)

                    for k in range(n):
                        tvu.save_image(
                            x[k],
                            os.path.join(sample_dir, str(c[k].item()), f"{img_id}.png"),
                            normalize=True,
                        )
                        img_id += 1

                    n_left -= n
                    
                    # Clear cache periodically to avoid OOM
                    if j % 10 == 0:
                        torch.cuda.empty_cache()

    def sample_one_class(self, model, cond_scale, class_label):
        """
        Samples one class only for classifier evaluation.
        """
        config = self.config
        args = self.args
        sample_dir = os.path.join(args.ckpt_folder, "class_" + str(class_label))
        os.makedirs(sample_dir, exist_ok=True)
        img_id = 0
        total_n_samples = 500

        if total_n_samples % config.sampling.batch_size == 0:
            n_rounds = total_n_samples // config.sampling.batch_size
        else:
            n_rounds = total_n_samples // config.sampling.batch_size + 1
        n_left = total_n_samples  # tracker on how many samples left to generate

        with torch.no_grad():
            for j in tqdm.tqdm(
                range(n_rounds),
                desc=f"Generating image samples for class {class_label}",
            ):
                if n_left >= config.sampling.batch_size:
                    n = config.sampling.batch_size
                else:
                    n = n_left

                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                c = torch.ones(x.size(0), device=self.device, dtype=int) * class_label
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        x = self.sample_image(x, model, c, cond_scale)
                else:
                    x = self.sample_image(x, model, c, cond_scale)
                    
                x = inverse_data_transform(config, x)

                for k in range(n):
                    tvu.save_image(
                        x[k], os.path.join(sample_dir, f"{img_id}.png"), normalize=True
                    )
                    img_id += 1

                n_left -= n
                
                # Clear cache periodically
                if j % 10 == 0:
                    torch.cuda.empty_cache()

    def sample_fid(self, model, cond_scale):
        config = self.config
        args = self.args
        img_id = 0

        classes, excluded_classes = create_class_labels(
            args.classes_to_generate, n_classes=config.data.n_classes
        )
        n_samples_per_class = args.n_samples_per_class

        sample_dir = f"fid_samples_guidance_{args.cond_scale}"
        if excluded_classes:
            excluded_classes_str = "_".join(str(i) for i in excluded_classes)
            sample_dir = f"{sample_dir}_excluded_class_{excluded_classes_str}"
        sample_dir = os.path.join(args.ckpt_folder, sample_dir)
        os.makedirs(sample_dir, exist_ok=True)

        for i in classes:
            if n_samples_per_class % config.sampling.batch_size == 0:
                n_rounds = n_samples_per_class // config.sampling.batch_size
            else:
                n_rounds = n_samples_per_class // config.sampling.batch_size + 1
            n_left = n_samples_per_class  # tracker on how many samples left to generate

            with torch.no_grad():
                for j in tqdm.tqdm(
                    range(n_rounds),
                    desc=f"Generating image samples for class {i} for FID",
                ):
                    if n_left >= config.sampling.batch_size:
                        n = config.sampling.batch_size
                    else:
                        n = n_left

                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                    c = torch.ones(x.size(0), device=self.device, dtype=int) * int(i)
                    
                    if self.mixed_precision:
                        with torch.cuda.amp.autocast():
                            x = self.sample_image(x, model, c, cond_scale)
                    else:
                        x = self.sample_image(x, model, c, cond_scale)
                        
                    x = inverse_data_transform(config, x)

                    for k in range(n):
                        tvu.save_image(
                            x[k],
                            os.path.join(sample_dir, f"{img_id}.png"),
                            normalize=True,
                        )
                        img_id += 1

                    n_left -= n
                    
                    # Clear cache periodically
                    if j % 10 == 0:
                        torch.cuda.empty_cache()

    def sample_image(self, x, model, c, cond_scale, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps_conditional

            xs = generalized_steps_conditional(
                x, c, seq, model, self.betas, cond_scale, eta=self.args.eta
            )
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps_conditional

            x = ddpm_steps_conditional(x, c, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def sample_visualization(self, model, name, cond_scale):
        config = self.config
        total_n_samples = config.training.visualization_samples
        assert total_n_samples % config.data.n_classes == 0
        n_rounds = (
            total_n_samples // config.sampling.batch_size
            if config.sampling.batch_size < total_n_samples
            else 1
        )

        # esd
        c = torch.repeat_interleave(
            torch.arange(config.data.n_classes),
            total_n_samples // config.data.n_classes,
        ).to(self.device)

        c_chunks = torch.chunk(c, n_rounds, dim=0)

        with torch.no_grad():
            all_imgs = []
            for i in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for visualization."
            ):
                c = c_chunks[i]
                n = c.size(0)
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        x = self.sample_image(x, model, c, cond_scale)
                else:
                    x = self.sample_image(x, model, c, cond_scale)
                    
                x = inverse_data_transform(config, x)

                all_imgs.append(x)

            all_imgs = torch.cat(all_imgs)
            grid = tvu.make_grid(
                all_imgs,
                nrow=total_n_samples // config.data.n_classes,
                normalize=True,
                padding=0,
            )

            try:
                tvu.save_image(
                    grid, os.path.join(self.config.log_dir, f"sample-{name}.png")
                )  # if called during training of base model
            except AttributeError:
                tvu.save_image(
                    grid, os.path.join(self.args.ckpt_folder, f"sample-{name}.png")
                )  # if called from sample.py

    def generate_mask(self):
        args, config = self.args, self.config
        logging.info(
            f"Generating mask of diffusion to achieve gradient sparsity. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        )

        _, D_forget_loader = get_forget_dataset(args, config, args.label_to_forget)
        
        logging.info("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        checkpoint_path = os.path.join(args.ckpt_folder, "ckpts/ckpt.pth")
        states = self._load_model_checkpoint(model, checkpoint_path)
        
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(config, model.parameters())

        gradients = {}
        for name, param in model.named_parameters():
            gradients[name] = 0

        model.eval()

        for x, forget_c in D_forget_loader:
            n = x.size(0)
            x = x.to(self.device, non_blocking=True)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            # loss 1
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x = x * a.sqrt() + e * (1.0 - a).sqrt()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = model(
                        x, t.float(), forget_c, cond_scale=args.cond_scale, mode="test"
                    )
                    loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
            else:
                output = model(
                    x, t.float(), forget_c, cond_scale=args.cond_scale, mode="test"
                )
                loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradient = param.grad.data.cpu()
                        gradients[name] += gradient

        with torch.no_grad():

            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])

            mask_path = os.path.join('results/cifar10/mask', str(args.label_to_forget))
            os.makedirs(mask_path, exist_ok=True)

            threshold_list = [0.5]
            for i in threshold_list:
                logging.info(f"Generating mask with threshold: {i}")
                sorted_dict_positions = {}
                hard_dict = {}

                # Concatenate all tensors into a single tensor
                all_elements = - torch.cat(
                    [tensor.flatten() for tensor in gradients.values()]
                )

                # Calculate the threshold index for the top 10% elements
                threshold_index = int(len(all_elements) * i)

                # Calculate positions of all elements
                positions = torch.argsort(all_elements)
                ranks = torch.argsort(positions)

                start_index = 0
                for key, tensor in gradients.items():
                    num_elements = tensor.numel()
                    tensor_ranks = ranks[start_index : start_index + num_elements]

                    sorted_positions = tensor_ranks.reshape(tensor.shape)
                    sorted_dict_positions[key] = sorted_positions

                    # Set the corresponding elements to 1
                    threshold_tensor = torch.zeros_like(tensor_ranks)
                    threshold_tensor[tensor_ranks < threshold_index] = 1
                    threshold_tensor = threshold_tensor.reshape(tensor.shape)
                    hard_dict[key] = threshold_tensor
                    start_index += num_elements

                torch.save(hard_dict, os.path.join(mask_path, f'with_{str(i)}.pt'))