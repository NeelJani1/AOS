import argparse
import glob
import os

import sys

# === FIX: Add parent directory to system path ===
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# ===============================================


import pdb
import random
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from convertModels import savemodelDiffusers
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# Util Functions
def load_model_from_config(config, ckpt, device="cpu", verbose=False):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    # === FIX (PickleError): Set weights_only=False to load PyTorch Lightning checkpoints ===
    # Newer PyTorch versions default to weights_only=True for security,
    # but PL checkpoints contain more than just weights, causing a load failure.
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    # ====================================================================================
    
    global_step = pl_sd.get("global_step", "?") # Use .get for safety
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    
    if len(m) > 0 and verbose:
        print("Missing keys in state_dict:")
        print(m)
    if len(u) > 0 and verbose:
        print("Unexpected keys in state_dict:")
        print(u)
        
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    print(f"Model loaded with global step {global_step}")
    return model


@torch.no_grad()
def sample_model(
    model,
    sampler,
    c,
    h,
    w,
    ddim_steps,
    scale,
    ddim_eta,
    start_code=None,
    n_samples=1,
    t_start=-1,
    log_every_t=None,
    till_T=None,
    verbose=True,
):
    """Sample the model"""
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(n_samples * [""])
    log_t = 100
    if log_every_t is not None:
        log_t = log_every_t
    shape = [4, h // 8, w // 8]
    samples_ddim, inters = sampler.sample(
        S=ddim_steps,
        conditioning=c,
        batch_size=n_samples,
        shape=shape,
        verbose=False,
        x_T=start_code,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=uc,
        eta=ddim_eta,
        verbose_iter=verbose,
        t_start=t_start,
        log_every_t=log_t,
        till_T=till_T,
    )
    if log_every_t is not None:
        return samples_ddim, inters
    return samples_ddim


def load_img(path, target_size=512):
    """Load an image, resize and output -1..1"""
    image = Image.open(path).convert("RGB")

    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


##################### ESD Functions
def get_models(config_path, ckpt_path, devices):
    # === FIX: Unpack device tuple and load models to correct devices ===
    # devices is expected to be a tuple: (device_train, device_orig)
    device_train, device_orig = devices
    
    model_orig = load_model_from_config(config_path, ckpt_path, device_orig, verbose=True)
    sampler_orig = DDIMSampler(model_orig)

    model = load_model_from_config(config_path, ckpt_path, device_train)
    sampler = DDIMSampler(model)
    # =================================================================

    return model_orig, sampler_orig, model, sampler


def train_esd(
    prompt,
    train_method,
    start_guidance,
    negative_guidance,
    iterations,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    devices,
    seperator=None,
    image_size=512,
    ddim_steps=50,
):
    """
    Function to train diffusion models to erase concepts from model weights

    Parameters
    ----------
    prompt : str
        The concept to erase from diffusion model (Eg: "Van Gogh").
    train_method : str
        The parameters to train for erasure (ESD-x, ESD-u, full, selfattn).
    start_guidance : float
        Guidance to generate images for training.
    negative_guidance : float
        Guidance to erase the concepts from diffusion model.
    iterations : int
        Number of iterations to train.
    lr : float
        learning rate for fine tuning.
    config_path : str
        config path for compvis diffusion format.
    ckpt_path : str
        checkpoint path for pre-trained compvis diffusion weights.
    diffusers_config_path : str
        Config path for diffusers unet in json format.
    devices : str
        2 devices used to load the models (Eg: '0,1' will load in cuda:0 and cuda:1).
    seperator : str, optional
        If the prompt has commas can use this to seperate the prompt for individual simulataneous erasures. The default is None.
    image_size : int, optional
        Image size for generated images. The default is 512.
    ddim_steps : int, optional
        Number of diffusion time steps. The default is 50.

    Returns
    -------
    None

    """
    # PROMPT CLEANING
    word_print = prompt.replace(" ", "")
    if prompt == "allartist":
        prompt = "Kelly Mckernan, Thomas Kinkade, Ajin Demi Human, Alena Aenami, Tyler Edlin, Kilian Eng"
    if prompt == "i2p":
        prompt = "hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, nudity, bodily fluids, blood"
    if prompt == "artifact":
        prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, blurred, text, watermark, grainy"

    if seperator is not None:
        words = prompt.split(seperator)
        words = [word.strip() for word in words]
    else:
        words = [prompt]
        
    # === FIX (IndexError): Handle single-device and multi-device assignment ===
    if len(devices) > 1:
        device_orig = devices[1]
        device_train = devices[0]
        print(f"Using {device_train} for training model and {device_orig} for frozen original model.")
    else:
        device_orig = devices[0]
        device_train = devices[0]
        # This print statement is already in your user log, confirming this part works
        # print(f"Using single device {device_train} for both models.")
    # =======================================================================

    ddim_eta = 0
    # MODEL TRAINING SETUP

    model_orig, sampler_orig, model, sampler = get_models(
        config_path, ckpt_path, (device_train, device_orig) # Pass devices as a tuple
    )
    print(f"all parts of {type(model.model)}")

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == "noxattn":
            if name.startswith("out.") or "attn2" in name or "time_embed" in name:
                pass
            else:
                parameters.append(param)
        # train only self attention layers
        if train_method == "selfattn":
            if "attn1" in name:
                parameters.append(param)
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                parameters.append(param)
        # train all layers
        if train_method == "full":
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == "notime":
            if not (name.startswith("out.") or "time_embed" in name):
                parameters.append(param)
        if train_method == "xlayer":
            if "attn2" in name:
                if "output_blocks.6." in name or "output_blocks.8." in name:
                    parameters.append(param)
        if train_method == "selflayer":
            if "attn1" in name:
                if "input_blocks.4." in name or "input_blocks.7." in name:
                    parameters.append(param)
    
    if not parameters:
        print(f"Warning: No parameters selected for training method '{train_method}'. Training will not proceed.")
        return
        
    # set model to train
    model.train()
    # create a lambda function for cleaner use of sampling code (only denoising till time step t)
    quick_sample_till_t = lambda x, s, code, t: sample_model(
        model,
        sampler,
        x,
        image_size,
        image_size,
        ddim_steps,
        s,
        ddim_eta,
        start_code=code,
        till_T=t,
        verbose=False,
    )

    losses = []
    opt = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()
    history = []

    if mask_path:
        mask = torch.load(mask_path)
        name = f"compvis-esd-mask-method_{train_method}-lr_{lr}"
    else:
        name = f"compvis-esd-method_{train_method}-lr_{lr}"

    # TRAINING CODE
    pbar = tqdm(range(iterations))
    for i in pbar:
        word = random.sample(words, 1)[0]
        # get text embeddings for unconditional and conditional prompts
        emb_0 = model.get_learned_conditioning([""])
        emb_p = model.get_learned_conditioning([word])
        emb_n = model.get_learned_conditioning([f"{word}"])

        opt.zero_grad()

        # === FIX: Use explicit device variables ===
        t_enc = torch.randint(ddim_steps, (1,), device=device_train)
        # time step from 1000 to 0 (0 being good)
        og_num = round((int(t_enc) / ddim_steps) * 1000)
        og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
        
        # Ensure og_num_lim is strictly greater than og_num
        if og_num_lim <= og_num:
             og_num_lim = og_num + 1

        t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=device_train)

        start_code = torch.randn((1, 4, int(image_size / 8), int(image_size / 8))).to(
            device_train
        )

        with torch.no_grad():
            # generate an image with the concept from ESD model
            z = quick_sample_till_t(
                emb_p.to(device_train), start_guidance, start_code, int(t_enc)
            )  # emb_p seems to work better instead of emb_0
            # get conditional and unconditional scores from frozen model at time step t and image z
            e_0 = model_orig.apply_model(
                z.to(device_orig), t_enc_ddpm.to(device_orig), emb_0.to(device_orig)
            )
            e_p = model_orig.apply_model(
                z.to(device_orig), t_enc_ddpm.to(device_orig), emb_p.to(device_orig)
            )
        # breakpoint()
        # get conditional score from ESD model
        e_n = model.apply_model(
            z.to(device_train), t_enc_ddpm.to(device_train), emb_n.to(device_train)
        )
        e_0.requires_grad = False
        e_p.requires_grad = False
        # reconstruction loss for ESD objective from frozen model and conditional score of ESD model
        loss = criteria(
            e_n.to(device_train),
            e_0.to(device_train)
            - (negative_guidance * (e_p.to(device_train) - e_0.to(device_train))),
        )  # loss = criteria(e_n, e_0) works the best try 5000 epochs
        # update weights to erase the concept
        loss.backward()
        losses.append(loss.item())
        pbar.set_postfix({"loss": loss.item()})
        history.append(loss.item())

        if mask_path:
            for n, p in model.named_parameters():
                if p.grad is not None:
                    # Check if key exists in mask
                    mask_key = n.split("model.diffusion_model.")[-1]
                    if mask_key in mask:
                        p.grad *= mask[mask_key].to(device_train)
                    else:
                        # Handle cases where parameter name doesn't match mask key
                        # This might happen for non-diffusion-model parameters if any
                        pass 

        opt.step()
        # =========================================
        
        # save checkpoint and loss curve
        if (i + 1) % 500 == 0 and i + 1 != iterations and i + 1 >= 500:
            save_model(model, name, i - 1, save_compvis=True, save_diffusers=False)

        if i % 100 == 0:
            save_history(losses, name, word_print)

    model.eval()

    save_model(
        model,
        name,
        None,
        save_compvis=True,
        save_diffusers=True,
        compvis_config_file=config_path,
        diffusers_config_file=diffusers_config_path,
    )
    save_history(losses, name, word_print)


def save_model(
    model,
    name,
    num,
    compvis_config_file=None,
    diffusers_config_file=None,
    device="cpu",
    save_compvis=True,
    save_diffusers=True,
):
    # SAVE MODEL

    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}-epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)

    if save_diffusers:
        # Check if config files are provided
        if not compvis_config_file or not diffusers_config_file:
            print("Warning: Config files not provided. Skipping diffusers model save.")
            return
        
        print("Saving Model in Diffusers Format")
        try:
            savemodelDiffusers(
                name, compvis_config_file, diffusers_config_file, device=device
            )
        except Exception as e:
            print(f"Error saving in diffusers format: {e}")
            print("Please ensure 'convertModels.py' is correct and all dependencies are installed.")


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="TrainESD",
        description="Finetuning stable diffusion model to erase concepts using ESD method",
    )
    parser.add_argument(
        "--prompt",
        help="prompt corresponding to concept to erase",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--start_guidance",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=3,
    )
    parser.add_argument(
        "--negative_guidance",
        help="guidance of negative training used to train",
        type=float,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--iterations",
        help="iterations used to train",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float, # === FIX: Changed type to float to allow scientific notation like 1e-5 ===
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="/home/neel/Unlearn-Saliency-master/Unlearn-Saliency-master/SD/models/ldm/stable-diffusion-v1/.cache/huggingface/download/sd-v1-4.ckpt",
    )
    parser.add_argument(
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--devices",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="0,0",
    )
    parser.add_argument(
        "--seperator",
        help="separator if you want to train bunch of words separately",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    args = parser.parse_args()
    
    # --- Check for file paths ---
    if not os.path.exists(args.config_path):
        print(f"Error: Config file not found at {args.config_path}")
        sys.exit(1)
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint file not found at {args.ckpt_path}")
        sys.exit(1)
    if args.mask_path and not os.path.exists(args.mask_path):
        print(f"Error: Mask file not found at {args.mask_path}")
        sys.exit(1)
    # -----------------------------

    prompt = args.prompt
    train_method = args.train_method
    start_guidance = args.start_guidance
    negative_guidance = args.negative_guidance
    iterations = args.iterations
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    diffusers_config_path = args.diffusers_config_path
    devices = [f"cuda:{int(d.strip())}" for d in args.devices.split(",")]
    seperator = args.seperator
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    train_esd(
        prompt=prompt,
        train_method=train_method,
        start_guidance=start_guidance,
        negative_guidance=negative_guidance,
        iterations=iterations,
        lr=lr,
        config_path=config_path,
        ckpt_path=ckpt_path,
        mask_path=mask_path,
        diffusers_config_path=diffusers_config_path,
        devices=devices,
        seperator=seperator,
        image_size=image_size,
        ddim_steps=ddim_steps,
    )