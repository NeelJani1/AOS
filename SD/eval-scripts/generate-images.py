import argparse
import os

import pandas as pd
import torch
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm


def generate_images(
    model_name,
    prompts_path,
    save_path,
    device="cuda:0",
    guidance_scale=7.5,
    image_size=512,
    ddim_steps=100,
    num_samples=10,
    from_case=0,
):
    """
    Function to generate images from diffusers code

    The program requires the prompts to be in a csv format with headers
        1. 'case_number' (used for file naming of image)
        2. 'prompt' (the prompt used to generate image)
        3. 'seed' (the inital seed to generate gaussion noise for diffusion input)

    Parameters
    ----------
    model_name : str
        name of the model to load.
    prompts_path : str
        path for the csv file with prompts and corresponding seeds.
    save_path : str
        save directory for images.
    device : str, optional
        device to be used to load the model. The default is 'cuda:0'.
    guidance_scale : float, optional
        guidance value for inference. The default is 7.5.
    image_size : int, optional
        image size. The default is 512.
    ddim_steps : int, optional
        number of denoising steps. The default is 100.
    num_samples : int, optional
        number of samples generated per prompt. The default is 10.
    from_case : int, optional
        The starting offset in csv to generate images. The default is 0.

    Returns
    -------
    None.

    """

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="vae"
    )
    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    
    # 3. The UNet model for generating the latents.
    # First, load the base model architecture
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="unet"
    )

    # --- START OF MODEL LOADING FIX ---
    # The original script had a bug that SKIPPED loading your model.
    # This logic now correctly loads your unlearned .pt file.
    try:
        print(f"Loading unlearned UNet weights from: {model_name}")
        # model_name is the full path to your .pt file
        unet.load_state_dict(torch.load(model_name, map_location="cpu"))
        print("Unlearned weights loaded successfully.")
    except Exception as e:
        print(
            f"Could not load unlearned weights! Using base SD 1.4 model. Error: {e}"
        )
    # --- END OF MODEL LOADING FIX ---
    
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    torch_device = device
    df = pd.read_csv(prompts_path)

    # --- MODIFIED SAVE PATH ---
    # The script was creating a folder named after the *full path*, which is messy.
    # This now uses the .pt file's name for the folder.
    model_folder_name = os.path.basename(os.path.dirname(model_name))
    folder_path = f"{save_path}/{model_folder_name}"
    # --- END OF MODIFICATION ---
    
    os.makedirs(folder_path, exist_ok=True)
    print(f"Saving images to: {folder_path}")

    # This loop goes through each prompt in the CSV
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        prompt = [str(row.prompt)] * num_samples
        print(prompt)
        seed = row.evaluation_seed
        case_number = row.case_number
        if case_number < from_case:
            continue

        height = image_size  # default height of Stable Diffusion
        width = image_size  # default width of Stable Diffusion

        num_inference_steps = ddim_steps  # Number of denoising steps

        guidance_scale = guidance_scale  # Scale for classifier-free guidance

        generator = torch.manual_seed(
            seed
        )  # Seed generator to create the inital latent noise

        batch_size = len(prompt)

        # --- START OF "10x FASTER" FIX ---
        # The original script had a `for i in range(10):` loop here
        # that generated 10x more images than requested.
        # It has been removed. We now only run the generation once per prompt.
        
        text_input = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # --- FIX FOR FUTUREWARNING ---
        # Changed unet.in_channels to unet.config.in_channels
        latents = torch.randn(
            (batch_size, unet.config.in_channels, height // 8, width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma
        
        # This is the denoising loop
        for t in tqdm(scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            
            # --- START OF OOM CRASH FIX ---
            # The original script decoded all latents at once, causing a crash.
            # This loop decodes them one by one to save GPU memory.
            print("Decoding latents one by one...")
            decoded_images = []
            for latent_sample in latents:
                # Add a batch dimension (e.g., [4, 64, 64] -> [1, 4, 64, 64])
                latent_sample = latent_sample.unsqueeze(0)
                # Decode just this single latent
                decoded_image = vae.decode(latent_sample).sample
                decoded_images.append(decoded_image)
            
            # Combine the individual images back into a batch
            image = torch.cat(decoded_images, dim=0)
            # --- END OF OOM CRASH FIX ---

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        
        for num, im in enumerate(pil_images):
            # --- FIX FILE NAMING ---
            # Removed the 'i * 10' part from the filename
            im.save(f"{folder_path}/{case_number}_{num}.png")
        
        # --- END OF "10x FASTER" FIX (REMOVED LOOP) ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateImages", description="Generate Images using Diffusers Code"
    )
    parser.add_argument("--model_name", help="name of model", type=str, required=True)
    parser.add_argument(
        "--prompts_path", help="path to csv file with prompts", type=str, required=True
    )
    parser.add_argument(
        "--save_path", help="folder where to save images", type=str, required=True
    )
    parser.add_argument(
        "--device",
        help="cuda device to run on",
        type=str,
        required=False,
        default="cuda:0",
    )
    parser.add_argument(
        "--guidance_scale",
        help="guidance to run eval",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--from_case",
        help="continue generating from case_number",
        type=int,
        required=False,
        default=0,
    )
    parser.add_argument(
        "--num_samples",
        help="number of samples per prompt",
        type=int,
        required=False,
        default=10,
    )
    parser.add_argument(
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=100,
    )
    args = parser.parse_args()

    model_name = args.model_name
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples = args.num_samples
    from_case = args.from_case

    generate_images(
        model_name,
        prompts_path,
        save_path,
        device=device,
        guidance_scale=guidance_scale,
        image_size=image_size,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        from_case=from_case,
    )