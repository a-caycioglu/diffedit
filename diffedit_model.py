import torch
from diffusers import DDIMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms

def load_models():
    print("Models are loading...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=torch.float16).to(device)
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float16).to(device)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float16).to(device)
    scheduler = DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
    print("Models are loaded successfully!")
    return unet, vae, tokenizer, text_encoder, scheduler

class DiffEdit:
    def __init__(self, image, unet, vae, tokenizer, text_encoder, scheduler, query_prompt, ref_prompt=''):
        self.image = image  # Size of 512x512
        self.unet = unet
        self.vae = vae
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.ref_prompt = ref_prompt
        self.query_prompt = query_prompt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def pil_to_latent(self, image):
        trns = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
        with torch.no_grad():
            latent = self.vae.encode((trns(image).unsqueeze(0).to(self.device) * 2 - 1).half())
        return self.vae.config.scaling_factor * latent.latent_dist.sample()

    def latents_to_pil(self, latents):
        latents = (1 / self.vae.config.scaling_factor) * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def text_enc(self, prompts, maxlen=None):
        if maxlen is None:
            maxlen = self.tokenizer.model_max_length
        inp = self.tokenizer(prompts, padding="max_length", max_length=maxlen, truncation=True, return_tensors="pt")
        return self.text_encoder(inp.input_ids.to(self.device))[0].half()

    def mk_samples_img_img(self, prompts, g=7.5, seed=200, num_inference_steps=50, strength=0.5):
        torch.manual_seed(seed)
        bs = len(prompts)

        text = self.text_enc(prompts)
        uncond = self.text_enc([""] * bs)
        emb = torch.cat([uncond, text])

        img_latents = self.pil_to_latent(self.image)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps[-int(num_inference_steps * strength):]

        noise = torch.randn_like(img_latents)
        latents = self.scheduler.add_noise(img_latents, noise, timesteps[0])

        for ts in tqdm(timesteps):
            inp = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)
            with torch.no_grad():
                u, t = self.unet(inp, ts, encoder_hidden_states=emb).sample.chunk(2)
            pred = u + g * (t - u)
            latents = self.scheduler.step(pred, ts, latents).prev_sample

        return latents, self.latents_to_pil(latents)[0]

    def mask(self): # Create mask by averaging 10 outputs for each prompt.
        batch_size = 10
        ref_tensor = torch.zeros(batch_size, 4, 64, 64, device=self.device)
        query_tensor = torch.zeros(batch_size, 4, 64, 64, device=self.device)

        for i in range(batch_size):
            torch.manual_seed(i**2 + 100)
            ref_tensor[i] = self.mk_samples_img_img(prompts=[self.ref_prompt], seed=i**2+100)[0]

            torch.manual_seed(i**2 + 100)
            query_tensor[i] = self.mk_samples_img_img(prompts=[self.query_prompt], seed=i**2+100)[0]

        difference = query_tensor - ref_tensor
        distance = torch.norm(difference, dim=1, keepdim=True)
        distance_mean = distance.mean(dim=0)
        distance_mean_scaled = (distance_mean - distance_mean.min()) / (distance_mean.max() - distance_mean.min())
        return distance_mean_scaled
    
    def mask_binarized(self, dms=None, threshold=0.35): # To binarize mask, we use a threshold. Values bigger than threshold will be 1, others will be 0.
        dms = self.mask() if dms is None else dms
        masked_area = (dms >= threshold).squeeze().int()
        return masked_area

    def visualize_mask(self, mask):
        plt.figure(figsize=(5, 5))
        plt.imshow(mask.cpu().numpy(), cmap='gray')
        plt.axis('off')
        return plt

    def improved_masked_diffusion(self, prompts, mask, g=7.5, seed=10, num_inference_steps=50, strength=0.5):
        torch.manual_seed(seed)

        text = self.text_enc(prompts[0])
        uncond = self.text_enc(prompts[1])
        emb = torch.cat([uncond, text])

        img_latents = self.pil_to_latent(self.image)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps[-int(num_inference_steps * strength):]

        noise = torch.randn_like(img_latents)
        latents = self.scheduler.add_noise(img_latents, noise, timesteps[0])

        for ts in tqdm(timesteps):
            original_noisy = self.scheduler.add_noise(img_latents, noise, ts)

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, ts)
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, ts, encoder_hidden_states=emb).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + g * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, ts, latents).prev_sample

            latents = mask * latents + (1 - mask) * original_noisy

        return self.latents_to_pil(latents)[0]