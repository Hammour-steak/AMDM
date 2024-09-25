from modules import StableDiffusionMIGCPipeline, load_migc, AttentionStore, MIGCProcessor, migc_seed_everything, InteractDiffusionPipeline, IPAdapter
from diffusers import EulerDiscreteScheduler, DDIMScheduler, AutoencoderKL
import torch 
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
import warnings
from PIL import Image
from  modules import get_generator
import random

def compute_mu(
    latents, 
    timestep, 
    pipeline,
    encoder_hidden_states=None, 
    cross_attention_kwargs=None,
    guidance_scale=7.5,
    do_classifier_free_guidance=True,
    noise_pred=None
):
    scheduler = pipeline.scheduler
    alpha_t = scheduler.alphas[timestep] 
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    
    if noise_pred is None:
        # compute epsilon_theta(x_t, t, y)
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        noise_pred = pipeline.unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
    # compute mu
    coeff1 = 1. / (alpha_t ** 0.5)
    coeff2 = (1. - alpha_t) / ((1. - alpha_prod_t) ** 0.5 * alpha_t ** 0.5)
    mu = coeff1 * latents - coeff2 * noise_pred
    
    return mu  

def compute_sigma(
    timestep,
    pipeline,
    eta=0.0
):
    scheduler = pipeline.scheduler
    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps     # t - 1
    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_timestep]
    sigma_t = eta * ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) ** 0.5 * (1. - alpha_prod_t / alpha_prod_t_prev) ** 0.5
    
    return sigma_t
    

def optimize(
    pipeline,
    latents,
    prev_latents,
    mixed_latents,
    timestep,
    encoder_hidden_states=None, 
    cross_attention_kwargs=None,
    guidance_scale=7.5,
    do_classifier_free_guidance=True,
    p=1,
    name="interact",
    prev_noise_pred=None
):
    latents_prev = prev_latents
    prev_timestep = timestep + pipeline.scheduler.config.num_train_timesteps // pipeline.scheduler.num_inference_steps     # t + 1
    mu = compute_mu(latents_prev, prev_timestep, pipeline, encoder_hidden_states, cross_attention_kwargs, guidance_scale, do_classifier_free_guidance, prev_noise_pred)
    optimized_latents = mixed_latents - p * (mixed_latents - mu) / torch.norm(mixed_latents - mu)
    
    return optimized_latents
    
class InteractIP:
    def __init__(
        self, 
        interact_ckpt_path='checkpoints/interactdiffusion',
        ip_base_model_ckpt_path='checkpoints/runwayml/stable-diffusion-v1-5',
        ip_adapter_ckpt_path='checkpoints/IPAdapter/models/ip-adapter_sd15.safetensors',
        vae_ckpt_path='checkpoints/stabilityai/sd-vae-ft-mse',
        image_encoder_ckpt_path='checkpoints/IPAdapter/models/image_encoder',
        device="cuda:6"
    ):
        self.device = device
        
        # InteractDiffusion setup
        self.interact_pipe = InteractDiffusionPipeline.from_pretrained(
            interact_ckpt_path,
            torch_dtype=torch.float16
        )
        self.interact_pipe = self.interact_pipe.to(device)
        
        # IP-Adapter setup
        scheduler = self.interact_pipe.scheduler
        self.ip_vae = AutoencoderKL.from_pretrained(vae_ckpt_path).to(dtype=torch.float16)
        self.ip_pipe = StableDiffusionPipeline.from_pretrained(
            ip_base_model_ckpt_path,
            torch_dtype=torch.float16,
            scheduler=scheduler,
            vae=self.ip_vae,
            feature_extractor=None,
            safety_checker=None
        )
        self.ip_adapter = IPAdapter(self.ip_pipe, image_encoder_ckpt_path, ip_adapter_ckpt_path, device)
    
    @torch.no_grad()    
    def interact_inference(self, prompt, subject_phrases, object_phrases, action_phrases, subject_bboxes, object_bboxes, 
                           beta=1, num_inference_steps=50, output_type="pil"):
        images = self.interact_pipe(
            prompt=prompt,
            interactdiffusion_subject_phrases=subject_phrases,
            interactdiffusion_object_phrases=object_phrases,
            interactdiffusion_action_phrases=action_phrases,
            interactdiffusion_subject_boxes=subject_bboxes,
            interactdiffusion_object_boxes=object_bboxes,
            interactdiffusion_scheduled_sampling_beta=beta,
            output_type=output_type,
            num_inference_steps=num_inference_steps,
            ).images

        images[0].save('interact_out.png')
    
    @torch.no_grad()
    def ipadapter_inference(self, prompt, ref_image_pil, scale=0.6, num_inference_steps=10, seed=42):
        images = self.ip_adapter.generate(
            pil_image=ref_image_pil, 
            num_samples=1, 
            num_inference_steps=num_inference_steps, 
            seed=seed,
            prompt=prompt, 
            scale=scale
        )[0]
        return images
        
        
    @torch.no_grad()
    def interactip_inference(
        self, 
        prompt: str = None,
        negative_prompt: str = None,
        interactdiffusion_subject_phrases: List[List[str]] = None,
        interactdiffusion_object_phrases: List[List[str]] = None,
        interactdiffusion_action_phrases: List[List[str]] = None,
        interactdiffusion_subject_boxes: List[List[List[float]]] = None,
        interactdiffusion_object_boxes: List[List[List[float]]] = None,
        interactdiffusion_scheduled_sampling_beta: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
       
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        ip_latents: Optional[torch.FloatTensor] = None,
        interact_latents: Optional[torch.FloatTensor] = None,
        eta: float = 0.0,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        
        ip_scale=1.0,
        ref_pil_image=None,
        num_samples=1,
        
        clip_skip: Optional[int] = None,
        
        omega: float = 0.5,
        is_mixed: bool = True,
        
        optim_start: int = 0,
        optim_end: int = 49,
        ip_p: float = 1.0,
        interact_p: float = 1.0
    ):
        # 1. Define call parameters
        device = self.interact_pipe._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        
        self.ip_adapter.set_scale(ip_scale)
        if ref_pil_image is not None:
            num_prompts = 1 if isinstance(ref_pil_image, Image.Image) else len(ref_pil_image)
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        
        batch_size = num_prompts
        prompt = [prompt] * num_prompts
        negative_prompt = [negative_prompt] * num_prompts
        
        height = height or self.interact_pipe.unet.config.sample_size * self.interact_pipe.vae_scale_factor     # 512
        width = width or self.interact_pipe.unet.config.sample_size * self.interact_pipe.vae_scale_factor       # 512
        
        # 2. [IP-Adapter] Encode reference image and prompt
        image_prompt_embeds, uncond_image_prompt_embeds = self.ip_adapter.get_image_embeds(
            pil_image=ref_pil_image, clip_image_embeds=None
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        
        with torch.inference_mode():
            ip_prompt_embeds_, ip_negative_prompt_embeds_ = self.ip_pipe.encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt
            )
            ip_prompt_embeds = torch.cat([ip_prompt_embeds_, image_prompt_embeds], dim=1)
            ip_negative_prompt_embeds = torch.cat([ip_negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        ip_prompt_embeds, ip_negative_prompt_embeds = self.ip_pipe.encode_prompt(
            prompt,
            device,
            num_samples,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=ip_prompt_embeds,
            negative_prompt_embeds=ip_negative_prompt_embeds,
            lora_scale=None,
            clip_skip=clip_skip
        )
        
        if do_classifier_free_guidance:
            ip_prompt_embeds = torch.cat([ip_negative_prompt_embeds, ip_prompt_embeds])

        # 3. [Interact] Encode input prompt
        interact_prompt_embeds, interact_negative_prompt_embeds = self.interact_pipe.encode_prompt(
            prompt,
            device,
            num_samples,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            clip_skip=clip_skip
        )
        
        if do_classifier_free_guidance:
           interact_prompt_embeds = torch.cat([interact_negative_prompt_embeds, interact_prompt_embeds])
        
        # 4. [Interact]Prepare timesteps
        self.interact_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.interact_pipe.scheduler.timesteps
        
        # 5.1. [Interact]Prepare latent variables
        num_channels_latents = self.interact_pipe.unet.config.in_channels
        interact_latents = self.interact_pipe.prepare_latents(
            batch_size * num_samples,
            num_channels_latents,
            height,
            width,
            interact_prompt_embeds.dtype,
            device,
            generator,
            interact_latents,
        )

        # 5.2. [IP-Adapter]Prepare IP-Adapter variables
        num_channels_latents = self.ip_pipe.unet.config.in_channels
        ip_latents = self.ip_pipe.prepare_latents(
            batch_size * num_samples,
            num_channels_latents,
            height,
            width,
            ip_prompt_embeds.dtype,
            device,
            generator,
            ip_latents,
        )

        # 6 [Interact]Prepare InteractDiffusion variables
        max_objs = 30
        if len(interactdiffusion_action_phrases) > max_objs:
            warnings.warn(
                f"More that {max_objs} objects found. Only first {max_objs} objects will be processed.",
                FutureWarning,
            )
            interactdiffusion_subject_phrases = interactdiffusion_subject_phrases[:max_objs]
            interactdiffusion_subject_boxes = interactdiffusion_subject_boxes[:max_objs]
            interactdiffusion_object_phrases = interactdiffusion_object_phrases[:max_objs]
            interactdiffusion_object_boxes = interactdiffusion_object_boxes[:max_objs]
            interactdiffusion_action_phrases = interactdiffusion_action_phrases[:max_objs]
        # prepare batched input to the InteractDiffusionInteractionProjection (boxes, phrases, mask)
        # Get tokens for phrases from pre-trained CLIPTokenizer
        tokenizer_inputs = self.interact_pipe.tokenizer(interactdiffusion_subject_phrases+interactdiffusion_object_phrases+interactdiffusion_action_phrases,
                                          padding=True, return_tensors="pt").to(device)
        # For the token, we use the same pre-trained text encoder
        # to obtain its text feature
        _text_embeddings = self.interact_pipe.text_encoder(**tokenizer_inputs).pooler_output
        n_objs = min(len(interactdiffusion_subject_boxes), max_objs)
        # For each entity, described in phrases, is denoted with a bounding box,
        # we represent the location information as (xmin,ymin,xmax,ymax)
        subject_boxes = torch.zeros(max_objs, 4, device=device, dtype=self.interact_pipe.text_encoder.dtype)
        object_boxes = torch.zeros(max_objs, 4, device=device, dtype=self.interact_pipe.text_encoder.dtype)
        subject_boxes[:n_objs] = torch.tensor(interactdiffusion_subject_boxes[:n_objs])
        object_boxes[:n_objs] = torch.tensor(interactdiffusion_object_boxes[:n_objs])
        subject_text_embeddings = torch.zeros(max_objs, 768, device=device, dtype=self.interact_pipe.text_encoder.dtype)
        subject_text_embeddings[:n_objs] = _text_embeddings[:n_objs*1]
        object_text_embeddings = torch.zeros(max_objs, 768, device=device, dtype=self.interact_pipe.text_encoder.dtype)
        object_text_embeddings[:n_objs] = _text_embeddings[n_objs*1:n_objs*2]
        action_text_embeddings = torch.zeros(max_objs, 768, device=device, dtype=self.interact_pipe.text_encoder.dtype)
        action_text_embeddings[:n_objs] = _text_embeddings[n_objs*2:n_objs*3]
        # Generate a mask for each object that is entity described by phrases
        masks = torch.zeros(max_objs, device=device, dtype=self.interact_pipe.text_encoder.dtype)
        masks[:n_objs] = 1

        repeat_batch = batch_size * num_samples
        subject_boxes = subject_boxes.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        object_boxes = object_boxes.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        subject_text_embeddings = subject_text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        object_text_embeddings = object_text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        action_text_embeddings = action_text_embeddings.unsqueeze(0).expand(repeat_batch, -1, -1).clone()
        masks = masks.unsqueeze(0).expand(repeat_batch, -1).clone()
        
        if do_classifier_free_guidance:
            repeat_batch = repeat_batch * 2
            subject_boxes = torch.cat([subject_boxes] * 2)
            object_boxes = torch.cat([object_boxes] * 2)
            subject_text_embeddings = torch.cat([subject_text_embeddings] * 2)
            object_text_embeddings = torch.cat([object_text_embeddings] * 2)
            action_text_embeddings = torch.cat([action_text_embeddings] * 2)
            masks = torch.cat([masks] * 2)
            masks[: repeat_batch // 2] = 0
        # if cross_attention_kwargs is None:
        #     cross_attention_kwargs = {}
        interact_cross_attention_kwargs = {}
        interact_cross_attention_kwargs['gligen'] = {
                'subject_boxes': subject_boxes,
                'object_boxes': object_boxes,
                'subject_positive_embeddings': subject_text_embeddings,
                'object_positive_embeddings': object_text_embeddings,
                'action_positive_embeddings': action_text_embeddings,
                'masks': masks
            }

        num_grounding_steps = int(interactdiffusion_scheduled_sampling_beta * len(timesteps))
        self.interact_pipe.enable_fuser(True)
        
        ip_cross_attention_kwargs = {}

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        interact_extra_step_kwargs = self.interact_pipe.prepare_extra_step_kwargs(generator, eta)
        ip_extra_step_kwargs = self.ip_pipe.prepare_extra_step_kwargs(generator, eta)
            
        # 8. Denoising loop
        interact_num_warmup_steps = len(timesteps) - num_inference_steps * self.interact_pipe.scheduler.order
        ip_num_warmup_steps = len(timesteps) - num_inference_steps * self.ip_pipe.scheduler.order

        with self.interact_pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                # [Interact]Scheduled sampling
                if i == num_grounding_steps:
                    self.interact_pipe.enable_fuser(False)

                if interact_latents.shape[1] != 4:
                    interact_latents = torch.randn_like(interact_latents[:, :4])    
                    
                    
                # expand the latents if we are doing classifier free guidance
                interact_latent_model_input = torch.cat([interact_latents] * 2) if do_classifier_free_guidance else interact_latents
                interact_latent_model_input = self.interact_pipe.scheduler.scale_model_input(interact_latent_model_input, t)
                
                ip_latent_model_input = torch.cat([ip_latents] * 2) if do_classifier_free_guidance else ip_latents
                ip_latent_model_input = self.ip_pipe.scheduler.scale_model_input(ip_latent_model_input, t)
                

                # predict the noise residual
                interact_noise_pred = self.interact_pipe.unet(
                    interact_latent_model_input,
                    t,
                    encoder_hidden_states=interact_prompt_embeds,
                    cross_attention_kwargs=interact_cross_attention_kwargs,
                ).sample
                
                ip_noise_pred = self.ip_pipe.unet(
                    ip_latent_model_input,
                    t,
                    encoder_hidden_states=ip_prompt_embeds,
                    timestep_cond=None,
                    cross_attention_kwargs=ip_cross_attention_kwargs,
                    added_cond_kwargs=None,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    ip_noise_pred_uncond, ip_noise_pred_text = ip_noise_pred.chunk(2)
                    ip_noise_pred = ip_noise_pred_uncond + guidance_scale * (
                            ip_noise_pred_text - ip_noise_pred_uncond
                    )
                    
                    interact_noise_pred_uncond, interact_noise_pred_text = interact_noise_pred.chunk(2)
                    interact_noise_pred = interact_noise_pred_uncond + guidance_scale * (interact_noise_pred_text - interact_noise_pred_uncond)
                
                ip_latents = self.ip_pipe.scheduler.step(ip_noise_pred, t, ip_latents, **ip_extra_step_kwargs, return_dict=False)[0]
                interact_latents = self.interact_pipe.scheduler.step(interact_noise_pred, t, interact_latents, **interact_extra_step_kwargs).prev_sample
                
                if is_mixed: 
                    ################################### CORE ###################################

                    # spherical linear interpolation aggregation
                    mixed_latents = omega * interact_latents + ((1. - omega ** 2) ** 0.5) * ip_latents 
                    
                    if i > 0 and i >= optim_start and i <= optim_end:
                        # manifold optimization for IP-Adapter
                        ip_latents = optimize(
                            pipeline=self.ip_pipe, 
                            latents=ip_latents,
                            prev_latents=prev_ip_latents,
                            mixed_latents=mixed_latents,
                            timestep=t, 
                            encoder_hidden_states=ip_prompt_embeds,
                            cross_attention_kwargs=ip_cross_attention_kwargs,
                            guidance_scale=guidance_scale, 
                            do_classifier_free_guidance=do_classifier_free_guidance,
                            p=ip_p,
                            name='ip',
                            prev_noise_pred=prev_ip_noise_pred
                        )
                        
                        # manifold optimization for InteractDiffusion
                        interact_latents = optimize(
                            pipeline=self.interact_pipe, 
                            latents=interact_latents,
                            prev_latents=prev_interact_latents,
                            mixed_latents=mixed_latents, 
                            timestep=t, 
                            encoder_hidden_states=interact_prompt_embeds,
                            cross_attention_kwargs=interact_cross_attention_kwargs,
                            guidance_scale=guidance_scale, 
                            do_classifier_free_guidance=do_classifier_free_guidance,
                            p=interact_p,
                            name='interact',
                            prev_noise_pred=prev_interact_noise_pred
                        )
                        
                    prev_ip_latents = ip_latents
                    prev_interact_latents = interact_latents
                    prev_ip_noise_pred = ip_noise_pred
                    prev_interact_noise_pred = interact_noise_pred

                    ############################################################################
                
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > ip_num_warmup_steps and (i + 1) % self.ip_pipe.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, ip_latents)
                        callback(i, t, interact_latents)

        if output_type == "latent":
            ip_image = ip_latents
            interact_image = interact_latents
        elif output_type == "pil":
            # 8. Post-processing
            ip_image = self.ip_pipe.decode_latents(ip_latents)
            ip_image = self.ip_pipe.numpy_to_pil(ip_image)
            
            interact_image = self.interact_pipe.vae.decode(interact_latents / self.interact_pipe.vae.config.scaling_factor, return_dict=False)[0]
            interact_image = (interact_image / 2 + 0.5).clamp(0, 1)
            interact_image = interact_image.cpu().permute(0, 2, 3, 1).float().numpy()
            interact_image = self.interact_pipe.numpy_to_pil(interact_image)
        else:
            # 8. Post-processing
            ip_image = self.ip_pipe.decode_latents(ip_latents)
            
            interact_image = self.interact_pipe.vae.decode(interact_latents / self.interact_pipe.vae.config.scaling_factor, return_dict=False)[0]
            interact_image = (interact_image / 2 + 0.5).clamp(0, 1)
            interact_image = interact_image.cpu().permute(0, 2, 3, 1).float().numpy()


        self.interact_pipe.maybe_free_model_hooks()
        self.ip_pipe.maybe_free_model_hooks()
        
        if not return_dict:
            return (ip_image, None), (interact_image, None)
        
        return StableDiffusionPipelineOutput(images=ip_image, nsfw_content_detected=None), StableDiffusionPipelineOutput(images=interact_image, nsfw_content_detected=None)
