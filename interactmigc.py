from modules import StableDiffusionMIGCPipeline, load_migc, AttentionStore, MIGCProcessor, migc_seed_everything, InteractDiffusionPipeline
from diffusers import EulerDiscreteScheduler
import torch 
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.pipelines.stable_diffusion import (
    StableDiffusionPipeline,
    StableDiffusionPipelineOutput,
    StableDiffusionSafetyChecker,
)
import warnings
import time

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
    name="interactdiffusion",
    prev_noise_pred=None
):

    latents_prev = prev_latents
    prev_timestep = timestep + pipeline.scheduler.config.num_train_timesteps // pipeline.scheduler.num_inference_steps     # t + 1
    mu = compute_mu(latents_prev, prev_timestep, pipeline, encoder_hidden_states, cross_attention_kwargs, guidance_scale, do_classifier_free_guidance, prev_noise_pred)
    optimized_latents = mixed_latents - p * (mixed_latents - mu) / torch.norm(mixed_latents - mu)
    
    return optimized_latents
    
class InteractMIGC:
    def __init__(
        self, 
        migc_ckpt_path='checkpoints/migc/MIGC_SD14.ckpt', 
        sd1x_ckpt_path='checkpoints/stablediffusion/v1/huggingfacepipeline',
        interact_ckpt_path='checkpoints/interactdiffusion',
        device="cuda:1"
    ):
        # InteractDiffusion setup
        self.interact_pipe = InteractDiffusionPipeline.from_pretrained(
            interact_ckpt_path,
            torch_dtype=torch.float16
        )
        self.interact_pipe = self.interact_pipe.to(device)
        
        # MIGC setup
        self.migc_pipe = StableDiffusionMIGCPipeline.from_pretrained(
            sd1x_ckpt_path,
            torch_dtype=torch.float16
        )
        self.migc_pipe.attention_store = AttentionStore()
        load_migc(self.migc_pipe.unet , self.migc_pipe.attention_store,
                migc_ckpt_path, attn_processor=MIGCProcessor)
        self.migc_pipe = self.migc_pipe.to(device)
        self.migc_pipe.scheduler = self.interact_pipe.scheduler
        
    def migc_inference(self, prompt_final, bboxes, negative_prompt, seed, num_inference_steps=50, guidance_scale=7.5):
        migc_seed_everything(seed)
        image = self.migc_pipe(prompt_final, bboxes, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, 
                    MIGCsteps=25, aug_phase_with_and=False, negative_prompt=negative_prompt).images[0]
        image.save('migc_output.png')
        image = self.migc_pipe.draw_box_desc(image, bboxes[0], prompt_final[0][1:])
        image.save('migc_anno_output.png')
        
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
    def interactmigc_inference(
        self, 
        caption: List[str] = None,
        prompt: List[List[str]] = None,
        bboxes: List[List[List[float]]] = None,
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
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        migc_latents: Optional[torch.FloatTensor] = None,
        interact_latents: Optional[torch.FloatTensor] = None,
        eta: float = 0.0,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        # cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        
        MIGCsteps=20,
        NaiveFuserSteps=-1,
        ca_scale=None,
        ea_scale=None,
        sac_scale=None,
        aug_phase_with_and=False,
        GUI_progress=None,
        
        clip_skip: Optional[int] = None,
        
        omega: float = 0.5,
        is_mixed: bool = True,
        
        optim_start: int = 0,
        optim_end: int = 49,
        migc_p: float = 1.0,
        interact_p: float = 1.0
    ):
        
        
        # Copied from StableDiffusionMIGCPipeline.__call__()
        def aug_phase_with_and_function(phase, instance_num):
            instance_num = min(instance_num, 7)
            copy_phase = [phase] * instance_num
            phase = ', and '.join(copy_phase)
            return phase
        
        if aug_phase_with_and:
            instance_num = len(prompt[0]) - 1
            for i in range(1, len(prompt[0])):
                prompt[0][i] = aug_phase_with_and_function(prompt[0][i],
                                                            instance_num)

        # 0. Default height and width to unet
        height = height or self.migc_pipe.unet.config.sample_size * self.migc_pipe.vae_scale_factor     # 512
        width = width or self.migc_pipe.unet.config.sample_size * self.migc_pipe.vae_scale_factor       # 512
        

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        prompt_nums = [0] * len(prompt)
        for i, _ in enumerate(prompt):
            prompt_nums[i] = len(_)

        device = self.migc_pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. [MIGC]Encode input prompt
        migc_prompt_embeds, migc_cond_prompt_embeds, migc_embeds_pooler = self.migc_pipe._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 3. [Interact]Encode input prompt
        interact_prompt_embeds, interact_negative_prompt_embeds = self.interact_pipe.encode_prompt(
            caption,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_skip=clip_skip,
        )
        
        # 3.1. [Interact]Concat prompt embeds
        if do_classifier_free_guidance:
           interact_prompt_embeds = torch.cat([interact_negative_prompt_embeds, interact_prompt_embeds])
        
        # 4. [MIGC]Prepare timesteps
        self.migc_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        migc_timesteps = self.migc_pipe.scheduler.timesteps
        
        # 4. [Interact]Prepare timesteps
        self.interact_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        interact_timesteps = self.interact_pipe.scheduler.timesteps

        # 5. [MIGC]Prepare latent variables
        num_channels_latents = self.migc_pipe.unet.config.in_channels
        migc_latents = self.migc_pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            migc_prompt_embeds.dtype,
            device,
            generator,
            migc_latents,
        )
        
        # 5. [Interact]Prepare latent variables
        num_channels_latents = self.interact_pipe.unet.config.in_channels
        interact_latents = self.interact_pipe.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            interact_prompt_embeds.dtype,
            device,
            generator,
            interact_latents,
        )

        # 5.1. [Interact]Prepare InteractDiffusion variables
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

        repeat_batch = batch_size * num_images_per_prompt
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

        num_grounding_steps = int(interactdiffusion_scheduled_sampling_beta * len(interact_timesteps))
        self.interact_pipe.enable_fuser(True)

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        migc_extra_step_kwargs = self.migc_pipe.prepare_extra_step_kwargs(generator, eta)
        interact_extra_step_kwargs = self.interact_pipe.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        migc_num_warmup_steps = len(migc_timesteps) - num_inference_steps * self.migc_pipe.scheduler.order
        interact_num_warmup_steps = len(interact_timesteps) - num_inference_steps * self.interact_pipe.scheduler.order

        with self.migc_pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, (migc_t, interact_t) in enumerate(zip(migc_timesteps, interact_timesteps)):
                if GUI_progress is not None:
                    GUI_progress[0] = int((i + 1) / len(migc_timesteps) * 100)
                    
                # [Interact]Scheduled sampling
                if i == num_grounding_steps:
                    self.interact_pipe.enable_fuser(False)

                if interact_latents.shape[1] != 4:
                    interact_latents = torch.randn_like(interact_latents[:, :4])    
                    
                    
                # expand the latents if we are doing classifier free guidance
                migc_latent_model_input = (
                    torch.cat([migc_latents] * 2) if do_classifier_free_guidance else migc_latents
                )

                migc_latent_model_input = self.migc_pipe.scheduler.scale_model_input(
                    migc_latent_model_input, migc_t
                )
                
                interact_latent_model_input = torch.cat([interact_latents] * 2) if do_classifier_free_guidance else interact_latents
                interact_latent_model_input = self.interact_pipe.scheduler.scale_model_input(interact_latent_model_input, interact_t)
                

                # predict the noise residual
                migc_cross_attention_kwargs = {'prompt_nums': prompt_nums,
                                          'bboxes': bboxes,
                                          'ith': i,
                                          'embeds_pooler': migc_embeds_pooler,
                                          'timestep': migc_t,
                                          'height': height,
                                          'width': width,
                                          'MIGCsteps': MIGCsteps,
                                          'NaiveFuserSteps': NaiveFuserSteps,
                                          'ca_scale': ca_scale,
                                          'ea_scale': ea_scale,
                                          'sac_scale': sac_scale}
                self.migc_pipe.unet.eval()
                migc_noise_pred = self.migc_pipe.unet(
                    migc_latent_model_input,
                    migc_t,
                    encoder_hidden_states=migc_prompt_embeds,
                    cross_attention_kwargs=migc_cross_attention_kwargs,
                ).sample
                
                interact_noise_pred = self.interact_pipe.unet(
                    interact_latent_model_input,
                    interact_t,
                    encoder_hidden_states=interact_prompt_embeds,
                    cross_attention_kwargs=interact_cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    migc_noise_pred_uncond, migc_noise_pred_text = migc_noise_pred.chunk(2)
                    migc_noise_pred = migc_noise_pred_uncond + guidance_scale * (
                            migc_noise_pred_text - migc_noise_pred_uncond
                    )
                    
                    interact_noise_pred_uncond, interact_noise_pred_text = interact_noise_pred.chunk(2)
                    interact_noise_pred = interact_noise_pred_uncond + guidance_scale * (interact_noise_pred_text - interact_noise_pred_uncond)
                
                migc_step_output = self.migc_pipe.scheduler.step(
                    migc_noise_pred, migc_t, migc_latents, **migc_extra_step_kwargs
                )
                migc_latents = migc_step_output.prev_sample
                
                interact_latents = self.interact_pipe.scheduler.step(interact_noise_pred, interact_t, interact_latents, **interact_extra_step_kwargs).prev_sample
                    

                    ################################### CORE ###################################

                if is_mixed: # spherical linear interpolation aggregation
                    mixed_latents = omega * interact_latents + ((1. - omega ** 2) ** 0.5) * migc_latents 
                    
                    if i > 0 and i >= optim_start and i <= optim_end:
                        # manifold optimization for MIGC
                        migc_latents = optimize(
                            pipeline=self.migc_pipe, 
                            latents=migc_latents,
                            prev_latents=prev_migc_latents,
                            mixed_latents=mixed_latents,
                            timestep=migc_t, 
                            encoder_hidden_states=migc_prompt_embeds,
                            cross_attention_kwargs=migc_cross_attention_kwargs,
                            guidance_scale=guidance_scale, 
                            do_classifier_free_guidance=do_classifier_free_guidance,
                            p=migc_p,
                            name='migc',
                            prev_noise_pred=prev_migc_noise_pred
                        )
                        
                        
                        # manifold optimization for InteractDiffusion
                        interact_latents = optimize(
                            pipeline=self.interact_pipe, 
                            latents=interact_latents,
                            prev_latents=prev_interact_latents,
                            mixed_latents=mixed_latents, 
                            timestep=interact_t, 
                            encoder_hidden_states=interact_prompt_embeds,
                            cross_attention_kwargs=interact_cross_attention_kwargs,
                            guidance_scale=guidance_scale, 
                            do_classifier_free_guidance=do_classifier_free_guidance,
                            p=interact_p,
                            name='interact',
                            prev_noise_pred=prev_interact_noise_pred
                        )
                        
                    prev_migc_latents = migc_latents
                    prev_interact_latents = interact_latents
                    prev_migc_noise_pred = migc_noise_pred
                    prev_interact_noise_pred = interact_noise_pred

                    ############################################################################
                
                
                # call the callback, if provided
                if i == len(migc_timesteps) - 1 or (
                        (i + 1) > migc_num_warmup_steps and (i + 1) % self.migc_pipe.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, migc_t, migc_latents)
                        callback(i, interact_t, interact_latents)

        if output_type == "latent":
            migc_image = migc_latents
            interact_image = interact_latents
        elif output_type == "pil":
            # 8. Post-processing
            migc_image = self.migc_pipe.decode_latents(migc_latents)
            migc_image = self.migc_pipe.numpy_to_pil(migc_image)
            
            interact_image = self.interact_pipe.vae.decode(interact_latents / self.interact_pipe.vae.config.scaling_factor, return_dict=False)[0]
            interact_image = (interact_image / 2 + 0.5).clamp(0, 1)
            interact_image = interact_image.cpu().permute(0, 2, 3, 1).float().numpy()
            interact_image = self.interact_pipe.numpy_to_pil(interact_image)
        else:
            # 8. Post-processing
            migc_image = self.migc_pipe.decode_latents(migc_latents)
            
            interact_image = self.interact_pipe.vae.decode(interact_latents / self.interact_pipe.vae.config.scaling_factor, return_dict=False)[0]
            interact_image = (interact_image / 2 + 0.5).clamp(0, 1)
            interact_image = interact_image.cpu().permute(0, 2, 3, 1).float().numpy()

        # Offload last model to CPU
        if hasattr(self.migc_pipe, "final_offload_hook") and self.migc_pipe.final_offload_hook is not None:
            self.migc_pipe.final_offload_hook.offload()

        self.interact_pipe.maybe_free_model_hooks()
        
        if not return_dict:
            return (migc_image, None), (interact_image, None)
        
        return StableDiffusionPipelineOutput(images=migc_image, nsfw_content_detected=None), StableDiffusionPipelineOutput(images=interact_image, nsfw_content_detected=None)



if __name__ == '__main__':
    model = InteractMIGC()
    
    caption = 'a white dressed man is feeding an orange cat'
    negative_caption = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    migc_prompt = [[caption, 'white dressed man', 'orange cat']]
    migc_bboxes = [[[0.0332, 0.1660, 0.3359, 0.7305],[0.2891, 0.4766, 0.6680, 0.7930]]]

    interactdiffusion_subject_phrases=["white dressed man"]
    interactdiffusion_object_phrases=["orange cat"]
    interactdiffusion_action_phrases=["feeding"]
    interactdiffusion_subject_boxes=[[0.0332, 0.1660, 0.3359, 0.7305]]
    interactdiffusion_object_boxes=[[0.2891, 0.4766, 0.6680, 0.7930]]
    
    migc_seed_everything(122955)
    
    migc_images, interact_images = model.interactmigc_inference(
        prompt=migc_prompt, bboxes=migc_bboxes, caption=caption, negative_prompt=negative_caption,
        interactdiffusion_subject_phrases=interactdiffusion_subject_phrases,
        interactdiffusion_object_phrases=interactdiffusion_object_phrases,
        interactdiffusion_action_phrases=interactdiffusion_action_phrases,
        interactdiffusion_subject_boxes=interactdiffusion_subject_boxes,
        interactdiffusion_object_boxes=interactdiffusion_object_boxes,
        omega=0.5,
        num_inference_steps=50,
        optim_start=1,
        optim_end=7,
        migc_p=45,
        interact_p=55,
        is_mixed=True,
        )

    interact_images = interact_images.images
    interact_images[0].save("InteractDiffusion(+MIGC).png")

    migc_images, interact_images = model.interactmigc_inference(
        prompt=migc_prompt, bboxes=migc_bboxes, caption=caption, negative_prompt=negative_caption,
        interactdiffusion_subject_phrases=interactdiffusion_subject_phrases,
        interactdiffusion_object_phrases=interactdiffusion_object_phrases,
        interactdiffusion_action_phrases=interactdiffusion_action_phrases,
        interactdiffusion_subject_boxes=interactdiffusion_subject_boxes,
        interactdiffusion_object_boxes=interactdiffusion_object_boxes,
        omega=0.5,
        num_inference_steps=50,
        optim_start=1,
        optim_end=3,
        migc_p=45,
        interact_p=55,
        is_mixed=False
        )
    migc_images = migc_images.images
    interact_images = interact_images.images
    migc_images[0].save("MIGC_origin.png")
    interact_images[0].save("InteractDiffusion_origin.png")
        
        
        
        
        