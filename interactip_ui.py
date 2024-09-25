import gradio as gr
from interactip import InteractIP
import random
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
import argparse
import yaml
import torch
from modules import get_generator
import os

DEVICE = 'cpu'
os.environ["GRADIO_TEMP_DIR"] = "tmp"

def draw_bounding_boxes(
    boxes,
    labels=None,
    image_size=(512, 512), 
    bg_image=None, 
    font_size=20, 
    box_width=5
):  
    bg_image = deepcopy(bg_image)
    white_image = Image.new("RGB", (image_size[0], image_size[1]), "white")
    draw_white = ImageDraw.Draw(white_image)
    if bg_image is not None:
        draw_bg_img = ImageDraw.Draw(bg_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        # print("using font: arial.ttf with size ", font_size)
    except IOError:
        font = ImageFont.load_default(font_size)
        # print("using default font with size ", font_size)

    for box, label in zip(boxes, labels):            
        box = [ int(coord * image_size[1]) if idx % 2 else int(coord * image_size[0]) for (idx, coord) in enumerate(box) ]
        # print(label, box)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        draw_white.rectangle(box, outline=color, width=box_width)
        
        text_position = (box[0] + box_width, box[1] + box_width)        # 左上
        draw_white.text(text_position, label, fill=color, font=font)
        
        if bg_image is not None:
            draw_bg_img.rectangle(box, outline=color, width=box_width)
            draw_bg_img.text(text_position, label, fill=color, font=font)

    if bg_image is not None:
        return white_image, bg_image
    else:
        return white_image

# Preprocessing instance description and location information
def process_phrases_locations(phrases, locations):
    phrase_each = phrases.strip().split(';')
    location_each = locations.strip().split(';')
    assert len(phrase_each) == len(location_each)
    phrases = []
    locations = []
    for p, l in zip(phrase_each, location_each):
        # print(p, l)
        p = p.strip()
        phrases.append(p)
        
        l = l.strip().split(',')
        assert len(l) == 4
        l = [float(coord.strip()) for coord in l]
        locations.append(l)
    return phrases, locations

# 边界框预览
def boxes_preview(subject_boxes, object_boxes, subject_labels, object_labels, font_size=20, box_width=5, bg=None):
    phrases = ";".join([subject_labels, object_labels])
    locations = ";".join([subject_boxes,object_boxes])
    phrases, locations = process_phrases_locations(phrases, locations)
    boxes_preview_pil = draw_bounding_boxes(locations, phrases, font_size=font_size, box_width=box_width, bg_image=bg)
    return boxes_preview_pil

@torch.no_grad()
def generate(prompt, negative_prompt, subject_boxes, object_boxes, subjects, objects, actions, omega, total_steps, seed=None, 
             optim_start=None, optim_end=None, ip_scale_p=1.0, interact_scale_p=1.0,
             ref_pil_image=None, ip_scale=0.8):
    subject_phrases, subject_locations = process_phrases_locations(subjects, subject_boxes)
    object_phrases, object_locations = process_phrases_locations(objects, object_boxes)
    actions_each = actions.strip().split(';')
    actions = []
    for a in actions_each:
        a = a.strip()
        actions.append(a)
    
    
    interactdiffusion_subject_phrases=subject_phrases
    interactdiffusion_object_phrases=object_phrases
    interactdiffusion_action_phrases=actions
    interactdiffusion_subject_boxes=subject_locations
    interactdiffusion_object_boxes=object_locations
    
    if seed == "":
        seed = random.randint(0, 1000000)
    else:
        seed = eval(seed)
        
    generator = get_generator(seed, DEVICE)
    print("set seed ", seed)
    
    ip_images, interact_images = MODEL.interactip_inference(
        prompt=prompt, negative_prompt=negative_prompt,
        interactdiffusion_subject_phrases=interactdiffusion_subject_phrases,
        interactdiffusion_object_phrases=interactdiffusion_object_phrases,
        interactdiffusion_action_phrases=interactdiffusion_action_phrases,
        interactdiffusion_subject_boxes=interactdiffusion_subject_boxes,
        interactdiffusion_object_boxes=interactdiffusion_object_boxes,
        omega=omega,
        num_inference_steps=total_steps,
        is_mixed=True,
        optim_start=optim_start, 
        optim_end=optim_end,
        ip_p=ip_scale_p,
        interact_p=interact_scale_p,
        ref_pil_image=ref_pil_image,
        ip_scale=ip_scale,
        generator=generator
        )
    
    generator = get_generator(seed, DEVICE)
    
    ip_images_origin, interact_images_origin = MODEL.interactip_inference(
        prompt=prompt, negative_prompt=negative_prompt,
        interactdiffusion_subject_phrases=interactdiffusion_subject_phrases,
        interactdiffusion_object_phrases=interactdiffusion_object_phrases,
        interactdiffusion_action_phrases=interactdiffusion_action_phrases,
        interactdiffusion_subject_boxes=interactdiffusion_subject_boxes,
        interactdiffusion_object_boxes=interactdiffusion_object_boxes,
        omega=omega,
        num_inference_steps=total_steps,
        is_mixed=False,
        optim_start=optim_start, 
        optim_end=optim_end,
        ip_p=ip_scale_p,
        interact_p=interact_scale_p,
        ref_pil_image=ref_pil_image,
        ip_scale=ip_scale,
        generator=generator
        )
    
    interact_result = boxes_preview(subject_boxes, object_boxes, subjects, objects, bg=interact_images.images[0])[1]
    ip_result_origin = boxes_preview(subject_boxes, object_boxes, subjects, objects, bg=ip_images_origin.images[0])[1]
    interact_result_origin = boxes_preview(subject_boxes, object_boxes, subjects, objects, bg=interact_images_origin.images[0])[1]
    
    return interact_result, ip_result_origin, interact_result_origin
    
def get_args():
    parser = argparse.ArgumentParser(description='InteractIP Args')
    parser.add_argument('--config', type=str, default='configs/inference2.yaml',
                        help='Path to the config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device name, default to `cuda`')
    parser.add_argument('--port', type=int, default=8082,
                        help='port name, default to `8082`')

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = get_args()
    DEVICE = args.device
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    MODEL = InteractIP(
        interact_ckpt_path=config['interact_ckpt_path'],
        ip_base_model_ckpt_path=config['ip_base_model_ckpt_path'],
        ip_adapter_ckpt_path=config['ip_adapter_ckpt_path'],
        vae_ckpt_path=config['vae_ckpt_path'],
        image_encoder_ckpt_path=config['image_encoder_ckpt_path'],
        device=args.device
    )
    
    with gr.Blocks() as UI:
        with gr.Row():
            with gr.Column():
                caption = gr.Textbox(label="Caption", value='A girl is feeding a cat, style painting')
                negative_caption = gr.Textbox(label="Negative Caption", value='monochrome, lowres, bad anatomy, worst quality, low quality')
                with gr.Blocks():
                    introduction1 = gr.Label(value="A set of input contains subject-action-object, where the subject/object is a text description, and the position is controlled by the bounding box")
                    action_subject = gr.Textbox(label="Subject (Separate with semicolons)", value="a girl")
                    subject_boxes = gr.Textbox(label="Bondding Box of Subjects (The bboxes of different subjects are separated by semicolons, and the coordinates of the same subject are separated by commas.)", value="0.2,0.05,0.8,1.0")
                    action_object = gr.Textbox(label="Object (Separate with semicolons)", value='a cat')
                    object_boxes = gr.Textbox(label="Bondding Box of Objects (The bboxes of different subjects are separated by semicolons, and the coordinates of the same subject are separated by commas.)", value='0.7,0.7,1.0,1.0')
                    action = gr.Textbox(label="Action", value='feeding')
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            omega = gr.Number(label="Weight w (0-1)", minimum=0, value=0.45, interactive=True)
                            total_steps = gr.Number(label="Sampling Step", minimum=1, maximum=100, step=1, value=10, interactive=True)
                    with gr.Column():
                        ip_scale_p = gr.Number(label="IP-Adapter Optimization Step", value=45, interactive=True)
                        interact_scale_p = gr.Number(label="Interact Optimization Step", value=55, interactive=True)
                with gr.Row():
                    optim_start = gr.Number(label="Start of Aggregation", value=1, interactive=True)
                    optim_end = gr.Number(label="End of Aggregation", value=3, interactive=True)
                seed = gr.Textbox(label="Random Seed", value="1", interactive=True)
                ip_scale = gr.Number(label="IP-Adapter Hyperparameters", value=0.8, interactive=True)
            with gr.Column():
                boxes_board = gr.Image(label="Overview of Bondding Box")
                boxes_preview_btn = gr.Button(value="Overview of Bondding Box")
                ref_image = gr.Image(label="Reference image of IP-Adapter", type='pil')
                
        generate_btn = gr.Button(value="Generate")
        with gr.Row(): 
            interact_result = gr.Image(label="InteractDiffusion(+IP-Adapter)", type='pil')
            ip_result_origin = gr.Image(label="IP-Adapter Origin", type='pil')
            interact_result_origin = gr.Image(label="InteractDiffusion Origin", type='pil')
        
        
        boxes_preview_btn.click(
            boxes_preview,
            inputs=[subject_boxes, object_boxes, action_subject, action_object],
            outputs=[boxes_board]
        )
        
        generate_btn.click(
            generate,
            inputs=[caption, negative_caption, subject_boxes, object_boxes, action_subject, action_object, action, omega, total_steps, seed, 
                    optim_start, optim_end, ip_scale_p, interact_scale_p,
                    ref_image, ip_scale],
            outputs=[interact_result, ip_result_origin, interact_result_origin]
        )
                
    
    UI.launch(server_name='0.0.0.0', server_port=args.port, share=True)
    