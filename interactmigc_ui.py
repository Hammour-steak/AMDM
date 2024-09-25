import gradio as gr
from interactmigc import InteractMIGC
import random
from PIL import Image, ImageDraw, ImageFont
from copy import deepcopy
from modules import migc_seed_everything
import argparse
import yaml
import torch
import os

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

def boxes_preview(subject_boxes, object_boxes, subject_labels, object_labels, font_size=20, box_width=5, bg=None):
    phrases = ";".join([subject_labels, object_labels])
    locations = ";".join([subject_boxes,object_boxes])
    phrases, locations = process_phrases_locations(phrases, locations)
    boxes_preview_pil = draw_bounding_boxes(locations, phrases, font_size=font_size, box_width=box_width, bg_image=bg)
    return boxes_preview_pil

@torch.no_grad()
def generate(caption, negative_caption, subject_boxes, object_boxes, subjects, objects, actions, omega, total_steps, seed=None, 
             optim_start=None, optim_end=None, migc_scale_p=1.0, interact_scale_p=1.0,
             MIGCsteps=25, NaiveFuserSteps=25, ca_scale=1, ea_scale=1, sac_scale=1):
    subject_phrases, subject_locations = process_phrases_locations(subjects, subject_boxes)
    object_phrases, object_locations = process_phrases_locations(objects, object_boxes)
    actions_each = actions.strip().split(';')
    actions = []
    for a in actions_each:
        a = a.strip()
        actions.append(a)
    
    migc_prompt = [[caption] + subject_phrases + object_phrases]
    migc_bboxes = [subject_locations + object_locations]
    
    ca_scale = None
    ea_scale = None
    sac_scale = None
    
    interactdiffusion_subject_phrases=subject_phrases
    interactdiffusion_object_phrases=object_phrases
    interactdiffusion_action_phrases=actions
    interactdiffusion_subject_boxes=subject_locations
    interactdiffusion_object_boxes=object_locations

    if seed == "":
        seed = random.randint(0, 1000000)
    else:
        seed = eval(seed)
    migc_seed_everything(seed)
    print("set seed ", seed)
    
    migc_images, interact_images = MODEL.interactmigc_inference(
        prompt=migc_prompt, bboxes=migc_bboxes, caption=caption, negative_prompt=negative_caption,
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
        migc_p=migc_scale_p,
        interact_p=interact_scale_p,
        MIGCsteps=MIGCsteps,
        NaiveFuserSteps=NaiveFuserSteps,
        ca_scale=ca_scale,
        ea_scale=ea_scale,
        sac_scale=sac_scale
        )
    
    migc_seed_everything(seed)
    
    migc_images_origin, interact_images_origin = MODEL.interactmigc_inference(
        prompt=migc_prompt, bboxes=migc_bboxes, caption=caption, negative_prompt=negative_caption,
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
        migc_p=migc_scale_p,
        interact_p=interact_scale_p,
        MIGCsteps=MIGCsteps,
        NaiveFuserSteps=NaiveFuserSteps,
        ca_scale=ca_scale,
        ea_scale=ea_scale,
        sac_scale=sac_scale
        )
    
    interact_result = boxes_preview(subject_boxes, object_boxes, subjects, objects, bg=interact_images.images[0])[1]
    migc_result_origin = boxes_preview(subject_boxes, object_boxes, subjects, objects, bg=migc_images_origin.images[0])[1]
    interact_result_origin = boxes_preview(subject_boxes, object_boxes, subjects, objects, bg=interact_images_origin.images[0])[1]
    
    return interact_result, migc_result_origin, interact_result_origin
    
def get_args():
    parser = argparse.ArgumentParser(description='InteractMIGC Args')
    parser.add_argument('--config', type=str, default='configs/inference.yaml',
                        help='Path to the config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device name, default to `cuda`')
    parser.add_argument('--port', type=int, default=8081,
                        help='port name, default to `8080`')

    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = get_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    MODEL = InteractMIGC(
        migc_ckpt_path=config['migc_ckpt_path'], 
        sd1x_ckpt_path=config['sd1x_ckpt_path'],
        interact_ckpt_path=config['interact_ckpt_path'],
        device=args.device
    )
    
    with gr.Blocks() as UI:
        with gr.Row():
            with gr.Column():
                caption = gr.Textbox(label="Caption", value='a man in red suit is holding a blue bag, a man in black suit is carrying a yellow suitcase')
                negative_caption = gr.Textbox(label="Negative Caption", value='worst quality, low quality, bad anatomy, watermark, text, blurry')
                with gr.Blocks():
                    introduction1 = gr.Label(value="A set of input contains subject-action-object, where the subject/object is a text description, and the position is controlled by the bounding box")
                    action_subject = gr.Textbox(label="Subject (Separate with semicolons)", value="a man in red suit; a man in black suit")
                    subject_boxes = gr.Textbox(label="Bondding Box of Subjects (The bboxes of different subjects are separated by semicolons, and the coordinates of the same subject are separated by commas.)", value="0.071875,0.0585480093676815,0.4453125,0.9882903981264637;0.671875,0.0585480093676815,0.9453125,0.9882903981264637")
                    action_object = gr.Textbox(label="Object (Separate with semicolons)", value='a blue bag; a yellow suitcase')
                    object_boxes = gr.Textbox(label="Bondding Box of Objects (The bboxes of different subjects are separated by semicolons, and the coordinates of the same subject are separated by commas.)", value='0.0075,0.404918032786885,0.534375,0.6037939110070258;0.575,0.404918032786885,0.7634375,0.6037939110070258')
                    action = gr.Textbox(label="Action", value='holding; carrying')
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            omega = gr.Number(label="Weight w (0-1)", minimum=0, value=0.5, interactive=True)
                            total_steps = gr.Number(label="Sampling Step", minimum=1, maximum=100, step=1, value=20, interactive=True)
                    with gr.Column():
                        migc_scale_p = gr.Number(label="MIGC Optimization Step", value=45, interactive=True)
                        interact_scale_p = gr.Number(label="Interact Optimization Step", value=55, interactive=True)
                with gr.Row():
                    optim_start = gr.Number(label="Start of Aggregation", value=1, interactive=True)
                    optim_end = gr.Number(label="End of Aggregation (s)", value=3, interactive=True)
                seed = gr.Textbox(label="Random Seed", value="1", interactive=True)
            with gr.Column():
                boxes_board = gr.Image(label="Overview of Bondding Box")
                boxes_preview_btn = gr.Button(value="Overview of Bondding Box")
        with gr.Row():
            with gr.Column():
                with gr.Accordion("MIGC Hyperparameters", open=False):
                    MIGCsteps = gr.Number(label="MIGCsteps", value=25, interactive=True)
                    NaiveFuserSteps = gr.Number(label="NaiveFuserSteps", value=25, interactive=True)
                    ca_scale = gr.Number(label="ca_scale", value=1, interactive=True)
                    ea_scale = gr.Number(label="ea_scale", value=1, interactive=True)
                    sac_scale = gr.Number(label="sac_scale", value=1, interactive=True)
            with gr.Column():
                with gr.Accordion("InteractDiffusion Hyperparameters", open=False):
                    interact_label = gr.Label(value="InteractDiffusion Hyperparameters")
                
        generate_btn = gr.Button(value="Generate")
        with gr.Row(): 
            interact_result = gr.Image(label="InteractDiffusion(+MIGC)", type='pil')
            migc_result_origin = gr.Image(label="MIGC Origin", type='pil')
            interact_result_origin = gr.Image(label="InteractDiffusion Origin", type='pil')
        
        
        boxes_preview_btn.click(
            boxes_preview,
            inputs=[subject_boxes, object_boxes, action_subject, action_object],
            outputs=[boxes_board]
        )
        
        generate_btn.click(
            generate,
            inputs=[caption, negative_caption, subject_boxes, object_boxes, action_subject, action_object, action, omega, total_steps, seed, 
                    optim_start, optim_end, migc_scale_p, interact_scale_p,
                    MIGCsteps, NaiveFuserSteps, ca_scale, ea_scale, sac_scale],
            outputs=[interact_result, migc_result_origin, interact_result_origin]
        )
                
            

    
    UI.launch(server_name='0.0.0.0', server_port=args.port, share=True)
    