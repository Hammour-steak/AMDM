from interactmigc import InteractMIGC
import json
import os

if __name__ == '__main__':

    model = InteractMIGC(device='cuda:1')

    with open("mig_bench.json", 'r') as file:
        res = json.load(file)
    
    migc_output_dir = 'generation/migc'
    interact_output_dir = 'generation/interactmigc'
    
    for i, r in res.items():
        """
        "0": {
            "caption": "a photo of a black cat and a white cat ", 
            "segment": [{"bbox": [0.127859375, 0.466743648960739, 0.691125, 0.7394688221709007], "label": "a black cat"}, {"bbox": [0.542578125, 0.5063741339491916, 0.8065468750000001, 0.7475750577367205], "label": "a white cat"}], 
            "image_id": "500084"
            }
        """
        img_id = r['image_id']
        caption = r['caption']
        subject_phrases = []
        object_phrases = []
        action_phrases = []
        subject_boxes = []
        object_boxes = []
        res_l = r['segment']
        for e in res_l:
            subject_phrases.append(e['label'])
            subject_boxes.append(e['bbox'])
            action_phrases.append('and')
        object_phrases = subject_phrases
        object_boxes = subject_boxes

        negative_caption = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
        migc_prompt = [[caption] + subject_phrases]
        migc_boxes = [subject_boxes]

        interactdiffusion_subject_phrases = subject_phrases
        interactdiffusion_object_phrases = object_phrases
        interactdiffusion_action_phrases = action_phrases
        interactdiffusion_subject_boxes = subject_boxes
        interactdiffusion_object_boxes = object_boxes

        migc_images, interact_images = model.interactmigc_inference(
            prompt=migc_prompt, bboxes=migc_boxes, caption=caption, negative_prompt=negative_caption,
            interactdiffusion_subject_phrases=interactdiffusion_subject_phrases,
            interactdiffusion_object_phrases=interactdiffusion_object_phrases,
            interactdiffusion_action_phrases=interactdiffusion_action_phrases,
            interactdiffusion_subject_boxes=interactdiffusion_subject_boxes,
            interactdiffusion_object_boxes=interactdiffusion_object_boxes,
            omega=0.5,
            num_inference_steps=50,
            optim_start=1,
            optim_end=3,
            migc_p=55,
            interact_p=45,
            is_mixed=True,
            mixed_type='squared'
        )
        migc_images = migc_images.images
        interact_images = interact_images.images
        
        savename = str(img_id).zfill(6) + '_' + str(i) + '.png'
        migc_images[0].save(os.path.join(migc_output_dir, savename))
        interact_images[0].save(os.path.join(interact_output_dir, savename))