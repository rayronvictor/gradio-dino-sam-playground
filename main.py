import random
import argparse

import gradio as gr
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# diffusers
from diffusers import StableDiffusionInpaintPipeline

config_file = 'config/GroundingDINO_SwinT_OGC.py'
ckpt_repo_id = "ShilongLiu/GroundingDINO"
# baixar em https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
ckpt_filenmae = "checkpoints/groundingdino_swint_ogc.pth"
# baixar em https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
sam_checkpoint = 'checkpoints/sam_vit_h_4b8939.pth'
output_dir = "outputs"

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', help='Usar CPU ao invés de GPU')
args = parser.parse_args()

device = "cpu" if args.cpu else "cuda"

groundingdino_model = None
sam_predictor = None
inpaint_pipeline = None


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    full_img = None

    # for ann in sorted_anns:
    for i in range(len(sorted_anns)):
        ann = anns[i]
        m = ann['segmentation']
        if full_img is None:
            full_img = np.zeros((m.shape[0], m.shape[1], 3))
            map = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint16)
        map[m != 0] = i + 1
        color_mask = np.random.random((1, 3)).tolist()[0]
        full_img[m != 0] = color_mask
    full_img = full_img*255
    # anno encoding from https://github.com/LUSSeg/ImageNet-S
    res = np.zeros((map.shape[0], map.shape[1], 3))
    res[:, :, 0] = map % 256
    res[:, :, 1] = map // 256
    res.astype(np.float32)
    full_img = Image.fromarray(np.uint8(full_img))
    return full_img, res


def transform_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    scores = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        scores.append(logit.max().item())

    return boxes_filt, torch.Tensor(scores), pred_phrases


def draw_mask(mask, draw, random_color=False):
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
    else:
        color = (30, 144, 255, 153)

    nonzero_coords = np.transpose(np.nonzero(mask))

    for coord in nonzero_coords:
        draw.point(coord[::-1], fill=color)


def draw_box(box, draw, label):
    # random color
    color = tuple(np.random.randint(0, 255, size=3).tolist())

    draw.rectangle(((box[0], box[1]), (box[2], box[3])), outline=color,  width=2)

    if label:
        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((box[0], box[1]), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (box[0], box[1], w + box[0], box[1] + h)
        draw.rectangle(bbox, fill=color)
        draw.text((box[0], box[1]), str(label), fill="white")

        draw.text((box[0], box[1]), label)

def run_sam(image, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode, scribble_mode):
    global groundingdino_model, sam_predictor, sam_automask_generator, inpaint_pipeline

    size = image.size

    if sam_predictor is None:
        # initialize SAM
        assert sam_checkpoint, 'sam_checkpoint is not found!'
        sam = build_sam(checkpoint=sam_checkpoint)
        sam.to(device=device)
        sam_predictor = SamPredictor(sam)
        sam_automask_generator = SamAutomaticMaskGenerator(sam)

    if groundingdino_model is None:
        groundingdino_model = load_model(config_file, ckpt_filenmae, device=device)

    image_pil = image.convert("RGB")
    image = np.array(image_pil)

    if task_type == 'automask':
        masks = sam_automask_generator.generate(image)
    else:
        transformed_image = transform_image(image_pil)

        # run grounding dino model
        boxes_filt, scores, pred_phrases = get_grounding_output(
            groundingdino_model, transformed_image, text_prompt, box_threshold, text_threshold
        )

        # process boxes
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()

        if task_type == 'seg' or task_type == 'inpainting':
            sam_predictor.set_image(image)

            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)

            masks, _, _ = sam_predictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = transformed_boxes,
                multimask_output = False,
            )

    if task_type == 'det':
        image_draw = ImageDraw.Draw(image_pil)
        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, image_draw, label)

        return [image_pil]
    elif task_type == 'automask':
        full_img, res = show_anns(masks)
        return [full_img]
    elif task_type == 'seg':

        mask_image = Image.new('RGBA', size, color=(0, 0, 0, 0))

        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

        image_draw = ImageDraw.Draw(image_pil)

        for box, label in zip(boxes_filt, pred_phrases):
            draw_box(box, image_draw, label)

        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return [image_pil, mask_image]
    elif task_type == 'inpainting':
        assert inpaint_prompt, 'inpaint_prompt is not found!'
        # inpainting pipeline
        if inpaint_mode == 'merge':
            masks = torch.sum(masks, dim=0).unsqueeze(0)
            masks = torch.where(masks > 0, True, False)
        mask = masks[0][0].cpu().numpy()  # simply choose the first mask, which will be refine in the future release
        mask_pil = Image.fromarray(mask)

        if inpaint_pipeline is None:
            inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16
            )
            inpaint_pipeline = inpaint_pipeline.to(device)

        image = inpaint_pipeline(prompt=inpaint_prompt, image=image_pil.resize((512, 512)),
                                 mask_image=mask_pil.resize((512, 512))).images[0]
        image = image.resize(size)

        return [image, mask_pil]
    else:
        print("task_type:{} error!".format(task_type))

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Imagem", sources='upload', type="pil", value="assets/vitiligo_example01.png")
            # task_type = gr.Dropdown(["scribble", "automask", "det", "seg", "inpainting", "automatic"],
            #                         value="automatic", label="task_type")
            task_type = gr.Dropdown(["scribble", "automask", "det", "seg", "inpainting"], value="automask", label="task_type")
            text_prompt = gr.Textbox(label="Text Prompt")
            inpaint_prompt = gr.Textbox(label="Inpaint Prompt")
            run_button = gr.Button(value="Run")
            with gr.Accordion("Advanced options", open=False):
                box_threshold = gr.Slider(
                    label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.05
                )
                text_threshold = gr.Slider(
                    label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.05
                )
                iou_threshold = gr.Slider(
                    label="IOU Threshold", minimum=0.0, maximum=1.0, value=0.5, step=0.05
                )
                inpaint_mode = gr.Dropdown(["merge", "first"], value="merge", label="inpaint_mode")
                scribble_mode = gr.Dropdown(["merge", "split"], value="split", label="scribble_mode")

        with gr.Column():
            image_gallery = gr.Gallery(label="Resultado")
            # output_image = gr.Image(label="Resultado")

    run_button.click(run_sam, inputs=[
        input_image, text_prompt, task_type, inpaint_prompt, box_threshold, text_threshold, iou_threshold, inpaint_mode, scribble_mode], outputs=image_gallery)

if __name__ == '__main__':
    demo.queue()
    demo.launch(server_name='0.0.0.0', debug=True, inbrowser=True)
