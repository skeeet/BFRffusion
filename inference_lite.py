import os
import torch
import numpy as np
from omegaconf import OmegaConf
from argparse import ArgumentParser
import cv2
from models.models import load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

import pathlib
from tqdm import tqdm
import random


def img2tensor(img, bgr2rgb=True, float32=True):
    if img.shape[2] == 3 and bgr2rgb:
        if img.dtype == 'float64':
            img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    if float32:
        img = img.float()
    return img


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output

@torch.no_grad()
def main(opt):
    # ------------------------ input & output ------------------------
    inpath = pathlib.Path(opt.input)
    img_list = sorted([str(x) for x in inpath.glob('**/*.[jJpP][pPnN]*[gG]')])

    base_outpath = pathlib.Path(opt.output)
    base_outpath.mkdir(exist_ok=True, mode=0o777, parents=True)

    # initialize model
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    target_dtype = torch.float32
    if opt.half:
        target_dtype = torch.half

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model).to(device)
    model.load_state_dict(load_state_dict(opt.ckpt), strict=True)

    if opt.half:
        model = model.to(target_dtype)
        model.control_model.dtype = torch.float16
        model.control_model = model.control_model.to(torch.float16)

    ddim_sampler = DDIMSampler(model)
    # if opt.half:
    #     for bname, buff in ddim_sampler.named_buffers:
    #         pass
    #     ddim_sampler = ddim_sampler.to(target_dtype)

    H = W = opt.image_size
    shape = (4, H // 8, W // 8)


    if opt.compile:
        model.compile_models()

    # ------------------------ restore ------------------------
    for img_path in tqdm(img_list, dynamic_ncols=True):
        # read image
        img_name = os.path.basename(img_path)
        # print(f'Processing {img_name} ...')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f'Failed to read {img_name}')
            continue

        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        # prepare data
        cropped_face_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        if opt.half:
            cropped_face_t = cropped_face_t.to(target_dtype)

        # face restoration
        cond = {"c_concat": [cropped_face_t], "c_crossattn": []}
        samples, _ = ddim_sampler.sample(opt.ddim_steps, 1, shape, cond, verbose=False)
        hq_imgs = model.decode_first_stage(samples)

        # convert to image
        restored_face = tensor2img_fast(hq_imgs.float(), rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype('uint8')

        savepath = base_outpath.joinpath(pathlib.Path(img_path).relative_to(inpath))
        savepath.parent.mkdir(exist_ok=True, parents=True, mode=0o777)

        cv2.imwrite(str(savepath), restored_face)

    print(f'Results are in the [{opt.output}] folder.')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input image folder')
    parser.add_argument('--output', type=str, required=True, help='Output folder')
    parser.add_argument("--ckpt", type=str, default='experiments/weights/checkpoint_BFRffusion_FFHQ.ckpt', help='dir of ckpt to load')
    parser.add_argument("--config", type=str, default="options/test.yaml", help="path to config which constructs model")
    parser.add_argument("--ddim_steps", type=int, default=50, help="number of ddpm sampling steps")
    parser.add_argument("--image_size", type=int, default=512, help='Image size as the model input.')
    parser.add_argument("--seed", type=int, default=132456)
    parser.add_argument("--compile", action="store_true", help='Use torch compiler for models')
    parser.add_argument("--half", action="store_true", help='Use f16 precision')
    opt = parser.parse_args()

    main(opt)
