import torch
from ldm.modules.diffusionmodules.util import timestep_embedding
import os
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf


def get_state_dict(d):
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location),weights_only=True))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model

class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, **kwargs):
        dtype = self.time_embed[0].weight.dtype

        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(dtype)

        emb = self.time_embed(t_emb)
        # h = x
        h = x.type(dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            if ((i+1)%3 == 0) and control is not None:
                h = h + control.pop(0)
            hs.append(h)
        h = self.middle_block(h, emb, context)
            

        if control is not None:
            h += control.pop(0)

        for i, module in enumerate(self.output_blocks):
            if  control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if ((i+2)%3 == 0) and control is not None:
                h = h + control.pop(0)

        # h = h.type(x.dtype)
        h = h.type(dtype)
        return self.out(h)

class BFRffusion(LatentDiffusion):

    def __init__(self, control_stage_config, control_key,sd_locked_steps,CosineAnnealing_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        # self.control_key = control_key
        # self.sd_locked_steps = sd_locked_steps
        # self.CosineAnnealing_steps = CosineAnnealing_steps
        # self.top5_psnr_dict = {}

    def compile_models(self):
        self.model = torch.compile(self.model)
        self.control_model = torch.compile(self.control_model)
        self.first_stage_model = torch.compile(self.first_stage_model)
        self.cond_stage_model = torch.compile(self.cond_stage_model)


    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        cond_txt = self.cond_stage_model(t)
        # cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None)
        else:
            control = self.control_model(x=x_noisy, hint=torch.cat(cond['c_concat'], 1), timesteps=t, context=cond_txt)
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control)

        return eps
