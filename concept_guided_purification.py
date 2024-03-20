import requests
import torch
from PIL import Image
from io import BytesIO

import argparse
import itertools
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

torch.manual_seed(39)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers import DPMSolverMultistepScheduler



use_liif = ""

for data_name in ["Cosal2015", "iCoseg", "CoCA", "CoSOD3k"]:
    for strength in [0.25]:
        for distortion in ["jadena"]:
            version = "{}_224_mixdataset_ratio0.5_768".format(data_name)            
            dataset_root = "dataset/{}/img_{}{}".format(version, distortion, use_liif)

            learned_concept_root = "textual_models/{}_object/img_{}".format(version,distortion)

            guidance_scale = 7.5
            target_concept_root = "{}_CosalPure".format(dataset_root,strength,guidance_scale)

            import requests
            import glob
            from io import BytesIO

            folders = os.listdir(dataset_root)
            folder_num = len(folders)

            for i in range(folder_num):
                folder_name = folders[i]
                target_concept_path = os.path.join(target_concept_root,folder_name)
                if not os.path.exists(target_concept_path):
                    os.makedirs(target_concept_path)
                dataset_folder = os.path.join(dataset_root,folder_name)
                placeholder_token = "my_{}".format(folder_name)
                model_root = "{}/{}/model".format(learned_concept_root,folder_name)

                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_root,
                    scheduler=DPMSolverMultistepScheduler.from_pretrained(model_root, subfolder="scheduler"),
                    torch_dtype=torch.float16)
                pipe = pipe.to("cuda")

                img_names = os.listdir(dataset_folder)
                len_imgs = len(img_names)

                for img_name in img_names:
                    image_inp = Image.open(os.path.join(dataset_folder, img_name)).convert("RGB")
                    prompt_concept = "{}".format(placeholder_token)
                    images = pipe(prompt=prompt_concept, 
                                image=image_inp, 
                                strength=strength, 
                                guidance_scale=guidance_scale,
                                ).images[0]

                    images.save(os.path.join(target_concept_path, img_name))
                    
