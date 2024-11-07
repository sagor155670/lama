#!/usr/bin/env python3

# Example command:
# ./bin/predict.py \
#       model.path=<path to checkpoint, prepared by make_checkpoint.py> \
#       indir=<path to input data> \
#       outdir=<where to store predicts>


import os
from pathlib import Path 
import sys 

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
import sys
import traceback

import gradio as gr


from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import threading
import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from saicinpainting.training.data.datasets import make_default_val_dataset
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

from moviepy.editor import ImageSequenceClip

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path='/media/mlpc2/workspace/sagor/TestingModels/lama/configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32' and threading.current_thread() is threading.main_thread():
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device("cuda")

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        print(f"train config path: {train_config_path}")
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        out_ext = predict_config.get('out_ext', '.jpg')

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cuda')
        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        if not predict_config.indir.endswith('/'):
            predict_config.indir += '/'

        dataset = make_default_val_dataset(predict_config.indir, **predict_config.dataset)
        
        # w = predict_config.get('w', 1080)
        # h = predict_config.get('h', 1920)
        fps = predict_config.get('fps', 25)
        frames = [] 
        # writer = cv2.VideoWriter("/media/mlpc2/workspace/sagor/TestingModels/lama/output_new.webm", cv2.VideoWriter_fourcc(*"VP90"), fps, (w, h))
        for img_i in tqdm.trange(len(dataset)):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                predict_config.outdir, 
                os.path.splitext(mask_fname[len(predict_config.indir):])[0] + out_ext
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
            batch = default_collate([dataset[img_i]])
            if predict_config.get('refine', True):
                # print("inside Refiner")
                assert 'unpad_to_size' in batch, "Unpadded size is required for the refinement"
                # image unpadding is taken care of in the refiner, so that output image
                # is same size as the input image
                cur_res = refine_predict(batch, model, **predict_config.refiner)
                cur_res = cur_res[0].permute(1,2,0).detach().cpu().numpy()
            else:
                with torch.no_grad():
                    batch = move_to_device(batch, device)
                    batch['mask'] = (batch['mask'] > 0) * 1
                    batch = model(batch)                    
                    cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
                    unpad_to_size = batch.get('unpad_to_size', None)
                    if unpad_to_size is not None:
                        orig_height, orig_width = unpad_to_size
                        cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            frames.append(cur_res)
            # writer.write(cur_res)
            cv2.imwrite(cur_out_fname, cur_res)
            
        
        write_video_from_images(frames,'/media/mlpc2/workspace/sagor/TestingModels/lama/output.webm',fps)
       
        # writer.release()
    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


def write_video_from_images(image_array, output_file, fps=30):

    if len(image_array) == 0:
        raise ValueError("The image array is empty.")
    
    # Get the dimensions of the images
    height, width, _ = image_array[0].shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'VP90')  # or use 'XVID', 'MJPG', etc.
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in image_array:
        # Ensure the image is in BGR color space
        if image.shape[2] != 3:
            raise ValueError("Each image must be in BGR color space with 3 channels.")
        
        video_writer.write(image)

    # Release the video writer object
    video_writer.release()
    print(f"Video saved as {output_file}")

if __name__ == '__main__':
    main()
