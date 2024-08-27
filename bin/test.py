import os
from pathlib import Path 
import sys 

sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
import traceback

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.refinement import refine_predict
os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
# os.environ['NUMEXPR_NUM_THREADS'] = '1'

import cv2
import hydra
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers

LOGGER = logging.getLogger(__name__)

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()  # kill -10 <pid> will result in traceback dumped into log

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        print(f"train config path: {train_config_path}")
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device)
        
        print(model)

        model.freeze()
        if not predict_config.get('refine', False):
            model.to(device)

        # Load input image and mask
        input_image_path = predict_config.input_image
        mask_image_path = predict_config.mask_image
        output_image_path = predict_config.output_image

        input_image = cv2.imread(input_image_path)
        mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

        # Prepare input image and mask
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        mask_image = (mask_image > 0).astype(np.float32)

        # Add batch dimension and convert to torch tensors
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0)
        mask_image = torch.from_numpy(mask_image).unsqueeze(0).unsqueeze(0)

        batch = {'image': input_image, 'mask': mask_image}

        # Inference
        with torch.no_grad():
            batch = move_to_device(batch, device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = model(batch)
            cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, cur_res)

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()