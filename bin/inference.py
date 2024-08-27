import os
import sys
import logging
import traceback
import cv2
import numpy as np
import torch
import yaml
from pathlib import Path 
import sys 

sys.path.append(str(Path(__file__).resolve().parent.parent))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)

# class ConfigWrapper:
#     def __init__(self, config_dict):
#         self.__dict__.update(config_dict)
        
#     def __getattr__(self, item):
#         return self.__dict__.get(item)

def load_train_config(train_config_path):
    with open(train_config_path, 'r') as f:
        train_config = yaml.safe_load(f)
    train_config['training_model']['predict_only'] = True
    train_config['visualizer']['kind'] = 'noop'
    return train_config

def main(model_path = '/media/mlpc2/workspace/sagor/TestingModels/lama/big-lama', checkpoint = '/media/mlpc2/workspace/sagor/TestingModels/lama/big-lama/models/best.ckpt', input_image_path = '/media/mlpc2/workspace/sagor/TestingModels/lama/testImages/image126.png', mask_image_path = '/media/mlpc2/workspace/sagor/TestingModels/lama/testImages/image126_mask.png', output_image_path = '/media/mlpc2/workspace/sagor/TestingModels/lama/result_new.png', out_key='inpainted'):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_config_path = os.path.join(model_path, 'config.yaml')
        print(f"train config path: {train_config_path}")
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(model_path, 'models', checkpoint)
        model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location=device)
        
        print(model)

        model.freeze()
        model.to(device)

        # Load input image and mask
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
            cur_res = batch[out_key][0].permute(1, 2, 0).detach().cpu().numpy()

        cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, cur_res)
        return cur_res

    except KeyboardInterrupt:
        LOGGER.warning('Interrupted by user')
    except Exception as ex:
        LOGGER.critical(f'Prediction failed due to {ex}:\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    # if len(sys.argv) != 6:
    #     print("Usage: python script.py <model_path> <checkpoint> <input_image> <mask_image> <output_image>")
    #     sys.exit(1)

    # model_path = sys.argv[1]
    # checkpoint = sys.argv[2]
    # input_image_path = sys.argv[3]
    # mask_image_path = sys.argv[4]
    # output_image_path = sys.argv[5]

    main()