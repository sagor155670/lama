from collections import defaultdict
import time,re,math
import cv2,numpy as np
import torch
from ultralytics import YOLO,SAM
from ultralytics.utils.plotting import Annotator, colors
from shapely.geometry import Point, Polygon
import gradio as gr
from pathlib import Path 
import sys ,os,glob
import logging
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parent.parent))

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.training.trainers import load_checkpoint
import yaml
import os
from omegaconf import OmegaConf
# from bin.inference import main
from bin.testWithRefiner import main
LOGGER = logging.getLogger(__name__)

root = '/media/mlpc2/workspace/sagor/TestingModels/lama/'
testImageDir = os.path.join(root,'testImages')
outputDir = os.path.join(root,'output')
trackedFrameDir = os.path.join(root,'TrackedFrames')
defaultConfigPath = os.path.join(root,'configs/prediction/default.yaml')

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier


def get_object_id(seg_boxes,track_ids, event: gr.SelectData, video):
    selectPoint = Point(event.index[0],event.index[1])
    for mask, track_id in zip(seg_boxes, track_ids):
        polygon = Polygon(mask)
        is_inside = polygon.contains(selectPoint)
        if is_inside:
            return videoTracker(video,track_id)    

def getFirstTrackedFrame(video):
    model = YOLO("yolov8x-seg.pt")  # segmentation model
    # model = SAM('sam2_b.pt')
    cap = cv2.VideoCapture(video)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    ret, firstFrame = cap.read()
    cap.release()
    annotator = Annotator(firstFrame, line_width=2)
    results = model.track(firstFrame, persist=True)
    if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            im_gpu = torch.from_numpy(firstFrame).permute(2,0,1)
            for mask, track_id in zip(masks, track_ids):
                color = colors(int(track_id), True)
                txt_color = annotator.get_txt_color(color)
                maskPoly = Polygon(mask)
                expanded_mask  = maskPoly.buffer(10.0)
                maskNumpy = np.array(expanded_mask.exterior.coords, dtype=np.int32)
                annotator.seg_bbox(mask=maskNumpy, mask_color=color, label=str(track_id), txt_color=txt_color)
    seg_frame = "seg_frame.jpg"
    cv2.imwrite(seg_frame, firstFrame)
    return gr.Image(value=seg_frame, visible = True), masks, track_ids

def getAllTrackedFrames(video,track_id) :
    
    if os.path.exists(trackedFrameDir):
        delete_all_files(trackedFrameDir)
    # track_history = defaultdict(lambda: [])
    frames = []

    model = YOLO("yolov8x-seg.pt")  # segmentation model
    # lama_model = load_lama_model() #inpaint model
    # model = SAM('sam2_b.pt')
    cap = cv2.VideoCapture(video)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    t_unit = truncate_float(1 / 25 , 2)
    t = t_unit
    # out = cv2.VideoWriter("output.webm", cv2.VideoWriter_fourcc(*"VP90"), fps, (w, h))

    setFrameRate(fps if fps <= 25 else 25,defaultConfigPath)

    i = 0
    while True:
        i += 1 
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        skip_interval = math.ceil(fps) / 25
        tolerence = 1e-9     

        cur_time = truncate_float(i / fps ,2)
        if cur_time < t:
            continue

        # cv2.imwrite(f"/media/mlpc2/workspace/sagor/TestingModels/lama/testImages/image{i}.png",im0)
        annotator = Annotator(im0, line_width=2)

        results = model.track(im0, persist=True)
        blackMask = np.zeros(im0.shape[:2], dtype=np.uint8)   
        print(f"shape: {im0.shape[:2]}")     
        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            im_gpu = torch.from_numpy(im0).permute(2,0,1)
            for mask, id in zip(masks, track_ids):
                if id == track_id:
                    print(type(mask))
                    polygon = np.array(mask , np.int32)
                    color = colors(int(id), True)
                    txt_color = annotator.get_txt_color(color)
                    #expansion
                    maskPoly = Polygon(mask)
                    expanded_mask  = maskPoly.buffer(10.0)
                    maskNumpy = np.array(expanded_mask.exterior.coords, dtype=np.int32)

                    annotator.seg_bbox(mask=maskNumpy, mask_color=color, label=str(id), txt_color=txt_color)
                    # cv2.fillPoly(blackMask,[maskNumpy], 255)
            cv2.imwrite(f'{root}/TrackedFrames/image{i}_mask.jpg', im0)
            
        mask_img = blackMask 
        t = truncate_float(t + t_unit , 2)            

def onClear(input_video):
    return gr.Image( visible = False ) 


def videoTracker(video,track_id) :

    if os.path.exists(testImageDir):
        delete_all_files(testImageDir)
    if os.path.exists(outputDir):
        delete_all_files(outputDir)
    # track_history = defaultdict(lambda: [])
    frames = []

    model = YOLO("yolov8x-seg.pt")  # segmentation model
    # lama_model = load_lama_model() #inpaint model
    # model = SAM('sam2_b.pt')
    cap = cv2.VideoCapture(video)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    t_unit = truncate_float(1 / 25 , 2)
    t = t_unit
        
    # out = cv2.VideoWriter("output.webm", cv2.VideoWriter_fourcc(*"VP90"), fps, (w, h))

    setFrameRate(fps if fps <= 25 else 25 , defaultConfigPath)

    i = 0
    while True:
        i += 1 
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break
        
        # if fps > 25:         
        #     if i%skip_interval < tolerence:
        #         continue
        # print(f"i = {i}, i%skip: {i%skip_interval}")
        cur_time = truncate_float(i / fps ,2)
        if cur_time < t:
            continue


        cv2.imwrite(f"{root}/testImages/image{i}.jpg",im0)
        annotator = Annotator(im0, line_width=2)

        results = model.track(im0, persist=True)
        blackMask = np.zeros(im0.shape[:2], dtype=np.uint8)   
        print(f"shape: {im0.shape[:2]}")     
        if results[0].boxes.id is not None and results[0].masks is not None:
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()
            im_gpu = torch.from_numpy(im0).permute(2,0,1)
            for mask, id in zip(masks, track_ids):
                if id == track_id:
                    print(type(mask))
                    polygon = np.array(mask , np.int32)
                    color = colors(int(id), True)
                    txt_color = annotator.get_txt_color(color)
                    #expansion
                    maskPoly = Polygon(mask)
                    expanded_mask  = maskPoly.buffer(10.0)
                    maskNumpy = np.array(expanded_mask.exterior.coords, dtype=np.int32)

                    annotator.seg_bbox(mask=maskNumpy, mask_color=color, label=str(id), txt_color=txt_color)
                    cv2.fillPoly(blackMask,[maskNumpy], 255)
            cv2.imwrite(f'{root}/testImages/image{i}_mask.jpg', blackMask)
            
        mask_img = blackMask 
        t = truncate_float(t + t_unit , 2)    
        
    
    main()
    
    # write_video_from_images(frames,"/media/mlpc2/workspace/sagor/TestingModels/lama/output.webm",fps)
    # frames = create_video_from_images("/media/mlpc2/workspace/sagor/TestingModels/lama/output",'/media/mlpc2/workspace/sagor/TestingModels/lama/output.mp4',fps=fps)
    # out.release()
    cap.release()
    getAllTrackedFrames(video,track_id)

    frames = getFrames(output_path=outputDir)
    masks = getFrames(output_path= trackedFrameDir)

    return gr.Video(f"{root}/output.webm"), gr.Gallery(frames),gr.Gallery(masks),
# cv2.destroyAllWindows()




def getFrames(output_path):
    return sorted(glob.glob(os.path.join(output_path, '*mask*.jpg'), recursive=True),key= extract_number)


def extract_number(path):
    # Extract number from the filename using regex
    match = re.search(r'(\d+)_mask\.jpg$', os.path.basename(path))
    if match:
        return int(match.group(1))
    return 0  # Default to 0 if no number is found

def setFrameRate(fps,path: str):
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
        print(f"fps= {fps}")
        data["fps"] = fps
    with open(path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)    
    

def create_video_from_image(image_folder, output_video_path, fps=30):
    # Get the list of image files
    images = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()  # Sort images if necessary

    if not images:
        raise ValueError("No images found in the specified folder.")

    # Create a video clip from the images
    clip = ImageSequenceClip(images, fps=fps)

    # Write the video to a file
    clip.write_videofile(output_video_path, codec='libx264', threads=4)
    print(f"Video saved as {output_video_path}")
    return images

def delete_all_files(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path {folder_path} is not a directory or does not exist.")
    
    # Construct a pattern to match all files
    pattern = os.path.join(folder_path, '*')
    
    # Get all files in the folder
    files = glob.glob(pattern)
    
    for file in files:
        try:
            if os.path.isfile(file):
                os.remove(file)
                print(f"Deleted file: {file}")
            else:
                print(f"Skipped non-file: {file}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_video = gr.Video()
            input_img = gr.Image(visible=False,label="Input")
        with gr.Column():
            output_video = gr.Video()
            out_gallery = gr.Gallery()   
            out_mask = gr.Gallery()

    masks = gr.State()
    track_ids = gr.State()
    input_video.change(getFirstTrackedFrame,input_video, [input_img,masks,track_ids])
    input_img.select(get_object_id,[masks,track_ids,input_video] , [output_video,out_gallery,out_mask])
    input_video.clear(onClear,input_video, input_img)

demo.launch(share= True)
