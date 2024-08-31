import time
import os
import numpy as np
import uuid
import cv2
import tqdm
import shutil
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel

def merge_audio_video(video_path, audio_path, output_video_name):
    print(f"Video path is set to: {video_path}")
    print(f"Audio path is set to: {audio_path}")
    print(f"Output video name is set to: {output_video_name}")

    audioModel = AudioModel()
    audioModel.loadModel("checkpoint/audio.pkl")

    renderModel = RenderModel()
    renderModel.loadModel("checkpoint/render.pth")
    pkl_path = os.path.join(video_path, "keypoint_rotate.pkl")
    video_file_path = os.path.join(video_path, "circle.mp4")
    renderModel.reset_charactor(video_file_path, pkl_path)

    wavpath = audio_path
    mouth_frame = audioModel.interface_wav(wavpath)
    cap_input = cv2.VideoCapture(video_file_path)
    vid_width = cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
    vid_height = cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
    cap_input.release()

    task_id = str(uuid.uuid1())
    os.makedirs(f"output/{task_id}", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = f"output/{task_id}/silence.mp4"
    videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width), int(vid_height)))
    
    for frame in tqdm.tqdm(mouth_frame):
        frame = renderModel.interface(frame)
        videoWriter.write(frame)

    videoWriter.release()
    
    final_video_path = f"../output/{output_video_name}.mp4"
    os.system(f"ffmpeg -i {save_path} -i {wavpath} -c:v libx264 -pix_fmt yuv420p -loglevel quiet {final_video_path}")
    shutil.rmtree(f"output/{task_id}")

    return final_video_path

if __name__ == "__main__":
    video_path = "path_to_video"
    audio_path = "path_to_audio"
    output_video_name = "output_video_name"
    merge_audio_video(video_path, audio_path, output_video_name)

