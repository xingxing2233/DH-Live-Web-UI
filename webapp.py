import time
import gradio as gr
import os
import cv2
import uuid
import shutil
import subprocess
import sys
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel
from scipy.io import wavfile
import sounddevice as sd
from data_preparation import CirculateVideo, ExtractFromVideo
from demo import merge_audio_video
import pickle
import mediapipe as mp
import numpy as np

from scipy.io import wavfile
import requests

import base64
import io
import pygame
import edge_tts
import asyncio
from io import BytesIO
import tqdm
from datetime import datetime


mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection



def display_selected_folder(folder):
    return f"你选择了: {folder}"


def process_video(video_path):
    try:
        pts_3d = ExtractFromVideo(video_path)
        if isinstance(pts_3d, np.ndarray) and pts_3d.ndim == 3:
            result_message = "人脸点位生成成功！"
        elif pts_3d == -1:
            result_message = "第一帧人脸检测异常，请检查视频。"
            return result_message
        elif pts_3d == -2:
            result_message = "人脸区域变化幅度太大，请检查视频。"
            return result_message
        else:
            result_message = "检查点生成失败，可能是人脸检测异常或其他问题。"
            return result_message

        # 使用固定的文件夹名
        folder_name = "circle"

        # 如果文件夹已存在，则先删除
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        
        # 创建文件夹
        os.makedirs(folder_name)
        
        # 保存检查点文件
        checkpoint_path = os.path.join(folder_name, "keypoint_rotate.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(pts_3d, f)

        # 移动视频文件到新文件夹
        new_video_path = os.path.join(folder_name, "circle.mp4")
        shutil.move(video_path, new_video_path)

        print("folder_name：", folder_name)
        print("new_video_path：", new_video_path)

        #result_message += f"'{folder_name}'"
        return result_message, folder_name, new_video_path, checkpoint_path, new_video_path
    except Exception as e:
        return f"处理视频时出错：{str(e)}"





def convert_audio_format(audio_path):
    rate, wav = wavfile.read(audio_path, mmap=False)
    converted_audio_path = audio_path.replace(".wav", "_converted.wav")
    wavfile.write(converted_audio_path, rate, wav)
    return f"{converted_audio_path} 转换成功！"





# 合成音频和视频的函数video_file_path
def merge_audio_video(folder_name, audio_path, wav_path, pkl_path, video_file_path, output_video_name):
    try:
        # 将音频文件从临时目录移动到指定的根目录文件夹
        
        
        # 如果没有传入音频路径，使用默认路径
        #if audio_path is None:
            #audio_path = "2bj.wav"  # 替换为你默认的音频路径
            #audio_path = f"{new_audio_path}.wav"

        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            return f"音频文件路径错误，未找到音频文件：{audio_path}", None, None

        # 使用唯一的输出视频文件名
        task_id = str(uuid.uuid1())
        unique_output_video_name = f"{task_id}.mp4"
        output_video_name = os.path.join("output", unique_output_video_name)
        print(f"output video name is set to: {output_video_name}")

        # 加载音频和渲染模型
        audioModel = AudioModel()
        audioModel.loadModel("checkpoint/audio.pkl")

        renderModel = RenderModel()
        renderModel.loadModel("checkpoint/render.pth")

        # 初始化路径
        pkl_path = os.path.join(folder_name, "keypoint_rotate.pkl")
        video_file_path = os.path.join(folder_name, "circle.mp4")
        renderModel.reset_charactor(video_file_path, pkl_path)

        # 处理音频帧并生成视频
        wavpath = audio_path
        mouth_frame = audioModel.interface_wav(wavpath)
        cap_input = cv2.VideoCapture(video_file_path)
        vid_width = cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)
        vid_height = cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap_input.release()

        os.makedirs(f"output/{task_id}", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        save_path = f"output/{task_id}/silence.mp4"
        videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width), int(vid_height)))

        for frame in tqdm.tqdm(mouth_frame):
            frame = renderModel.interface(frame)
            videoWriter.write(frame)

        videoWriter.release()

        # 使用 ffmpeg 合并音频和视频并自动覆盖输出文件
        output_video_path = os.path.join("output", unique_output_video_name)
        os.system(
            f"ffmpeg -y -i {save_path} -i {wavpath} -c:v libx264 -pix_fmt yuv420p -loglevel quiet {output_video_path}"

           

            #f"ffmpeg -y -i {save_path} -i {wavpath} -c:v libx264 -pix_fmt yuv420p -af 'async=1' -loglevel quiet {output_video_name}"



           
        )
        shutil.rmtree(f"output/{task_id}")

        return f"数字人生成成功！", output_video_path   #输出文件：{output_video_path}

    except Exception as e:
        return f"合成视频时出错：{str(e)}", None, None






# 定义文本转语音函数
async def convert_audio_format(text, voice):
    try:
        # 使用 edge_tts 将文本转换为音频
        communicate = edge_tts.Communicate(text, voice)

        # 存储音频数据
        audio_data = bytearray()
        async for message in communicate.stream():
            if message["type"] == "audio":
                audio_data.extend(message["data"])

        # 将音频数据保存为 WAV 文件
        audio_file_path = "output_audio.wav"
        with open(audio_file_path, "wb") as f:
            f.write(audio_data)

        # 返回成功消息和音频文件路径
        return "文本转语音成功！", audio_file_path
    except Exception as e:
        return str(e), None

# Gradio 接口
def tts_interface(text, voice_selector):
    # 调用异步函数并等待结果
    result = asyncio.run(convert_audio_format(text, voice_selector))
    return result


    

def play_audio(audio_path):
    return audio_path




def save_audio_locally(wav_upload):
    # 将音频文件从临时目录移动到指定的根目录文件夹
    audio_folder = "audio_files"  # 你希望保存音频文件的根目录文件夹
    os.makedirs(audio_folder, exist_ok=True)
    audio_filename = os.path.basename(wav_upload)
    new_audio_path = os.path.join(audio_folder, audio_filename)
    shutil.move(wav_upload, new_audio_path)
    return new_audio_path

    





# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("<h1 style='text-align: center;'>数字人生成工具</h1>")
    gr.Markdown("第一步")
    with gr.Row():#视频处理
        
        # 打开视频并展示预览
        video_upload_for_slicing = gr.Video(label="上传视频", include_audio=True, sources=["upload", "webcam"])
        video_output = gr.Textbox(label="人脸点位生成结果")
        process_video_btn = gr.Button("生成人脸点位文件", variant="primary")

        folder_name = gr.State()  # 用于存储检查点生成路径
        pkl_path = gr.State()
        video_file_path = gr.State()

        # 在点击按钮时，获取视频处理结果并保存检查点生成路径
        def process_and_store_video(video_path):
            # 处理视频并保存路径
            result_message, folder_name_value, pkl_path_value, video_file_value = process_video(video_path)
            return result_message, folder_name_value, pkl_path_value, video_file_value

        process_video_btn.click(
            fn=process_video,
            inputs=video_upload_for_slicing,
            #outputs=video_output
            outputs=[video_output, folder_name, pkl_path, video_file_path]
        )
    gr.Markdown("第二步")
    with gr.Row():
    # 音频处理部分

         # 定义发音人选项
        voices = [
            "zh-CN-XiaoxiaoNeural",
            "zh-CN-YunxiNeural",
            "zh-CN-YunjianNeural",
            "zh-CN-YunyangNeural",
            "zh-CN-shaanxi-XiaoniNeural",
            "zh-HK-WanLungNeural",
            "zh-CN-liaoning-XiaobeiNeural",
            "zh-TW-HsiaoYuNeural"
            
        ]

        
        wav_path_input = gr.Textbox(label="输入文本")
        voice_selector = gr.Dropdown(label="选择发音人", choices=voices, value=voices[0])

        convert_btn = gr.Button("文本到语音转换", variant="primary")
        convert_result = gr.Textbox(label="转换结果")

        # 音频文件加载和播放窗口
        audio_playback = gr.Audio(label="音频播放窗口")
        #vc_output2 = gr.Audio(label="Output Audio", interactive=False)


        # 绑定转换按钮点击事件
        convert_btn.click(
           
            fn=tts_interface, 
            inputs=[wav_path_input, voice_selector],
            outputs=[convert_result, audio_playback]  # 输出转换结果文本和音频文件路径

        )

        # 绑定音频文件加载和播放事件
        wav_path_input.change(
            fn=play_audio,
            inputs=wav_path_input,
            outputs=audio_playback
        )


                 
    gr.Markdown("第三步")
    with gr.Row():

     
        # 音频文件上传并播放
        wav_upload = gr.Audio(label="上传合成音频文件", type="filepath")
        checkpoint_path_output = gr.Textbox(label="数字人生成结果")
      
        process_btn = gr.Button("生成数字人", variant="primary")
        output_folder_list = gr.Video(label="生成的数字人", width=720, height=480)

        

        # 在视频合成按钮点击时，将生成的视频路径和检查点路径关联显示
        def merge_and_display_checkpoint(folder_name, audio_path, wav_upload, pkl_path, video_file_path):
            audio_path = save_audio_locally(wav_upload)
            # 调用封装的merge_audio_video函数
            output_video_path = merge_audio_video(folder_name, audio_path, wav_upload, pkl_path, video_file_path)
            return folder_name, output_video_path
        

        process_btn.click(
            fn=merge_audio_video,
            inputs=[folder_name, wav_upload, pkl_path, video_file_path],
            
            outputs=[checkpoint_path_output, output_folder_list]
        )
        


        
    
    gr.Markdown("<h1 style='text-align: center;'>麦克风实时驱动数字人</h1>")
   

  
                
     
           

    # Display the uploaded video
    video_upload_for_slicing.change(
        fn=lambda file_path: file_path if file_path.endswith(('.mp4', '.avi')) else None,
        inputs=video_upload_for_slicing,
        outputs=video_upload_for_slicing
    )

demo.launch()


