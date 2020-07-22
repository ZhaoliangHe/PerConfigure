import sys
import os
import subprocess

path = "./dataset/youtube"
video_path = os.path.join(path,"11SecondVideo.mp4")
ffmpeg_path = '/home/ctang/ffmpeg/ffmpeg-4.2.3/ffmpeg'
image_path = os.path.join(path,"11image_720p/image-%3d.png")
command = [ffmpeg_path, "-i", video_path, "", "", "", "", "-f", "image2", image_path]

def do_ffmpeg():
    print("do command: ", command)
    subprocess.call(command)
    return 0

def main():
    command[3:7] = sys.argv[-4:]
    do_ffmpeg()
    return 0

main()