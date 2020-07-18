import os
from subprocess import call
path = "./dataset/youtube"

ffmpeg_path = "/home/hezhaoliang/ffmpeg/ffmpeg-4.2.3/ffmpeg"
image_path = os.path.join(path,"11image_720p/image-%3d.png")
resolution = "1280x720"
os.system('mkdir ./dataset/youtube/11image_720p')
call([ffmpeg_path,"-i",os.path.join(path,"11SecondVideo.mp4"),"-r","15","-s",resolution,"-f","image2",image_path])
