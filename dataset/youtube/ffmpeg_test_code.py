import os
from subprocess import call
# import ffmpeg # 网上说也可以使用python-ffmpeg包，不知道好用不
ffmpeg_path = "/home/hezhaoliang/ffmpeg/ffmpeg-4.2.3/ffmpeg"

os.system('mkdir 11image_720p')
# system方法和call方法都可以
os.system(ffmpeg_path+" -i 11SecondVideo.mp4 -r 15 -s 1280x720 -f image2 ./11image_720p/image-%3d.png")
# call([ffmpeg_path,"-i","11SecondVideo.mp4","-r","15","-s","1280x720","-f","image2","./11image_720p/image-%3d.png"])

# 在pycharm不用全路径的ffmpeg会报错，但在服务器直接python ffmpeg_test_code.py却不报错
# os.system("ffmpeg -i 11SecondVideo.mp4 -r 15 -s 1280x720 -f image2 ./11image_720p/image-%3d.png")
# call(["ffmpeg","-i","11SecondVideo.mp4","-r","15","-s","1280x720","-f","image2","./11image_720p/image-%3d.png"])

# linux命令 ffmpeg -i 11SecondVideo.mp4 -r 15 -s 1280x720 -f image2 ./11image_720p/image-%3d.png