import os
import argparse
from subprocess import call

# you can use
# python ffmpeg_code.py -h #查看可传入参数
# python ffmpeg_code.py -r 15 -s 1024x1024 640x640 320x320
# //剪切视频
# ffmpeg -ss 0:0:2 -t 0:0:10 -i demo1.mp4 -vcodec copy -acodec copy demo1_10s.mp4
# // -ss 开始时间; -t 持续时间
ffmpeg_path = "/home/hezhaoliang/ffmpeg/ffmpeg-4.2.3/ffmpeg" # instead of yours

def videotoimages(fps,resolution,ffmpeg_path):
    path = "./dataset/youtube/"
    # images_file = '11images' + '_' + fps + '_' + resolution
    images_file = 'demo1_10s' + '_' + fps + '_' + resolution
    image_path = path + images_file + '/image-%d.jpg'
    # input_video = "11SecondVideo.mp4"
    input_video = "demo1_10s.mp4"
    call(["mkdir", path + images_file])
    call([ffmpeg_path, "-i", os.path.join(path, input_video), "-r", fps, "-s", resolution, "-f", "image2",
          image_path])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ffmpeg')
    parser.add_argument("-r", help="fps", default="15", type=str)
    parser.add_argument("-s", help="resolution", default="640x640", type=str)
    args = parser.parse_args()
    fps = args.r
    resolution = args.s
    videotoimages(fps,resolution,ffmpeg_path)

