import os

FPS = 4
RSL = 6
video_path = '/home/tangchen/dataset_4k/demo.mp4'
ffmpeg_path = '/home/tangchen/ffmpeg/ffmpeg'
result_path = ''
command = [ffmpeg_path, " -i ", video_path, " -r ", "", " -s ", "", " -f image2 /home/tangchen/ffmpeg-opt/image-%4d.jpg"]

def do_ffmpeg(c):
    print("do command: ", c)
    os.system(c)
    return 0

def main():
    rsl = input("please input resolution...")
    print("resolution is ", rsl)
    fps = input("please nput fps...")
    print("fps is ", fps)

    command[RSL] = rsl
    command[FPS] = fps

    do_ffmpeg("".join(command))
    return 0

main()