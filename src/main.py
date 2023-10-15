import cv2
import numpy as np
import os
import json
from progress.bar import IncrementalBar

input_video_path = input("Введите путь к входному видеофайлу: ")

if not os.path.exists(input_video_path):
    print(f"Файл '{input_video_path}' не существует.")
    exit(1)
else:
    if not input_video_path.lower().endswith(".mp4"):
        print("Формат видео должен быть .mp4")
        exit(1)

output_video_path = input("Введите путь к выходному видеофайлу: ")

if not output_video_path.lower().endswith(".mp4"):
    print("Формат видео должен быть .mp4")
    exit(1)

# input_video_path = "../data/input_task.mp4"
# # input_video_path = "../data/in.mp4"
# output_video_path = "../output/out1.mp4"


print("Сбор метаданных видео...")
os.system(f"ffprobe -loglevel quiet -show_streams -show_format \
          -print_format json {input_video_path} > ../output/meta.txt")
with open("../output/meta.txt", "r", encoding="utf-8") as file:
    meta = json.loads(file.read())
    # print(meta)
os.remove("../output/meta.txt")


dur = int(meta["format"]["duration"].split(".")[0])
# print(meta["format"]["duration"].split("."))
frmrt = meta["streams"][0]["r_frame_rate"].split("/")
# print(frmrt)
framerate = int(frmrt[1]) / int(frmrt[0])
print(f"Продолжительность: {dur}s\nЧастота кадров: {round(1/framerate,3)}")


print("Идет обработка видео...")
load_bar = IncrementalBar(max=(dur))
load_bar.start()
fgbg = cv2.createBackgroundSubtractorMOG2()
min_contour_area = 800

start_time = None
end_time = None
intervals = []


os.system(f"ffmpeg -accurate_seek -ss 1 -i {input_video_path} \
          -frames:v 1 ../output/frame.bmp >/dev/null 2>&1")
image = cv2.imread('../output/frame.bmp')
fgmask = fgbg.apply(image)
os.remove("../output/frame.bmp")
load_bar.next()

for i in range(2, dur+1):
    os.system(f"ffmpeg -accurate_seek -ss {i} -i {input_video_path} \
          -frames:v 1 ../output/frame.bmp >/dev/null 2>&1")
    image = cv2.imread('../output/frame.bmp')
    fgmask = fgbg.apply(image)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    person_in_frame = any(cv2.contourArea(contour) > min_contour_area for contour in contours)

    if person_in_frame:
        if start_time is None:
            start_time = i
            # print(f"start {start_time}")
        if start_time is not None and i == dur:
            end_time = i
            # print(f"end {end_time}")
            intervals.append((start_time, end_time))
            start_time = None
    else:
        if start_time is not None:
            end_time = i
            # print(f"end {end_time}")
            intervals.append((start_time, end_time))
            start_time = None        
    
    # print(i)
    os.remove("../output/frame.bmp")
    load_bar.next()
load_bar.finish()

print("Результат...")
for start, end in intervals:
    print(f"Человек появился в кадре с {start} секунд и ушел с {end} секунд")

if len(intervals) > 0:
    file = open("../output/intrevals.txt", "w")

    for i, [start, end] in enumerate(intervals):
        os.system(f"ffmpeg -accurate_seek -i {input_video_path} \
                -ss {start-1} -t {end-start} -codec copy \
                    ../output/seg{i}.mp4 >/dev/null 2>&1")
        file.write(f"file '../output/seg{i}.mp4'\n")
        
    file.close()
    
    os.system(f"ffmpeg -f concat -safe 0 -i ../output/intrevals.txt \
            -codec copy {output_video_path} >/dev/null 2>&1")

    for i in range(len(intervals)):
        os.remove(f"../output/seg{i}.mp4")

    os.remove("../output/intrevals.txt")

