#from https://stackoverflow.com/a/33399711/7723026

import cv2

vidcap = cv2.VideoCapture('./saliency_maps/movies/a2c/BreakoutToyboxNoFrameskip-v4/IVsymbricks-250-breakouttoyboxnoframeskip-v4-3.mp4')

success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("./saliency_maps/experiments/results/breakout_flipBrick_ex/frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1