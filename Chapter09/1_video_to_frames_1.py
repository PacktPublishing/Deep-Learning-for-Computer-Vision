import cv2
video_path = '/Users/i335713/Desktop/epat/lecture recordings and live lectures/batch35epat  (batch 35) Lecture Recordings  Live Lec  Additional Lecture on Machine Learning (.mp4'
video_handle = cv2.VideoCapture(video_path)
frame_no = 0
while True:
  eof, frame = video_handle.read()
  if not eof:
      break
  cv2.imwrite("frame%d.jpg" % frame_no, frame)
  frame_no += 1