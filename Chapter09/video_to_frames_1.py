import cv2
video_path =
video_handle = cv2.VideoCapture(video_path)
frame_no = 0
while True:
  eof, frame = video_handle.read()
  if not eof:
      break
  cv2.imwrite("frame%d.jpg" % frame_no, frame)
  frame_no += 1