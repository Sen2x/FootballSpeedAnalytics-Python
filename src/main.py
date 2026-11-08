import cv2

video_path = "videos/match.mp4"

video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Could not open video")
else:
    print("Video opened successfully")
    fps=video.get(cv2.CAP_PROP_FPS)
    width =video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print("FPS:", fps)
    print("Width:", width)
    print("Height:", height)
    print("Frames:", frames)
    success, frame = video.read()
    if success ==True: 
        cv2.imshow("First frame", frame)
        cv2.waitKey(0)
    else:
        print("Could not read the first frame")
        
video.release()
cv2.destroyAllWindows()