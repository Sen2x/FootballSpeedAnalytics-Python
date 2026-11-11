import cv2

video_path = "videos/match.mp4"
video = cv2.VideoCapture(video_path)

if not video.isOpened():
    print("Could not open video")
else:
    print("Video opened successfully")
    print("FPS:", video.get(cv2.CAP_PROP_FPS))
    print("Width:", video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Height:", video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Frames:", video.get(cv2.CAP_PROP_FRAME_COUNT))

    firstFrame = True
    frame_number = 0

    while True:
        success, frame = video.read()

        if not success:
            if firstFrame:
                print("Could not read the first frame")
            break

        if firstFrame:
            print(type(frame))
            print(frame.shape)
            print(frame[100, 200])
            firstFrame = False

        cv2.imshow("Video", frame)
        cv2.putText(
            frame,
            f"Frame {frame_number}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        if cv2.waitKey(33) == ord("w"):
            break
        frame_number += 1

video.release()
cv2.destroyAllWindows()