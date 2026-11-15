
import cv2
import os
from ultralytics import YOLO
from collections import defaultdict, deque
video_path = "videos/match.mp4"
output_path = "output/players_detected.mp4"

os.makedirs("output", exist_ok=True)

video = cv2.VideoCapture(video_path)
model = YOLO("yolo26s.pt")

if not video.isOpened():
    print("Could not open video")

else:
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("Could not create output video")

    else:
        frame_number = 0
        track_history = defaultdict(lambda: deque(maxlen=60))
        while True:
            success, frame = video.read()

            if not success:
                break

            results = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=[0],
                conf=0.20,
                imgsz=960,
                verbose=False
            )


            annotated_frame = results[0].plot()

            boxes = results[0].boxes

            if boxes is not None and boxes.id is not None:

                player_ids = boxes.id.int().cpu().tolist()
                coordinates = boxes.xyxy.cpu().numpy()


                for player_id, box in zip(player_ids, coordinates):

                    # 1. Get player coordinates
                    x1, y1, x2, y2 = box

                    center_x = int((x1 + x2) / 2)
                    bottom_y = int(y2)

                    # 2. Save position to history
                    history = track_history[player_id]
                    history.append((center_x, bottom_y))

                    # 3. Draw player trajectory
                    for i in range(1, len(history)):
                        previous_point = history[i - 1]
                        current_point = history[i]

                        cv2.line(
                            annotated_frame,
                            previous_point,
                            current_point,
                            (0, 255, 255),
                            2
                        )

                    # 4. Draw current position
                    cv2.circle(
                        annotated_frame,
                        (center_x, bottom_y),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    # 5. Print coordinates
                    if frame_number % 30 == 0:
                        print(
                            f"Player {player_id}: "
                            f"x={center_x}, y={bottom_y}"
                        )



            people_count = len(boxes) if boxes is not None else 0

            cv2.putText(
                annotated_frame,
                f"People detected: {people_count}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            writer.write(annotated_frame)

            cv2.imshow("Football Analytics", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_number += 1

        writer.release()
        print("Video saved:", output_path)

video.release()
cv2.destroyAllWindows()
