import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_distance(landmarks1, landmarks2):
    total_distance = 0
    num_landmarks = 0
    if landmarks1 and landmarks2:
        for lm1, lm2 in zip(landmarks1.landmark, landmarks2.landmark):
            total_distance += ((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) ** 0.5
            num_landmarks += 1
    average_distance = total_distance / num_landmarks if num_landmarks > 0 else 0
    return total_distance, average_distance



def process_video(caps, output_path):
    frames = []
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'h264')  # or use 'XVID'

    size = (int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    close_frames = 0
    total_frames = 0
    scores = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while all(cap.isOpened() for cap in caps):
            rets, frames = zip(*[cap.read() for cap in caps])

            # Recolor images to RGB
            images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
            for image in images:
                image.flags.writeable = False

            # Make detections
            results = [pose.process(image) for image in images]

            # Draw the pose landmarks on each frame
            for i, result in enumerate(results):
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(frames[i], result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            # Resize all frames to the size of the first frame
            frames = [cv2.resize(frame, (frames[0].shape[1], frames[0].shape[0])) for frame in frames]

            # Scale down the frames
            scale_factor = 0.5
            frames = [cv2.resize(frame, (int(frame.shape[1]*scale_factor), int(frame.shape[0]*scale_factor))) for frame in frames]

            # Write the frames to the output video
            out.write(np.hstack(frames))

            # Display the frames side by side
            cv2.imshow('Mediapipe Feed', np.hstack(frames))

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        for cap in caps:
            cap.release()
        out.release()
        cv2.destroyAllWindows()

def main():
    caps = [cv2.VideoCapture(path) for path in [
        'C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\woman.mp4',
        'C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\krisijanis.mp4',
        'C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\jurgis.mp4',
        'C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\emils.mp4',
        'C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\zach.mp4'
    ]]
    output_path = 'C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\team.mp4'
    process_video(caps, output_path)

if __name__ == "__main__":
    main()