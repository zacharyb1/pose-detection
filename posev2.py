import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def process_video(cap, cap2):
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and cap2.isOpened():
            ret, frame = cap.read()
            ret2, frame2 = cap2.read()

            # Recolor images to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            image2.flags.writeable = False

            # Make detections
            results = pose.process(image)
            results2 = pose.process(image2)

            # Create blank images with the same dimensions as the frames
            blank_image = np.zeros(frame.shape, dtype=np.uint8)
            blank_image2 = np.zeros(frame2.shape, dtype=np.uint8)

            # Render detections on the blank images
            mp_drawing.draw_landmarks(blank_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            mp_drawing.draw_landmarks(blank_image2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            # Concatenate images horizontally
            combined_image = np.concatenate((blank_image, blank_image2), axis=1)

            cv2.imshow('Mediapipe Feed', combined_image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cap2.release()
        cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture('C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\jurg.is.mp4')
    cap2 = cv2.VideoCapture('C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\nephew.mp4')
    process_video(cap, cap2)

if __name__ == "__main__":
    main()