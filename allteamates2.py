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

def process_video(cap, cap2, cap3, cap4, cap5, output_path):
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'h264')  # or use 'XVID'

    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    size1 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out1 = cv2.VideoWriter(output_path, fourcc, fps, size1)

    size3 = (int(cap3.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap3.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out3 = cv2.VideoWriter(output_path, fourcc, fps, size3)

    size4 = (int(cap4.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap4.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out4 = cv2.VideoWriter(output_path, fourcc, fps, size4)

    size5 = (int(cap5.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap5.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out5 = cv2.VideoWriter(output_path, fourcc, fps, size5)

    close_frames = 0
    total_frames = 0
    scores = []

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and cap2.isOpened() and cap3.isOpened() and cap4.isOpened() and cap5.isOpened():
            ret, frame = cap.read()
            ret2, frame2 = cap2.read()
            ret3, frame3 = cap3.read()
            ret4, frame4 = cap4.read()
            ret5, frame5 = cap5.read()

            # Recolor images to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            image3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGB)
            image4 = cv2.cvtColor(frame4, cv2.COLOR_BGR2RGB)
            image5 = cv2.cvtColor(frame5, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            image2.flags.writeable = False
            image3.flags.writeable = False
            image4.flags.writeable = False
            image5.flags.writeable = False


            # Make detections
            results = pose.process(image)
            results2 = pose.process(image2)
            results3 = pose.process(image3)
            results4 = pose.process(image4)
            results5 = pose.process(image5)

            # Create a blank image for drawing the landmarks
            blank_image2 = np.zeros(frame2.shape, dtype=np.uint8)
            blank_image3 = np.zeros(frame3.shape, dtype=np.uint8)
            blank_image4 = np.zeros(frame4.shape, dtype=np.uint8)
            blank_image5 = np.zeros(frame5.shape, dtype=np.uint8)

            # Draw the pose landmarks on blank_image2
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            if results2.pose_landmarks:
                mp_drawing.draw_landmarks(frame2, results2.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
                
            if results3.pose_landmarks:
                mp_drawing.draw_landmarks(frame3, results3.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            if results4.pose_landmarks:
                mp_drawing.draw_landmarks(frame4, results4.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            if results5.pose_landmarks:
                mp_drawing.draw_landmarks(frame5, results5.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                          mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))
            

            # Overlay blank_image2 on frame

            blank_image2 = cv2.resize(blank_image2, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 0.5, blank_image2, 0.5, 0)

            combined_image = np.concatenate((frame, blank_image2), axis=1)

            frames.append(combined_image)
            
            out.write(combined_image)
           
            cv2.imshow('Mediapipe Feed', combined_image)

           

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cap2.release()
        cv2.destroyAllWindows()
        out.release()

def main():
    cap = cv2.VideoCapture('C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\woman.mp4')
    cap2 = cv2.VideoCapture('C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\krisijanis.mp4')
    cap3 = cv2.VideoCapture('C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\emils.mp4')
    cap4 = cv2.VideoCapture('C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\jurgis.mp4')
    cap5 = cv2.VideoCapture('C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\zach.mp4')
    output_path = 'C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\team.mp4'
    process_video(cap, cap2, cap3, cap4, cap5, output_path)

if __name__ == "__main__":
    main()