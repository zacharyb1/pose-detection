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

def process_video(cap, cap2, output_path):
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'h264')  # or use 'XVID'
    
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 2), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # Adjust the width
    out = cv2.VideoWriter(output_path, fourcc, fps, size)   

    close_frames = 0
    total_frames = 0
    scores = []

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

            # Calculate the total and average distance between corresponding landmarks
            total_distance, average_distance = calculate_distance(results.pose_landmarks, results2.pose_landmarks)

            # If the total distance is below the threshold, increment the counter
            threshold_distance = 2.8  # Adjust this value as needed
            if total_distance < threshold_distance:
                close_frames += 1

            total_frames += 1

           # Replace the line where you calculate the score with these lines
            score = (close_frames / total_frames) * 100
            scores.append(score)        

            # After the while loop, calculate the average score
            average_score = sum(scores) / len(scores) if scores else 0


            # Display the score and average distance on blank_image2
            cv2.putText(blank_image2, f'Score: {score:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blank_image2, f'Average Distance: {average_distance:.2f}', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(blank_image2, f'Average Score: {average_score:.2f}', (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw lines between corresponding landmarks
            if results.pose_landmarks and results2.pose_landmarks:
                for lm1, lm2 in zip(results.pose_landmarks.landmark, results2.pose_landmarks.landmark):
                    start_point = (int(lm1.x * frame.shape[1]), int(lm1.y * frame.shape[0]))
                    end_point = (int(lm2.x * frame2.shape[1]), int(lm2.y * frame2.shape[0]))
                    cv2.line(blank_image2, start_point, end_point, (0, 0, 255), 2)

            # Concatenate images horizontally
            combined_image = np.concatenate((blank_image, blank_image2), axis=1)
           
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
    cap = cv2.VideoCapture('C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\jurg.is.mp4')
    cap2 = cv2.VideoCapture('C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\nephew.mp4')
    output_path = 'C:\\Users\\zach\\Documents\\Projects\\pose-detection\\media\\new.mp4'
    process_video(cap, cap2, output_path)

if __name__ == "__main__":
    main()