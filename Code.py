import cv2
import mediapipe as mp
import pyautogui


# Initialize hand detector and drawing utility
def init_hand_detector():
    return mp.solutions.hands.Hands(), mp.solutions.drawing_utils


# Process the video feed and detect hands
def process_frame(frame, hand_detector):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hand_detector.process(rgb_frame).multi_hand_landmarks


# Draw landmarks on detected hands
def draw_landmarks(frame, hands, drawing_utils):
    for hand in hands:
        drawing_utils.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)


# Map coordinates for cursor movement
def get_screen_position(x, y, frame_width, frame_height, screen_width, screen_height):
    return int(screen_width * (x / frame_width)), int(screen_height * (y / frame_height))


# Control mouse with hand gestures
def control_mouse(landmarks, frame_width, frame_height, screen_width, screen_height):
    index_x, index_y = None, None
    thumb_x, thumb_y = None, None

    for id, landmark in enumerate(landmarks):
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)

        # Check for index finger tip (ID 8)
        if id == 8:
            index_x, index_y = get_screen_position(x, y, frame_width, frame_height, screen_width, screen_height)

        # Check for thumb tip (ID 4)
        if id == 4:
            thumb_x, thumb_y = get_screen_position(x, y, frame_width, frame_height, screen_width, screen_height)

    if index_x and thumb_x:
        if abs(index_y - thumb_y) < 20:  # Perform click when fingers are close
            pyautogui.click()
            pyautogui.sleep(1)
        elif abs(index_y - thumb_y) < 100:  # Move mouse otherwise
            pyautogui.moveTo(index_x, index_y)


# Main function to execute the virtual mouse control
def main():
    cap = cv2.VideoCapture(0)
    hand_detector, drawing_utils = init_hand_detector()
    screen_width, screen_height = pyautogui.size()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        hands = process_frame(frame, hand_detector)

        if hands:
            draw_landmarks(frame, hands, drawing_utils)
            for hand in hands:
                control_mouse(hand.landmark, frame_width, frame_height, screen_width, screen_height)

        cv2.imshow('Gesture Mouse Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
