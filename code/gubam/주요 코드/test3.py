
import cv2
import mediapipe as mp

# 이미지 경로
image_path = "C:/Users/82109/Desktop/test2.png"  # 🔁 여기 이미지 경로 입력
import cv2
import mediapipe as mp

# 이미지 불러오기
image = cv2.imread(image_path)  # 🔁 경로 바꾸세요
import cv2
import mediapipe as mp

# 이미지 경로
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w = image.shape[:2]

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(static_image_mode=True) as holistic:
    results = holistic.process(image_rgb)

    # 유틸 함수: 특정 landmark들을 점으로 표시
    def draw_landmarks_pointwise(image, landmarks, index_list, color):
        for idx in index_list:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(image, (x, y), 5, color, -1)

    # ✅ 상체 포인트 (어깨~손목)
    if results.pose_landmarks:
        pose_lms = results.pose_landmarks.landmark
        upper_body_indices = list(range(11, 17))  # 어깨~손목
        draw_landmarks_pointwise(image, pose_lms, upper_body_indices, (0, 255, 0))  # 초록색

    # ✅ 오른손 포인트 21개
    if results.right_hand_landmarks:
        rh_lms = results.right_hand_landmarks.landmark
        draw_landmarks_pointwise(image, rh_lms, range(21), (0, 0, 255))  # 빨간색

    # ✅ 왼손 포인트 21개
    if results.left_hand_landmarks:
        lh_lms = results.left_hand_landmarks.landmark
        draw_landmarks_pointwise(image, lh_lms, range(21), (255, 0, 0))  # 파란색

    # 결과 출력
    cv2.imshow("Upper Body + Full Hand Points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
