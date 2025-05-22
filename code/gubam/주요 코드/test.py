import cv2

# 웹캠 열기 (0: 기본 카메라)
cap = cv2.VideoCapture(0)

# 고해상도 설정 (예: 1920x1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # MJPG 포맷 권장
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 실제 적용된 해상도 확인
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"해상도: {width}x{height}, FPS: {fps}")

# 비디오 저장 설정
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 파일 코덱
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))
if not out.isOpened():
    print("❌ VideoWriter가 열리지 않았습니다. 저장 실패!")

# 녹화 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    out.write(frame)  # 프레임 저장
    # 보기 좋게 리사이즈해서 화면에 출력
    display_frame = cv2.resize(frame, (960, 540))
    cv2.imshow('녹화 중... (q를 눌러 종료)', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
