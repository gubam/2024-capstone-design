import cv2

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 웹캠 해상도를 FHD로 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 실제로 설정된 해상도 확인
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"녹화 해상도: {width}x{height}")

# 저장할 파일 설정 (코덱: MP4V, 확장자: .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_fhd.mp4', fourcc, 20.0, (width, height))

print("녹화를 시작합니다. 'q' 키를 누르면 종료됩니다.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)  # 원본 FHD 영상 저장

    # 미리보기용 작은 프레임 생성 (640x480)
    preview_frame = cv2.resize(frame, (640, 480))
    cv2.imshow('Recording Preview (1280x720)', preview_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("녹화를 종료합니다.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
