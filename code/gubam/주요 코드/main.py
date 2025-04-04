import cv2
import mp_keypoint
import training
import json
import os

label = "you"
folder_path = f"C:/Users/gubam/Desktop/sdata/{label}"
file_list = os.listdir(folder_path)
print(file_list)

for i in range(len(file_list)):
    VIDEO_SRC = f"C:/Users/gubam/Desktop/sdata/{label}/{file_list[i]}"
    keypoint = mp_keypoint.keypoint(kf_sw = True, draw_graph_sw = False, z_kill = True)
    cap = cv2.VideoCapture(VIDEO_SRC)
    sampling = training.ScoreSampling(50)
    counter = 0


    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        keypoint.extract_keypoint(frame)
        image = keypoint.frame
        output = keypoint.angle
        
        cv2.imshow( 'video', cv2.resize(image,dsize=(960,540)) )
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        counter += 1



    frame_list, angle_list = sampling.sampling(keypoint.score_list,keypoint.frame_list,keypoint.angle_list)
    # for frame in frame_list:
    #     cv2.imshow("Video Playback",cv2.resize(frame,dsize=(960,540)) )  # 이미지 표시
    #     if cv2.waitKey(int(1/30 * 1000)) & 0xFF == ord('q'):  # 'q'를 누르면 종료
    #         break

    print(i)

    with open(f"C:/Users/gubam/Desktop/sdata/json/{label}/{i}.json", "w", encoding="utf-8") as f:
        json.dump({"data": angle_list, "label" : label}, f, ensure_ascii=False, indent=4)
        
    cap.release()
    cv2.destroyAllWindows()