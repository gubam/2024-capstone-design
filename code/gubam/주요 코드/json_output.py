import cv2
import mp_keypoint
import training
import json
import os    
import gc
import mediapipe as mp

json_path = "C:/Users/gubam/Desktop/js"
folder_path = "C:/Users/gubam/Desktop/plus"
file_list = os.listdir(folder_path)

mp_holistic = mp.solutions.holistic
holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9)

file_list = [
    f for f in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, f)) and f != "json"
]
print(file_list)
print(len(file_list))
for i in range(len(file_list)):
    
    label = file_list[i]
    word_folder = f"{folder_path}/{label}"
    word_list = os.listdir(word_folder)
    
    for j in range(len(word_list)):
        print(len(word_list))
        VIDEO_SRC = f"{word_folder}/{word_list[j]}"
        keypoint = mp_keypoint.keypoint(kf_sw = True, draw_graph_sw = False, z_kill = True, mp_holistic = mp_holistic, holistic= holistic)
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
            
            #cv2.imshow( 'video', cv2.resize(image,dsize=(960,540)) )
            
            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     break
            
            counter += 1

        frame_list, angle_list = sampling.sampling(keypoint.score_list,keypoint.frame_list,keypoint.angle_list)
        # for frame in frame_list:
        #     cv2.imshow("Video Playback",cv2.resize(frame,dsize=(960,540)) )  # 이미지 표시
        #     if cv2.waitKey(int(1/30 * 1000)) & 0xFF == ord('q'):  # 'q'를 누르면 종료
        #         break

        os.makedirs(f"{json_path}/{label}", exist_ok=True)
        with open(f"{json_path}/{label}/{j}.json", "w", encoding="utf-8") as f:
            json.dump({"data": angle_list, "label" : label}, f, ensure_ascii=False, indent=4)
        print(f"{json_path}/{label}/{j}.json")
        cap.release()
        cv2.destroyAllWindows()
        del keypoint
        del cap
        gc.collect()