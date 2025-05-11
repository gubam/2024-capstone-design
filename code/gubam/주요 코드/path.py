import os
import shutil

parent_folder = "C:/Users/gubam/Desktop/new_video/you"

# parent_folder 내부의 모든 하위 폴더 반복
for subdir in os.listdir(parent_folder):
    subdir_path = os.path.join(parent_folder, subdir)
    
    if os.path.isdir(subdir_path):  # 폴더일 때만 처리
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            new_path = os.path.join(parent_folder, file)
            
            if os.path.isfile(file_path):
                shutil.move(file_path, new_path)

        # 하위 폴더가 비었으면 제거해도 됨 (선택사항)
        os.rmdir(subdir_path)
