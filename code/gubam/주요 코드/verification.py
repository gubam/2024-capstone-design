import numpy as np
import matplotlib.pyplot as plt

class verification:
    
    def __init__(self):
        self.angles = []
        plt.ion()  # 인터랙티브 모드 활성화
        self.fig, self.ax = plt.subplots(figsize=(12, 5))
        self.bar_container = None

    def angle_ver(self, angles):
        self.ax.clear()  # 이전 그래프 삭제
        indices = np.arange(len(angles))  # 각도 인덱스
        
        self.ax.bar(indices, angles, color='royalblue', alpha=0.7)
        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("Angle (radians)")
        self.ax.set_title("각도 시각화")
        self.ax.set_ylim(0, 3.2)  # 최대값보다 조금 크게 설정
        self.ax.grid(axis="y", linestyle="--", alpha=0.6)
        
        plt.draw()  # 그래프 갱신
        plt.pause(0.01)  # 잠시 멈추고 다시 실행 (빠르게 업데이트)
