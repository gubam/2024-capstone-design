# vision_utils.py

def detect_face(rgb_frame,mp_face):
    result = mp_face.process(rgb_frame)
    return result.detections is not None and len(result.detections) > 0


def detect_hands(rgb_frame,mp_hands):
    result = mp_hands.process(rgb_frame)
    left, right = False, False

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand in result.multi_handedness:
            label = hand.classification[0].label
            if label == "Left":
                left = True
            elif label == "Right":
                right = True
    return left, right

