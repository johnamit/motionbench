import cv2


def test_camera():
    camera_index = 0
    capture = cv2.VideoCapture(camera_index)

    print(f"Camera opened: {capture.isOpened()}")
    if not capture.isOpened():
        return

    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = capture.get(cv2.CAP_PROP_FPS)
    print("Width:", width)
    print("Height:", height)
    print("FPS:", fps)

    frame = None
    for _ in range(20):
        ret, candidate = capture.read()
        if not ret:
            continue
        frame = candidate
        if frame.mean() > 5.0:
            break

    if frame is None:
        print("Failed to capture frame")
    else:
        print("Captured frame:", frame.shape)

    capture.release()


if __name__ == "__main__":
    test_camera()
