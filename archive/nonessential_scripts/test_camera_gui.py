import cv2


def main():
    camera_index = 0
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print("Could not open webcam.")
        return

    print("Camera preview started. Press 'q' to close.")
    while True:
        ok, frame = capture.read()
        if not ok:
            print("Failed to capture frame.")
            break

        cv2.imshow("Camera Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
