import cv2
import asyncio

counter = 5
async def countdown(callback):
    global counter
    while counter > 0:
        print(counter)
        counter -= 1
        await asyncio.sleep(1)
    callback()

def process_after_countdown():
    print("Geri sayım tamamlandı! İşlev çalıştırılıyor...")

def show_camera():
    cap = cv2.VideoCapture(0)
    global counter

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, "Kamera Goruntusu", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if counter == 5:
            asyncio.run(countdown(process_after_countdown))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_camera()
