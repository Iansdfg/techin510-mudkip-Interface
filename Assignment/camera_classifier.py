import cv2
from cv2 import face
from tkinter import *
import webbrowser


def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def showVideo():
    haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    recognizer = LBPHFaceRecognizer_create()
    recognizer.read("trainner.yml")

    cap = cv2.VideoCapture(0)
    cap.set(3, 320)
    cap.set(4, 240)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', (600, 600))

    while (True):

        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            print(x , y, w, h)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y + h, x:x + w]

            id_, conf = recognizer.predict(roi_gray)
            if conf >= 45 and conf <= 85:
                print(id_)


            img_item = "face/my-image.png"
            cv2.imwrite(img_item,roi_gray)

            color = (255,0,0)
            stroke = 2
            cv2.rectangle(frame, (x,y),(x+w,y+h),color,stroke)


        text = "Faces found:" + str(len(faces))
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow('frame', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def ask_for_name():
    master = Tk()
    Label(master, text="Hi!, I am Muddy").grid(row=0)
    Label(master, text="What is you name").grid(row=1)
    name_entry = Entry(master)
    name_entry.grid(row=2, column=0)

    def get_name():
        return (name_entry.get())

    def ask_for_goal():
        user_name = get_name()
        Label(master, text="Hello, " + str(user_name) + "！Nice to meet you!").grid(row=4)
        Label(master, text="What is you fitness goal").grid(row=5)
        goal_entry = Entry(master)
        goal_entry.grid(row=6, column=0)

        def get_goal():
            return (goal_entry.get())

        def sample_exercise():
            Label(master, text="You goal is " + str(get_goal()) + ". You can make it!!!").grid(row=8)
            Button(master, text='Quit', command=master.quit).grid(row=9, column=0, sticky=W, pady=4)

            def open_page():
                url = 'https://www.nike.com/us/en_us/c/training/nike-training-club/workouts'
                webbrowser.open_new_tab(url + 'doc/')

            Button(master, text='Exercise', command=open_page).grid(row=9, column=1, sticky=W, pady=4)
            Button(master, text='Open Video', command=showVideo).grid(row=9, column=2, sticky=W, pady=4)

        Button(master, text='Quit', command=master.quit).grid(row=7, column=0, sticky=W, pady=4)
        Button(master, text='Next', command=sample_exercise).grid(row=7, column=1, sticky=W, pady=4)

    Button(master, text='Quit', command=master.quit).grid(row=3, column=0, sticky=W, pady=4)
    Button(master, text='Next', command=ask_for_goal).grid(row=3, column=1, sticky=W, pady=4)


if __name__ == "__main__":
    showVideo();
    # ask_for_name()
    # mainloop()

