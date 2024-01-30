import os
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class App:
    def __init__(self, window, window_title, video_source=0, model_path=r"D:\Jawad\TrainedModels\train2\weights\last.pt"):
        self.window = window
        self.window.title(window_title)
        self.window.resizable(False, False)

        # Lock the aspect ratio to square for the canvases
        canvas_size = min(window.winfo_screenwidth(), window.winfo_screenheight()) // 2
        # Set the distance from the bottom of the canvas to the buttons
        button_offset_y = 20  # Distance in pixels from the canvas to the buttons

        # Set the button width to be a bit wider
        button_width = 40  # Width in pixels

        # Set window size to accommodate two square canvases side by side and buttons/label below
        window_width = canvas_size * 2
        window_height = canvas_size + 50  # Additional space for buttons and label
        self.window.geometry(f'{window_width}x{window_height}')

        self.video_source = video_source
        self.current_image = None
        self.paused = False

        # Open the video source
        self.vid = cv2.VideoCapture(video_source)

        # Create two square canvases side by side
        self.canvas = tk.Canvas(window, width=canvas_size, height=canvas_size)
        self.canvas.place(x=0, y=0)

        self.processed_canvas = tk.Canvas(window, width=canvas_size, height=canvas_size)
        self.processed_canvas.place(x=canvas_size, y=0)

        # Place buttons slightly higher under the first canvas
        btn_y_position = canvas_size - button_offset_y

        self.btn_snapshot = tk.Button(window, text="Capture", width=button_width, command=self.snapshot)
        self.btn_snapshot.place(x=canvas_size // 4 - button_width // 4, y=btn_y_position)

        self.btn_resume = tk.Button(window, text="Resume", width=button_width, command=self.resume)
        self.btn_resume.place(x=canvas_size // 4 - button_width // 4,
                              y=btn_y_position + self.btn_snapshot.winfo_reqheight() + 5)

        # Place the status label under the second canvas
        self.lbl_status = tk.Label(window, text="Status: Please Capture The Image")
        self.lbl_status.place(x=canvas_size + canvas_size // 2 - self.lbl_status.winfo_reqwidth() // 2, y=canvas_size + 5)

        # Load the YOLO model
        self.model = YOLO(model_path)

        # Create output folder if it doesn't exist
        self.output_folder = 'output_images'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Update & display frames in the Tkinter window
        self.update()
        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.read()
        if ret:
            self.paused_frame = frame
            self.paused = True

            # Save the frame
            frame_name = "frame-" + str(int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))) + ".jpg"
            frame_path = os.path.join(self.output_folder, frame_name)
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # Process the image and display it
            self.predict_and_display(frame_path)

    def predict_and_display(self, img_path):
        # Load the image using OpenCV
        frame = cv2.imread(img_path)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference on the RGB image
        results = self.model(rgb_frame)  # results list

        # Process each result
        for r in results:
            # Use plot to visualize the predictions
            im_array = r.plot()  # plot a BGR numpy array of predictions

            # Since plot returns a BGR numpy array, convert it to RGB format
            rgb_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

            # Convert the RGB array to a PIL image
            im = Image.fromarray(rgb_array)

            # Save the image to the specified directory
            final_output_dir = r'D:\Jawad\infermb'
            if not os.path.exists(final_output_dir):
                os.makedirs(final_output_dir)
            predicted_img_filename = os.path.basename(img_path)
            final_output_path = os.path.join(final_output_dir, predicted_img_filename)
            im.save(final_output_path)  # save image

            # Display the image in the new Tkinter processed_canvas
            self.processed_photo = ImageTk.PhotoImage(image=im)
            self.processed_canvas.create_image(0, 0, image=self.processed_photo, anchor=tk.NW)
            # Keep a reference to avoid garbage collection
            self.processed_canvas.photo = self.processed_photo
            self.lbl_status.config(text="Status: Yellow Mosaic Detected")

    def resume(self):
        self.paused = False
        self.lbl_status.config(text="Status: Please Capture The Image")

    def update(self):
        if not self.paused:
            # Get a frame from the video source
            ret, frame = self.vid.read()
        else:
            # Use the paused frame
            ret, frame = True, self.paused_frame

        if ret:
            # Convert the image from BGR color (which OpenCV uses) to RGB color
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_image = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.current_image, anchor=tk.NW)

        self.window.after(10, self.update)

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Run the application
App(tk.Tk(), "Tkinter and OpenCV", model_path=r"D:\Jawad\TrainedModels\train2\weights\last.pt")
