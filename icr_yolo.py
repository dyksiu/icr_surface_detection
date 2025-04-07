import cv2
from ultralytics import YOLO
from tkinter import Tk, Label, Button, filedialog, Scale, HORIZONTAL, messagebox
import csv


class YoloVideoApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Detektor powierzchni")
        self.window.geometry("420x400")
        self.window.resizable(False, False)

        self.model = YOLO("runs/detect/train20/weights/best.pt")
        self.class_confidences = {}  # Nowe: do zbierania wyników

        self.label = Label(window, text="Wybierz model YOLO")
        self.label.pack(pady=10)

        self.model_button = Button(window, text="Model", command=self.select_model)
        self.model_button.pack(pady=5)

        self.model_name_label = Label(window, text="Aktualny model: best.pt", fg="gray")
        self.model_name_label.pack()

        self.label = Label(window, text="Wybierz plik MP4 i ustaw próg confidence")
        self.label.pack(pady=10)

        self.conf_label = Label(window, text="Confidence: 0.25")
        self.conf_label.pack()

        self.conf_slider = Scale(window, from_=5, to=95, orient=HORIZONTAL, command=self.update_conf_label)
        self.conf_slider.set(25)
        self.conf_slider.pack()

        self.video_button = Button(window, text="Wybierz wideo", command=self.select_video)
        self.video_button.pack(pady=10)

        self.save_csv_button = Button(window, text="Zapisz wyniki do CSV", command=self.save_csv)
        self.save_csv_button.pack(pady=10)

    def update_conf_label(self, val):
        self.conf_label.config(text=f"Confidence: {int(val) / 100:.2f}")

    def select_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("YOLO model files", "*.pt")])
        if model_path:
            try:
                self.model = YOLO(model_path)
                model_name = model_path.split("/")[-1]
                self.model_name_label.config(text=f"Aktualny model: {model_name}")
                messagebox.showinfo("Model załadowany", f"Pomyślnie załadowano model:\n{model_name}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się załadować modelu:\n{e}")

    def select_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not video_path:
            return

        self.class_confidences = {}  # Resetuj dane

        conf_value = self.conf_slider.get() / 100
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, conf=conf_value, device=0)
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        if cls_id not in self.class_confidences:
                            self.class_confidences[cls_id] = []
                        self.class_confidences[cls_id].append(conf)

            annotated = results[0].plot()
            cv2.imshow("YOLOv8 Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Gotowe", "Analiza zakończona. Możesz teraz zapisać wyniki do CSV.")

    def save_csv(self):
        if not self.class_confidences:
            messagebox.showwarning("Brak danych", "Nie ma danych do zapisania. Najpierw przeanalizuj film.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return

        try:
            with open(save_path, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                writer.writerow(["Klasa", "Średni confidence", "Liczba wykryć"])
                for cls_id, confs in self.class_confidences.items():
                    avg_conf = sum(confs) / len(confs)
                    writer.writerow([cls_id, round(avg_conf, 4), len(confs)])
            messagebox.showinfo("Zapisano", f"Wyniki zapisane do:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać pliku:\n{e}")


if __name__ == "__main__":
    root = Tk()
    app = YoloVideoApp(root)
    root.mainloop()
