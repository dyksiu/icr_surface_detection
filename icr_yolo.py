import cv2
from ultralytics import YOLO
from tkinter import Tk, Label, Button, filedialog, Scale, HORIZONTAL, messagebox
import csv
import random
import numpy as np

class YoloVideoApp:
    # Konstruktor klasy - tworzy GUI i tu sa inicjalizowane wszystkie elementy apki:
    # -> okno - detektor powierzchni
    # -> przyciski i suwaki
    def __init__(self, window):
        self.window = window
        self.window.title("Detektor powierzchni")
        self.window.geometry("420x400")
        self.window.resizable(False, False)

        self.model = YOLO("runs/detect/train11/weights/best.pt")
        self.class_confidences = {}
        self.detections = []

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

    # Metoda - zwraca stały kolor RGB dla danej klasy na podstawie jego ID:
    # -> random.seed - kazda klasa dostaje ten sam kolor ale rozne klasy maja rozne kolory
    # -> zwracany jest kolor jako krotka (bezpiecznie bo moze przechowywac dane roznego typu)
    def get_class_color(self, cls_id):
        random.seed(cls_id)
        return tuple(random.randint(0, 255) for _ in range(3))

    # Metoda - aktualizuje etykiety obok suwaka conf, tak aby wyswietlana byla aktualna wartosc
    def update_conf_label(self, val):
        self.conf_label.config(text=f"Confidence: {int(val) / 100:.2f}")

    # Metoda - umozliwia otworzenia okna wyboru pliku .pt z zapisanym modelem YOLO
    # -> zabezpieczona wyjatkiem na wypadek niemozliwosci zaladowania modelu
    def select_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("YOLO model files", "*.pt")])
        # Dodany wyjatek dla zabezpieczenia przed wykrzaczeniem
        if model_path:
            try:
                self.model = YOLO(model_path)
                model_name = model_path.split("/")[-1]
                self.model_name_label.config(text=f"Aktualny model: {model_name}")
                messagebox.showinfo("Model załadowany", f"Pomyślnie załadowano model:\n{model_name}")
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie udało się załadować modelu:\n{e}")

    # Metoda - glowna funkcja odpowiedzialna za przetworzenie i detekcje w wybranym filmie - format .mp4
    def select_video(self):
        # Wybor pliku wideo - otwiera sie nowe okno wyboru
        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if not video_path:
            return

        self.class_confidences = {}
        self.detections = []

        conf_value = self.conf_slider.get() / 100
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            overlay = frame.copy()

            results = self.model(frame, conf=conf_value, device=0)
            for r in results:
                if hasattr(r, 'masks') and r.masks is not None:
                    masks = r.masks.data.cpu().numpy()
                    for idx, mask in enumerate(masks):
                        cls_id = int(r.boxes.cls[idx].item())
                        conf = float(r.boxes.conf[idx].item())

                        if cls_id not in self.class_confidences:
                            self.class_confidences[cls_id] = []
                        self.class_confidences[cls_id].append(conf)
                        self.detections.append((frame_count, cls_id, round(conf, 4)))

                        color = self.get_class_color(cls_id)
                        mask_resized = cv2.resize((mask * 255).astype("uint8"), (width, height))

                        # Wypelnia wykryty obszar kolorem klasy
                        for c in range(3):
                            overlay[:, :, c][mask_resized > 0] = (
                                color[c]
                            )

                        # Rysowanie konturow
                        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(frame, contours, -1, color, 2)

                        # Rysowanie etykiety klasy
                        if contours:
                            M = cv2.moments(contours[0])
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                label = f"{self.model.names[cls_id]}: {conf:.2%}"
                                cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    alpha = 0.4
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                else:
                    # --- KLASYCZNA DETEKCJA ---
                    boxes = r.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())

                        if cls_id not in self.class_confidences:
                            self.class_confidences[cls_id] = []
                        self.class_confidences[cls_id].append(conf)
                        self.detections.append((frame_count, cls_id, round(conf, 4)))

                        color = self.get_class_color(cls_id)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{self.model.names[cls_id]}: {conf:.2%}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    frame = results[0].plot()

            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Gotowe", "Analiza zakończona. Możesz teraz zapisać wyniki do CSV.")

    # Metoda - zapis wynikow do pliku CSV
    def save_csv(self):
        # Sprawdzenie czy dane sa gotowe do zapisania -> po przeprowadzeniu analizy czyli jak skonczy sie film
        if not self.class_confidences or not self.detections:
            messagebox.showwarning("Brak danych", "Nie ma danych do zapisania. Najpierw przeanalizuj film.")
            return

        # Podanie sciezki do zapisu pliku
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return

        # Wyjatek:
        # -> ma na celu przechwycenie bledu np. podczas zapisu - ochrona przed crashem programu
        ##
        # -> wewnatrz wyjatku tworzony jest plik csv ktory zawiera:
        #    -> legende klasy czyli ID i nazwe
        #    -> podsumowanie czyli sredni conf i liczbe wykryc danej klasy
        #    -> kazde wykrycie klatka po klatce
        try:
            with open(save_path, mode='w', newline='') as file:
                writer = csv.writer(file, delimiter=';')

                writer.writerow(["# LEGENDA KLAS (format: ID = Nazwa)"])
                for cls_id, name in self.model.names.items():
                    writer.writerow([f"# {cls_id} = {name}"])
                writer.writerow([])

                writer.writerow(["PODSUMOWANIE"])
                writer.writerow(["Klasa", "Średni confidence", "Liczba wykryć"])
                for cls_id, confs in self.class_confidences.items():
                    avg_conf = sum(confs) / len(confs)
                    writer.writerow([cls_id, round(avg_conf, 4), len(confs)])
                writer.writerow([])

                writer.writerow(["WSZYSTKIE WYKRYCIA"])
                writer.writerow(["Numer klatki", "Klasa", "Confidence"])
                for detection in self.detections:
                    writer.writerow(detection)

            messagebox.showinfo("Zapisano", f"Wyniki zapisane do:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie udało się zapisać pliku:\n{e}")

if __name__ == "__main__":
    root = Tk()
    app = YoloVideoApp(root)
    root.mainloop()
