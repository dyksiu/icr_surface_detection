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
        self.window.geometry("420x440")
        self.window.resizable(False, False)

        self.model = None
        self.class_confidences = {}
        self.detections = []
        ## domyslnie ustawiony jezyk - PL
        self.lang = "pl"

        ## Zdefiniowanie slownika z tlumaczeniami, slowa kluczowe:
        # -> pl - jezyk polski
        # -> en - jezyk angielski
        self.translations = {
            "pl": {
                "select_model": "Wybierz model YOLO",
                "model": "Model",
                "current_model": "Aktualny model: ",
                "no_model": "BRAK",
                "select_video_conf": "Wybierz plik MP4 i ustaw próg confidence",
                "confidence": "Confidence",
                "choose_video": "Wybierz wideo",
                "save_csv": "Zapisz wyniki do CSV",
                "model_loaded": "Model załadowany",
                "model_loaded_msg": "Pomyślnie załadowano model:\n{}",
                "model_error": "Błąd",
                "model_error_msg": "Nie udało się załadować modelu:\n{}",
                "no_model_title": "Brak modelu",
                "no_model_msg": "Nie wybrano modelu YOLO!\nWybierz model przed rozpoczęciem analizy.",
                "done": "Gotowe",
                "done_msg": "Analiza zakończona. Możesz teraz zapisać wyniki do CSV.",
                "no_data": "Brak danych",
                "no_data_msg": "Nie ma danych do zapisania. Najpierw przeanalizuj film.",
                "saved": "Zapisano",
                "saved_msg": "Wyniki zapisane do:\n{}",
                "save_error": "Błąd",
                "save_error_msg": "Nie udało się zapisać pliku:\n{}",
                "lang_toggle": "Zmień język"
            },
            "en": {
                "select_model": "Select YOLO model",
                "model": "Model",
                "current_model": "Current model: ",
                "no_model": "NONE",
                "select_video_conf": "Select MP4 file and set confidence threshold",
                "confidence": "Confidence",
                "choose_video": "Choose video",
                "save_csv": "Save results to CSV",
                "model_loaded": "Model loaded",
                "model_loaded_msg": "Successfully loaded model:\n{}",
                "model_error": "Error",
                "model_error_msg": "Failed to load model:\n{}",
                "no_model_title": "No model",
                "no_model_msg": "No YOLO model selected!\nPlease select a model before analysis.",
                "done": "Done",
                "done_msg": "Analysis completed. You can now save the results to CSV.",
                "no_data": "No data",
                "no_data_msg": "No data to save. Please analyze a video first.",
                "saved": "Saved",
                "saved_msg": "Results saved to:\n{}",
                "save_error": "Error",
                "save_error_msg": "Failed to save file:\n{}",
                "lang_toggle": "Switch language"
            }
        }

        ## Zdefiniowanie slownika do tlumaczenia nazw klas zdefiniowanych w .yaml bez ingerencji w niego
        # -> pl - jezyk polski
        # -> en - jezyk angielski
        self.class_name_translations = {
            "pl": {},
            "en": {
                "asfalt": "asphalt",
                "trawa": "grass",
                "beton": "concrete",
                "kostka": "paving stones"
            }
        }

        self.model_label = Label(window, text=self.tr("select_model"))
        self.model_label.pack(pady=10)

        self.model_button = Button(window, text=self.tr("model"), command=self.select_model)
        self.model_button.pack(pady=5)

        self.model_name_label = Label(window, text=f"{self.tr('current_model')}{self.tr('no_model')}", fg="gray")
        self.model_name_label.pack()

        self.video_label = Label(window, text=self.tr("select_video_conf"))
        self.video_label.pack(pady=10)

        self.conf_label = Label(window, text=f"{self.tr('confidence')}: 0.25")
        self.conf_label.pack()

        self.conf_slider = Scale(window, from_=5, to=95, orient=HORIZONTAL, command=self.update_conf_label)
        self.conf_slider.set(25)
        self.conf_slider.pack()

        self.video_button = Button(window, text=self.tr("choose_video"), command=self.select_video)
        self.video_button.pack(pady=10)

        self.save_csv_button = Button(window, text=self.tr("save_csv"), command=self.save_csv)
        self.save_csv_button.pack(pady=10)

        self.lang_button = Button(window, text=self.tr("lang_toggle"), command=self.toggle_language)
        self.lang_button.pack(pady=5)

    # Metoda pomocnicza - tłumaczenie kluczy zgodnie z aktualnym językiem
    def tr(self, key):
        return self.translations[self.lang].get(key, key)

    # Metoda - zmiana języka aplikacji (PL <-> EN)
    def toggle_language(self):
        self.lang = "en" if self.lang == "pl" else "pl"
        self.update_labels()

    # Metoda - tlumaczenie nazw klas zgodnie z przyjetym kluczem (PL <-> EN)
    def translate_class_name(self,name):
        return self.class_name_translations.get(self.lang, {}).get(name,name)

    # Metoda - aktualizuje widoczne etykiety po zmianie języka
    def update_labels(self):
        self.model_label.config(text=self.tr("select_model"))
        self.model_button.config(text=self.tr("model"))
        model_name = self.model_name_label.cget("text").split(":")[-1].strip()
        self.model_name_label.config(
            text=f"{self.tr('current_model')}{model_name if self.model else self.tr('no_model')}"
        )
        self.video_label.config(text=self.tr("select_video_conf"))
        self.conf_label.config(text=f"{self.tr('confidence')}: {self.conf_slider.get() / 100:.2f}")
        self.video_button.config(text=self.tr("choose_video"))
        self.save_csv_button.config(text=self.tr("save_csv"))
        self.lang_button.config(text=self.tr("lang_toggle"))

    # Metoda - zwraca stały kolor RGB dla danej klasy na podstawie jego ID:
    def get_class_color(self, cls_id):
        random.seed(cls_id)
        return tuple(random.randint(0, 255) for _ in range(3))

    # Metoda - aktualizuje etykiety obok suwaka conf, tak aby wyswietlana byla aktualna wartosc
    def update_conf_label(self, val):
        self.conf_label.config(text=f"{self.tr('confidence')}: {int(val) / 100:.2f}")

    # Metoda - umozliwia otworzenia okna wyboru pliku .pt z zapisanym modelem YOLO
    def select_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("YOLO model files", "*.pt")])
        if model_path:
            try:
                self.model = YOLO(model_path)
                model_name = model_path.split("/")[-1]
                self.model_name_label.config(text=f"{self.tr('current_model')}{model_name}")
                messagebox.showinfo(self.tr("model_loaded"), self.tr("model_loaded_msg").format(model_name))
            except Exception as e:
                messagebox.showerror(self.tr("model_error"), self.tr("model_error_msg").format(e))

    # Metoda - glowna funkcja odpowiedzialna za przetworzenie i detekcje w wybranym filmie - format .mp4
    def select_video(self):
        if self.model is None:
            messagebox.showwarning(self.tr("no_model_title"), self.tr("no_model_msg"))
            return

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

                        for c in range(3):
                            overlay[:, :, c][mask_resized > 0] = color[c]

                        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(frame, contours, -1, color, 2)

                        if contours:
                            M = cv2.moments(contours[0])
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                                translated_name = self.translate_class_name(self.model.names[cls_id])
                                label = f"{translated_name}: {conf:.2%}"
                                cv2.putText(frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    alpha = 0.4
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                else:
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
                        translated_name = self.translate_class_name(self.model.names[cls_id])
                        label = f"{translated_name}: {conf:.2%}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    frame = results[0].plot()

            cv2.imshow("YOLOv8 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo(self.tr("done"), self.tr("done_msg"))

    # Metoda - zapis wynikow do pliku CSV
    def save_csv(self):
        if not self.class_confidences or not self.detections:
            messagebox.showwarning(self.tr("no_data"), self.tr("no_data_msg"))
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return

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

            messagebox.showinfo(self.tr("saved"), self.tr("saved_msg").format(save_path))
        except Exception as e:
            messagebox.showerror(self.tr("save_error"), self.tr("save_error_msg").format(e))


if __name__ == "__main__":
    root = Tk()
    app = YoloVideoApp(root)
    root.mainloop()
