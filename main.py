import os, shutil
from ultralytics import YOLO
import cv2

# Memastikan YOLOv8 model diunduh atau dilatih terlebih dahulu
MODEL_PATH = "yolov8n.pt" # Menggunakan model pretrained dari YOLOv8
INPUT_FOLDER = "input_images" #Folder untuk gambar input
OUTPUT_FOLDER = "output"  #Folder untuk menyimpan hasil

def detection_object():
    # Load Model YOLOv8
    model = YOLO(MODEL_PATH)

    os.makedirs(OUTPUT_FOLDER)
    if os.path.exists(OUTPUT_FOLDER): #cek apakah didalam folder terdapat file atau tidak.
        shutil.rmtree(OUTPUT_FOLDER) #jikalau ya, maka hapus folder yg tersebut agar tidak dapat duplikasi disaat sudah adanya proses yg dijalankan sebelumnya
        os.makedirs(OUTPUT_FOLDER) #jikalau tidak | jikalau folder seblumnya ada dan sudah dihapus, maka create yang baru

    #Iterasi melalui semua gambar di folder input
    for image_name in os.listdir(INPUT_FOLDER):
        input_path = os.path.join(INPUT_FOLDER, image_name)
        output_path = os.path.join(OUTPUT_FOLDER, image_name)

        #memastikan hanya file gambar yang diproses
        if not input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Deteksi objek gambar
        results = model.predict(input_path, conf=0.5, save=False)

        #simpan hasil deteksi ke file baru
        for result in results:
            #ini adalah cara pertama untuk menyimpan hasil gambar yg sudah dideteksi, tetapi tidak divalidasi dan semua gambar mau ada detectionnya atau tidak maka akan masuk ke folder output.

            # annotated_image = result.plot(result) #Gambar hasil deteksi
            # cv2.imwrite(output_path, annotated_image)
            # print(f"Hasil deteksi disimpan ke: {image_name}")

            filtered_boxes = []
            for box in result.boxes:
                class_id = int(box.cls)
                if model.names[class_id] == "bottle":
                    filtered_boxes.append(box)
            if not filtered_boxes:
                print(f"Tidak ada botol yang terdeteksi di: {image_name}")
                continue

            # print(result)
            # exit()
            annotated_image = result.plot(boxes=filtered_boxes) # Gambar hasil deteksi
            cv2.imwrite(output_path, annotated_image)
            print(f"Hasil deteksi disimpan ke: {image_name}")


if __name__ == "__main__":
    detection_object()