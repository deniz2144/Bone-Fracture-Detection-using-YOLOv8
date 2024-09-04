import cv2
import os
import random
from ultralytics import YOLO

# Test klasörünün yolu
test_images_path = "C:\\Users\\deniz\\Desktop\\YOLOV8-BONE-FRACTURE\\test\\images"
test_labels_path = "C:\\Users\\deniz\\Desktop\\YOLOV8-BONE-FRACTURE\\test\\labels"

# Rastgele bir resim seç
random_image = random.choice(os.listdir(test_images_path))
imgtest = os.path.join(test_images_path, random_image)
imgAnot = os.path.join(test_labels_path, random_image.replace(".jpg", ".txt").replace(".png", ".txt"))

# Resmi yükle
img = cv2.imread(imgtest)
H, W, _ = img.shape

# Modeli yükle
model_path = os.path.join("C:\\Users\\deniz\\Desktop\\YOLOV8-BONE-FRACTURE\\best (7).pt")
model = YOLO(model_path)

threshold = 0.5

# Tahmin yap
results = model(img)[0]

# Tahmin sonuçlarını işleme
imgPredict = img.copy()
for box in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = box
    
    if score > threshold:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(imgPredict, (x1, y1), (x2, y2), (0, 255, 0), 1)
        class_name = results.names[int(class_id)].upper()
        cv2.putText(imgPredict, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Gerçek etiketleri yükle ve işle
imgTruth = img.copy()
if os.path.exists(imgAnot):
    with open(imgAnot, 'r') as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        values = line.strip().split()
        if len(values) < 5:
            continue  # Satırda yeterli bilgi yoksa atla
        
        class_id = int(values[0])
        x_center, y_center, w, h = map(float, values[1:])
        
        # Bounding box koordinatlarını hesapla
        x1 = int((x_center - w / 2) * W)
        y1 = int((y_center - h / 2) * H)
        x2 = int((x_center + w / 2) * W)
        y2 = int((y_center + h / 2) * H)
        
        annotations.append((x1, y1, x2, y2, class_id))

    # Gerçek etiketleri çiz
    for x1, y1, x2, y2, class_id in annotations:
        class_name = results.names[class_id].upper()
        cv2.rectangle(imgTruth, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(imgTruth, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

# Sonuçları göster
cv2.imshow("Model_Tahmin", imgPredict)
cv2.imshow("Gercek_dogruluk", imgTruth)
cv2.imshow("Orjinal_Resim", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
