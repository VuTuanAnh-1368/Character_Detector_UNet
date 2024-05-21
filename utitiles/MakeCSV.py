import os
import cv2

# Đường dẫn đến thư mục chứa các hình ảnh và nhãn
image_folder = "train-val/wb_localization_dataset/images/val"
label_folder = "train-val/wb_localization_dataset/labels/val"

# Đường dẫn đến file CSV đầu ra
output_csv = "val_noScale.csv"

class MakeCSV:
    def __init__(self, image_folder, label_folder, output_csv):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.output_csv = output_csv

    # Hàm đọc nhãn từ file
    def read_labels(label_file):
        with open(label_file, 'r') as file:
            lines = file.readlines()
            labels = []
            for line in lines:
                label = line.strip().split()
                labels.append(label)
            return labels

    # Hàm tạo dòng dữ liệu cho file CSV
    def create_csv_line(image_name, labels, image_width, image_height):
        line = os.path.splitext(image_name)[0]
        label_parts = []
        for label in labels:
            x_center = float(label[1]) * image_width
            y_center = float(label[2]) * image_height
            width = float(label[3]) * image_width
            height = float(label[4]) * image_height
            # Chuyển đổi tọa độ YOLO thành tọa độ góc trái trên và kích thước
            x_topLeft = int(x_center - width//2)
            y_topLeft = int(y_center - height//2)
            width = int(width)
            height = int(height)
            label_parts.append(f"{label[0]} {x_topLeft} {y_topLeft} {width} {height}")
        line += "," + " ".join(label_parts)
        return line + "\n"

    # Mở file CSV để ghi
    with open(output_csv, 'w') as csv_file:
        # Ghi dòng tiêu đề
        csv_file.write("image_id,labels\n")
        # Lặp qua tất cả các hình ảnh trong thư mục hình ảnh
        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            # Đọc hình ảnh để lấy kích thước
            image = cv2.imread(image_path)
            image_height, image_width, _ = image.shape
            # Tìm file nhãn tương ứng
            label_name = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(label_folder, label_name)
            # Đọc nhãn từ file
            if os.path.exists(label_path):
                labels = read_labels(label_path)
                # Tạo dòng dữ liệu cho file CSV
                csv_line = create_csv_line(image_name, labels, image_width, image_height)
                # Ghi vào file CSV
                csv_file.write(csv_line)