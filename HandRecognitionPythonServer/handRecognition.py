import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)

    def forward(self, x):
        y = self.pool(x)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class SEBlockWithConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SEBlockWithConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.se(x)
        return x

class RecognitionNetwork(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(RecognitionNetwork, self).__init__()
        self.se_block1_1 = SEBlockWithConv(input_channels, 32, kernel_size=(3, 7), stride=(1, 1))
        self.se_block1_2 = SEBlockWithConv(32, 64, kernel_size=(3, 5), stride=(1, 1))
        self.se_block1_3 = SEBlockWithConv(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.fc_spatial1 = nn.Linear(128, 128)
        self.dropout_spatial1 = nn.Dropout(p=0.5)
        self.fc_spatial2 = nn.Linear(128, 256)
        self.dropout_spatial2 = nn.Dropout(p=0.5)

        self.se_block2_1 = SEBlockWithConv(input_channels, 32, kernel_size=(3, 7), stride=(1, 1))
        self.se_block2_2 = SEBlockWithConv(32, 64, kernel_size=(3, 5), stride=(1, 1))
        self.se_block2_3 = SEBlockWithConv(64, 128, kernel_size=(3, 3), stride=(1, 1))
        self.fc_temporal1 = nn.Linear(128, 128)
        self.dropout_temporal1 = nn.Dropout(p=0.5)
        self.fc_temporal2 = nn.Linear(128, 512)
        self.dropout_temporal2 = nn.Dropout(p=0.5)

        self.fc_fusion1 = nn.Linear(768, 256)
        self.dropout_fusion1 = nn.Dropout(p=0.5)

        self.fc_classification1 = nn.Linear(256, num_classes)

    def forward(self, spatial_input, temporal_input):
        x1 = self.se_block1_1(spatial_input)
        x1 = self.se_block1_2(x1)
        x1 = self.se_block1_3(x1)
        x1 = F.avg_pool2d(x1, x1.size()[2:]).view(x1.size()[0], -1)
        x1 = F.relu(self.fc_spatial1(x1))
        x1 = self.dropout_spatial1(x1)
        x1 = F.relu(self.fc_spatial2(x1))
        x1 = self.dropout_spatial2(x1)

        x2 = self.se_block2_1(temporal_input)
        x2 = self.se_block2_2(x2)
        x2 = self.se_block2_3(x2)
        x2 = F.avg_pool2d(x2, x2.size()[2:]).view(x2.size()[0], -1)
        x2 = F.relu(self.fc_temporal1(x2))
        x2 = self.dropout_temporal1(x2)
        x2 = F.relu(self.fc_temporal2(x2))
        x2 = self.dropout_temporal2(x2)

        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.fc_fusion1(x))
        x = self.dropout_fusion1(x)

        x = self.fc_classification1(x)
        return x

def OpticalFlow(prev_gray, frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    mask = np.zeros_like(frame)
    mask[..., 1] = 255
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
    rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
    return gray, rgb, magnitude

def calculate_frame_similarity(frame1, frame2):
    gray_frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray_frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    similarity = cv.matchTemplate(gray_frame1, gray_frame2, cv.TM_CCOEFF_NORMED)
    return similarity[0][0]

def extract_keyframes(frames, num_keyframes=5):
    num_frames = len(frames)
    keyframes = [frames[0]]
    interval = num_frames // (num_keyframes - 1)
    for i in range(1, num_keyframes - 1):
        start_index = i * interval
        end_index = min(start_index + interval, num_frames)
        selected_frame = None
        max_similarity = -1
        for j in range(start_index, end_index):
            similarity = calculate_frame_similarity(keyframes[-1], frames[j])
            if similarity > max_similarity:
                max_similarity = similarity
                selected_frame = frames[j]
        keyframes.append(selected_frame)
    keyframes.append(frames[-1])
    return keyframes

def resize_and_concatenate_keyframes(keyframes, target_size=(180, 180)):
    resized_keyframes = [cv.resize(frame, target_size) for frame in keyframes]
    concatenated_image = cv.hconcat(resized_keyframes)
    return concatenated_image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(Image.fromarray(image))

def main():

    classesNames=["right","left", "rotate up","rotate down", "downright",
              "right-down", "clockwise","counter clock","zeta","cross"]

    # Load the trained model
    model = RecognitionNetwork(input_channels=3, num_classes=10)
    model.load_state_dict(torch.load('northwestern_classifier47.pt', map_location=torch.device('cpu')))
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    ret, first_frame = cap.read()
    first_frame = cv.flip(first_frame, 1)  # Flip the frame horizontally
    prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    real_frames = []
    optical_frames = []

    movement_threshold = 1.0  # Adjust this threshold based on your requirements

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)  # Flip the frame horizontally
        real_frames.append(frame)
        prev_gray, optical_frame, magnitude = OpticalFlow(prev_gray, frame)
        optical_frames.append(optical_frame)

        if len(real_frames) >= 30:  # Use 30 frames for analysis
            average_magnitude = np.mean(magnitude)

            if average_magnitude > movement_threshold:
                real_keyframes = extract_keyframes(real_frames[-30:], num_keyframes=5)
                optical_keyframes = extract_keyframes(optical_frames[-30:], num_keyframes=5)

                real_image = resize_and_concatenate_keyframes(real_keyframes)
                optical_image = resize_and_concatenate_keyframes(optical_keyframes)

                real_tensor = preprocess_image(real_image).unsqueeze(0).to(device)
                optical_tensor = preprocess_image(optical_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(real_tensor, optical_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted_class_name = classesNames[predicted.item()]
                    print(f"Predicted Hand Movement: {predicted_class_name}")

            real_frames.clear()
            optical_frames.clear()

        cv.imshow('Webcam', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()





