import sys
sys.path.append(r"C:\\Users\\jwg13\\Downloads\\SEG\\SEG_LOCAL_NEW\\yolov7-segmentation\\prednet\\")


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from prednet import PredNet


def load_video_frames(video_path, frame_count=20, resize=(128, 128)):
    """
    Load video frames, resize to target size, and normalize.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < frame_count:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), resize) / 255.0
        frames.append(frame)
    cap.release()
    return np.array(frames)


def create_prednet_model(input_shape, stack_sizes, R_stack_sizes):
    """
    Create and compile the PredNet model.
    """
    inputs = Input(shape=(None, *input_shape))  # Shape: (timesteps, height, width, channels)
    prednet = PredNet(
        stack_sizes=stack_sizes,
        R_stack_sizes=R_stack_sizes,
        A_filt_sizes=(3, 3),
        Ahat_filt_sizes=(3, 3, 3),
        R_filt_sizes=(3, 3, 3),
        output_mode='prediction',
        data_format='channels_last'
    )
    outputs = prednet(inputs)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mae')
    return model


def train_and_predict(video_frames, stack_sizes, R_stack_sizes, prediction_steps=10):
    """
    Train the PredNet model on video frames and perform prediction.
    """
    # Prepare input data
    input_frames = []
    target_frames = []
    for i in range(len(video_frames) - prediction_steps):
        input_frames.append(video_frames[i:i + prediction_steps])  # Input sequence
        target_frames.append(video_frames[i + prediction_steps])   # Target frame

    X = np.array(input_frames).reshape(-1, prediction_steps, video_frames.shape[1], video_frames.shape[2], 1)
    y = np.array(target_frames).reshape(-1, video_frames.shape[1], video_frames.shape[2], 1)

    # Create the model
    model = create_prednet_model(X.shape[2:], stack_sizes, R_stack_sizes)

    # Train the model
    model.fit(X, y, epochs=5, batch_size=4, verbose=1)

    # Predict the next frame
    predicted_frame = model.predict(X[-1:])
    predicted_frame = predicted_frame[0, :, :, 0] * 255.0  # Rescale for visualization
    return predicted_frame


if __name__ == "__main__":
    # Parameters
    video_path = "output.avi"  # Replace with your video path
    frame_count = 20
    prediction_steps = 5
    resize = (128, 128)
    stack_sizes = (1, 32, 64, 128)
    R_stack_sizes = (32, 64, 128, 256)

    # Load video frames
    video_frames = load_video_frames(video_path, frame_count, resize)

    # Train the model and predict the next frame
    predicted_frame = train_and_predict(video_frames, stack_sizes, R_stack_sizes, prediction_steps)

    # Display the predicted frame
    import matplotlib.pyplot as plt
    plt.imshow(predicted_frame, cmap='gray')
    plt.title("Predicted Frame")
    plt.show()
