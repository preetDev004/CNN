from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam


def build_model(learning_rate: float = 1e-3) -> Sequential:
    """
    Build and compile the Nvidia PilotNet steering-angle regressor.

    Output is a single unbounded value (steering angle), so the final
    Dense layer has no activation, and loss is MSE since this is
    regression, not classification.

    Args:
        learning_rate: Adam optimizer learning rate.

    Returns:
        A compiled Keras Sequential model with input shape
        (66, 200, 3) and output shape (1,).
    """
    model = Sequential([
        Conv2D(24, (5, 5), strides=(2, 2), activation="elu", input_shape=(66, 200, 3)),
        Conv2D(36, (5, 5), strides=(2, 2), activation="elu"),
        Conv2D(48, (5, 5), strides=(2, 2), activation="elu"),
        Conv2D(64, (3, 3), activation="elu"),
        Conv2D(64, (3, 3), activation="elu"),
        Flatten(),
        Dense(1164, activation="elu"),
        Dense(100, activation="elu"),
        Dense(50, activation="elu"),
        Dense(10, activation="elu"),
        Dense(1),
    ])
    model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
    return model
