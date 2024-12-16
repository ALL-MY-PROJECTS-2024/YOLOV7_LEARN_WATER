import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras.layers import Conv2D, ConvLSTM2D, UpSampling2D, MaxPooling2D, Layer, BatchNormalization


class PredNet(Layer):
    def __init__(self, stack_sizes, R_stack_sizes, A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1.0, error_activation='relu', A_activation='relu',
                 LSTM_activation='tanh', LSTM_inner_activation='hard_sigmoid',
                 output_mode='error', extrap_start_time=None, data_format='channels_last', **kwargs):
        super(PredNet, self).__init__(**kwargs)

        self.stack_sizes = stack_sizes
        self.nb_layers = len(stack_sizes)
        self.R_stack_sizes = R_stack_sizes
        self.A_filt_sizes = A_filt_sizes
        self.Ahat_filt_sizes = Ahat_filt_sizes
        self.R_filt_sizes = R_filt_sizes
        self.pixel_max = pixel_max

        # Activations
        self.error_activation = activations.get(error_activation)
        self.A_activation = activations.get(A_activation)
        self.LSTM_activation = activations.get(LSTM_activation)
        self.LSTM_inner_activation = activations.get(LSTM_inner_activation)

        self.output_mode = output_mode
        self.extrap_start_time = extrap_start_time
        self.data_format = data_format

        # Initialize layers for ConvLSTM, Conv2D, BatchNormalization
        self.conv_layers = {'ahat': [], 'a': [], 'lstm': []}
        self.batch_norm_layers = []  # Store BatchNormalization layers

        for l in range(self.nb_layers):
            self.conv_layers['ahat'].append(Conv2D(stack_sizes[l], Ahat_filt_sizes[l], activation=self.A_activation, padding='same'))
            if l < self.nb_layers - 1:
                self.conv_layers['a'].append(Conv2D(stack_sizes[l + 1], A_filt_sizes[l], activation=self.A_activation, padding='same'))
            self.conv_layers['lstm'].append(ConvLSTM2D(R_stack_sizes[l], R_filt_sizes[l], padding='same', return_sequences=False))
            self.batch_norm_layers.append(BatchNormalization())  # Initialize BatchNormalization for each layer

        self.upsample = UpSampling2D()
        self.pool = MaxPooling2D()

    def call(self, inputs, states=None):
        # Initialize states
        batch_size, time_steps, height, width, channels = inputs.shape
        r = [tf.zeros((batch_size, height // (2 ** l), width // (2 ** l), self.R_stack_sizes[l])) for l in range(self.nb_layers)]
        c = [tf.zeros_like(r[l]) for l in range(self.nb_layers)]

        outputs = []
        for t in range(time_steps):
            a = inputs[:, t]
            e = []

            # Feedforward prediction and error computation
            for l in range(self.nb_layers):
                ahat = self.conv_layers['ahat'][l](r[l])
                ahat = tf.clip_by_value(ahat, 0, self.pixel_max)
                if l == 0:
                    frame_prediction = ahat
                e_up = self.error_activation(ahat - a)
                e_down = self.error_activation(a - ahat)
                e.append(tf.concat([e_up, e_down], axis=-1))

                if l < self.nb_layers - 1:
                    a = self.conv_layers['a'][l](e[l])
                    a = self.pool(a)

            # Update R and C using ConvLSTM
            for l in reversed(range(self.nb_layers)):
                inputs_lstm = tf.concat([r[l], e[l]], axis=-1)
                c[l], r[l] = self.conv_layers['lstm'][l](inputs_lstm, initial_state=[c[l], r[l]])
                r[l] = self.batch_norm_layers[l](r[l])  # Use pre-initialized BatchNormalization

            outputs.append(frame_prediction)

        return tf.stack(outputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape
