from include.imports import *
from tensorflow.keras import mixed_precision
class BottleneckResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation_fn = keras.activations.get(activation)
        self.strides = strides
        self.filters = filters
        self.filters_expanded = filters * 4  # Expansion factor of 4

        # Layers will be initialized in build()
        self.main_layers = []
        self.skip_layers = []

    def build(self, input_shape):
        # Input channels
        input_channels = input_shape[-1]

        # Main path
        self.main_layers = [
            keras.layers.Conv2D(self.filters, 1, strides=self.strides,
                                padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.0005)),
            keras.layers.BatchNormalization(),
            self.activation_fn,
            keras.layers.Conv2D(self.filters, 3, strides=1,
                                padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.0005)),
            keras.layers.BatchNormalization(),
            self.activation_fn,
            keras.layers.Conv2D(self.filters_expanded, 1, strides=1,
                                padding='same', use_bias=False, kernel_regularizer=keras.regularizers.l2(0.0005)),
            keras.layers.BatchNormalization()
        ]

        # Shortcut path
        if self.strides != 1 or input_channels != self.filters_expanded:
            self.skip_layers = [
                keras.layers.Conv2D(self.filters_expanded, 1, strides=self.strides,
                                    padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        # Main path
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)

        # Shortcut path
        skip_Z = inputs
        if self.skip_layers:
            for layer in self.skip_layers:
                skip_Z = layer(skip_Z)

        # Add and activate
        Z = tf.cast(Z, dtype=skip_Z.dtype)
        output = self.activation_fn(Z + skip_Z)
        return output

def ResNet101():
    # Set mixed precision policy if applicable
    mixed_precision.set_global_policy('mixed_float16')

    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 1],
                                  padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'))

    prev_filters_expanded = 64  # Number of filters after MaxPool

    filters_list = [64, 128, 256, 512]
    blocks_per_stage = [3, 4, 23, 3]

    for filters, blocks in zip(filters_list, blocks_per_stage):
        for block in range(blocks):
            strides = 1
            if block == 0 and prev_filters_expanded != filters * 4:
                # Downsample at the first block of each stage except the first
                strides = 2
            model.add(BottleneckResidualUnit(filters, strides=strides))
        prev_filters_expanded = filters * 4  # Update for next stage

    model.add(keras.layers.GlobalAvgPool2D())
    model.add(keras.layers.Dense(1, activation='sigmoid', dtype='float32'))

    opt = keras.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model
