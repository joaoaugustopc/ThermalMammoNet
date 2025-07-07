from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Lambda

class Vgg_16_pre_trained:

    def __init__(self, input_shape=(224, 224, 1), num_classes=1, learning_rate=0.0001):
        mixed_precision.set_global_policy('mixed_float16')
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        # Cria uma camada para transformar 1 canal em 3 canais (repetindo os valores)
        input_layer = keras.layers.Input(shape=self.input_shape)  # (224, 224, 1)
        
        # Repete o canal único 3 vezes
        rgb_input = Lambda(lambda x: tf.repeat(x, 3, axis=-1))(input_layer)
        
        # Carrega a VGG16 pré-treinada
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_tensor=rgb_input
        )
    
        # Congela as camadas convolucionais - pode congelar só algumas
        for layer in base_model.layers:
            layer.trainable = False
        
        # Adiciona suas camadas personalizadas
        x = base_model.output
        
        # Adiciona BatchNormalization após cada bloco (como no seu código original)
        for layer in base_model.layers:
            if 'block' in layer.name and 'conv' in layer.name and layer.name.endswith('conv1'):
                x = BatchNormalization()(x)
        
        # Camadas fully connected personalizadas
        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        predictions = Dense(self.num_classes, activation='sigmoid', dtype='float32')(x)
        
        # Cria o modelo final
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Configura o otimizador com mixed precision
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model