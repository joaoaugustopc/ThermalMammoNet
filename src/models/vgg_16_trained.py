from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Lambda

class Vgg_16_pre_trained:

    def __init__(self, input_shape=(224, 224, 3), num_classes=1, learning_rate=0.0001):
        # A mixed_precision.set_global_policy() deve ser chamada uma única vez
        # no início do script, não dentro da classe ou função que cria o modelo.
        # Mas para o seu código, mantê-lo aqui funciona, apenas saiba que é
        # uma configuração global.
        mixed_precision.set_global_policy('mixed_float16')
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        # 1. Carrega o modelo VGG16 pré-treinado
        # Usar input_shape em vez de input_tensor é a forma mais direta
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
    
        # 2. Congela as camadas convolucionais
        for layer in base_model.layers:
            layer.trainable = False
        
        # 3. Adiciona suas camadas personalizadas (classifier head)
        x = base_model.output
        
        # Opcional, mas boa prática: adicione BatchNormalization aqui.
        # É mais eficaz normalizar a saída do extrator de features antes
        # de alimentá-la às camadas totalmente conectadas.
        x = BatchNormalization()(x)
        
        # Camadas fully connected personalizadas
        x = Flatten()(x)
        x = Dense(4096, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        
        # Camada de saída com o dtype 'float32' para evitar underflow na sigmoid
        predictions = Dense(self.num_classes, activation='sigmoid', dtype='float32')(x)
        
        # 4. Cria o modelo final
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # 5. Configura o otimizador com mixed precision
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        
        # 6. Compila o modelo
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

# Exemplo de uso (fora da classe, no seu script principal)
# vgg_model = Vgg_16_pre_trained()
# print(vgg_model.model.summary())