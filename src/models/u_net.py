from include.imports import *


def encoder_block(inputs, num_filters): 

	x = tf.keras.layers.Conv2D(num_filters, 
							3, 
							padding = 'valid')(inputs) 
	x = tf.keras.layers.Activation('relu')(x) 
	
	x = tf.keras.layers.Conv2D(num_filters, 
							3, 
							padding = 'valid')(x) 
	x = tf.keras.layers.Activation('relu')(x) 

	x = tf.keras.layers.MaxPool2D(pool_size = (2, 2), 
								strides = 2)(x) 
	
	return x

def decoder_block(inputs, skip_features, num_filters): 

	x = tf.keras.layers.Conv2DTranspose(num_filters, 
										(2, 2), 
										strides = 2, 
										padding = 'valid')(inputs) 
	
	# Redimensionar skip_features para o tamanho de x
	skip_features = tf.image.resize(skip_features, 
									size = (x.shape[1], 
											x.shape[2])) 
	
	# Concatenar skip_features e x (axis = -1) -> Concatenação de canais dos mapas de características
	x = tf.keras.layers.Concatenate()([x, skip_features]) 
	
	x = tf.keras.layers.Conv2D(num_filters, 
							3, 
							padding = 'valid')(x) 
	x = tf.keras.layers.Activation('relu')(x) 

	x = tf.keras.layers.Conv2D(num_filters, 3, padding = 'valid')(x) 
	x = tf.keras.layers.Activation('relu')(x) 
	
	return x

# Unet code 
def unet_model(input_shape = (480, 640, 1), num_classes = 2): 
	inputs = tf.keras.layers.Input(input_shape) 
	
	# Contracting Path (Encoder)
	s1 = encoder_block(inputs, 64) 
	s2 = encoder_block(s1, 128) 
	s3 = encoder_block(s2, 256) 
	s4 = encoder_block(s3, 512) 
	
	# Bottleneck 
	b1 = tf.keras.layers.Conv2D(1024, 3, padding = 'valid')(s4) 
	b1 = tf.keras.layers.Activation('relu')(b1) 
	b1 = tf.keras.layers.Conv2D(1024, 3, padding = 'valid')(b1) 
	b1 = tf.keras.layers.Activation('relu')(b1) 
	
	# Expansive Path (Decoder)
	s5 = decoder_block(b1, s4, 512) 
	s6 = decoder_block(s5, s3, 256) 
	s7 = decoder_block(s6, s2, 128) 
	s8 = decoder_block(s7, s1, 64) 
	
	# Output 
	outputs = tf.keras.layers.Conv2D(num_classes, 
									1, 
									padding = 'valid', 
									activation = 'sigmoid')(s8) 
	
	model = tf.keras.models.Model(inputs = inputs, 
								outputs = outputs, 
								name = 'U-Net') 
	return model 

