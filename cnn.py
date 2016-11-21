from keras.models import Sequential

model = Sequential()

# Verificar
# - Como deve ser a entrada, pela mnist no C1D é uma array com a frequencia de 0 a 255.
#   mas é realmente 1D, uma imagem em escala de cinza, o texto praticamente desenha a imagem
#   é necessário ver como fica a 2D com RGB, uma array com uma array de tamanho 3 apenas?
# - O que é o channel e onde ele entra na leitura da imagem, pois faz parte
#   e sua sequencia pode ser alterada com o dim_ordering

model.add(Convolution2D(32, 3, 3, border_mode='valid', activation='relu', input_shape=(3, 60, 60)))
model.add(Convolution2D(16, 3, 3, border_mode='valid', activation='relu')
model.add(MaxPooling2D(pool_size=(2, 2))

model.add(Convolution2D(16, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))

model.add(Convolution2D(16, 3, 3, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))

# Com a saída deve ser passados os resultado para 2 fully connecteds para a detecção das partes
# e regressão para os pontos das junções
