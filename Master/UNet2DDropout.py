# -*- coding: utf-8 -*-
"""
Created on Sat May  1 18:39:41 2021

@author: Ådne
"""
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Dropout, Conv2DTranspose
from tensorflow.keras.models import Model

# Create model blocks
def conv_block(inputs, filters, pool=True, dropout = 0.1):
    x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout)(x)
    
    x = Conv2D(filters, kernel_size=3, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    if pool == True:
        p = MaxPool2D((2,2))(x)
        return x, p
    else:
        return x
    
def build_unet(shape, num_classes):
    inputs = Input(shape)
    
    # Encoder / Contracting path
    x1, p1 = conv_block(inputs, 16, pool=True, dropout=0.1)
    x2, p2 = conv_block(p1, 32, pool=True, dropout=0.1)
    x3, p3 = conv_block(p2, 64, pool=True, dropout=0.2)
    x4, p4 = conv_block(p3, 128, pool=True, dropout=0.2)

    # Bridge
    b1 = conv_block(p4, 256, pool=False, dropout=0.3)
    
    # Decoder / Expanding path
    u1 = Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(b1)
    c1 = Concatenate()([u1, x4])
    x5 = conv_block(c1, 128, pool=False, dropout=0.2)
    
    u2 = Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(x5)
    c2 = Concatenate()([u2, x3])
    x6 = conv_block(c2, 64   , pool=False, dropout=0.2)
    
    u3 = Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(x6)
    c3 = Concatenate()([u3, x2])
    x7 = conv_block(c3, 32, pool=False, dropout=0.1)
    
    u4 = Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(x7)
    c4 = Concatenate()([u4, x1])
    x8 = conv_block(c4, 16, pool=False, dropout=0.1)
    
    # Output layer 1 by 1 con layer
    output = Conv2D(filters=num_classes, kernel_size=1, padding="same", activation="softmax")(x8)
    
    return Model(inputs, output)

if __name__ == '__main__':
    shape = (512,512,1) # our image dimentions 
    #our masks are RGB so shape is (256,256,3)
    num_classes = 3 
    model = build_unet(shape, num_classes)
    model.summary()

