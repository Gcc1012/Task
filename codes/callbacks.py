def scheduler(epoch, lr):
    if epoch < 10:
         return lr
    else:
        return lr * ops.exp(-0.1)

model = keras.models.Sequential([keras.layers.Dense(10)])
model.compile(keras.optimizers.SGD(), loss='mse')
round(model.optimizer.learning_rate, 5)

callback = keras.callbacks.LearningRateScheduler(scheduler)
callback = keras.callbacks.EarlyStopping(monitor='loss',patience=3)
history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),epochs=15, callbacks=[callback], verbose=0)
round(model.optimizer.learning_rate, 5)
len(history.history['loss'])


