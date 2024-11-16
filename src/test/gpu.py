import tensorflow as tf

print("Num logical CPUs Available: ", len(tf.config.list_logical_devices("CPU")))
print("Num physical CPUs Available: ", len(tf.config.list_physical_devices("CPU")))
print("Num logical GPUs Available: ", len(tf.config.list_logical_devices("GPU")))
print("Num physical GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
