import tensorflow as tf

# Définition des entrées
x1 = tf.constant(0.5, dtype=tf.float32)
x2 = tf.constant(0.8, dtype=tf.float32)

# Définition des poids et des biais
w1 = tf.Variable(0.2, dtype=tf.float32)
w2 = tf.Variable(0.4, dtype=tf.float32)
b = tf.Variable(0.1, dtype=tf.float32)

# Calcul de la sortie du neurone
z = w1 * x1 + w2 * x2 + b
y = tf.sigmoid(z)

# Initialisation des variables
tf.compat.v1.global_variables_initializer()

# Exécution du calcul en mode eager (calcul direct sans session)
output = y.numpy()
print(output)
