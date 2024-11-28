"IntroProy2" 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
num_classes = 10
average_images = []

for i in range(num_classes):
    digit_images = digits.images[digits.target == i]
    avg_image = np.mean(digit_images, axis=0)
    average_images.append(avg_image)

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()

for i in range(num_classes):
    axes[i].imshow(average_images[i], cmap='viridis', interpolation='nearest')
    axes[i].set_title(f'Dígito {i}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()