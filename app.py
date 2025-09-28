from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Load image
image = Image.open("sample.jpg")
image = image.resize((150, 150))  # resize to make processing faster
image_np = np.array(image)
image_np = image_np.reshape((-1, 3))  # flatten pixels

# Extract main colors
kmeans = KMeans(n_clusters=5)
kmeans.fit(image_np)
colors = kmeans.cluster_centers_.astype(int)

# Show colors
plt.figure(figsize=(8, 2))
plt.imshow([colors])
plt.axis("off")
plt.show()
