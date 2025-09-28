
img = cv2.imread("poster.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pixels = img.reshape(-1, 3)
kmeans = KMeans(n_clusters=5)
kmeans.fit(pixels)

colors = kmeans.cluster_centers_.astype(int)
plt.figure(figsize=(8,2))
for i, color in enumerate(colors):
    plt.subplot(1, 5, i+1)
    plt.imshow([[color/255]])
    plt.axis("off")
plt.show()
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])

hex_colors = [rgb_to_hex(c) for c in colors]
print(hex_colors)
