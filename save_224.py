from medmnist import PneumoniaMNIST
train = PneumoniaMNIST(split="train", download=False, size=224)

normal_img = None
pneumonia_img = None
for i in range(len(train)):
    img, label = train[i]
    if label[0] == 0 and normal_img is None:
        normal_img = img
    if label[0] == 1 and pneumonia_img is None:
        pneumonia_img = img
    if normal_img is not None and pneumonia_img is not None:
        break

normal_img.save("samples/pneumonia_mnist_normal_224.png")
pneumonia_img.save("samples/pneumonia_mnist_pneumonia_224.png")
print("Saved.")
