import numpy as np

data = np.load(r"C:\Users\PolDiaz\Desktop\PostProcessat\test.npz")
probs = data["probabilities"]
print(probs.shape)
