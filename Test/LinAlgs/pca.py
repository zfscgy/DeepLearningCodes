from CoreLinAlgs.Transformations import PCA
import numpy as np

xs = np.random.uniform(0, 10, 30)
ys = np.random.uniform(0, 20, 30)
zs = 3 * xs + ys + np.random.normal(0, 1, 30)
data = np.stack([xs, ys, zs]).T
print(data)
pca = PCA(data)
pca.pca()
vs, avg = pca.get_portion(0.9)
print(vs)
compressed_data = np.matmul(data - avg, vs)
recovered_data = np.matmul(compressed_data, vs.T) + avg
print(recovered_data)