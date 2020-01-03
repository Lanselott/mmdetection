import torch
import dsntnn

from IPython import embed

def coord_reg(feature_map):
    # 3. Normalize the heatmaps
    heatmaps = dsntnn.flat_softmax(feature_map)
    # 4. Calculate the coordinates
    coords = dsntnn.dsnt(heatmaps)
    embed()


if __name__ == "__main__":
    features = torch.rand([1, 1, 1, 10])
    features[0,..., 2] = 100
    coord_reg(features)
    pass