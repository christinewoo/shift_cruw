import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import generalized_box_iou as giou


class SHIFT_XYZ(nn.Module):    
    def __init__(self, K, img_size):
        super(SHIFT_XYZ, self).__init__()
        # One hidden layer
        self.lin_one = nn.Linear(3, 3) # input_size, hidden_neurons
        self.lin_two = nn.Linear(3, 3) # hidden_neurons, output_size
        # Set layer attributes?
        self.layer_in = None
        self.act = None
        self.layer_out = None
        self.K = K
        self.img_size = torch.tensor(img_size)
    
    # x is (N, 7), 7: x, y, z, rot_y, h, w, l
    def forward(self, bbox):
        pos = bbox[:, :3] # (N, 3)
        ry = bbox[:, 3] # (N, )
        dims = bbox[:, 4:7] # (N, 3)
        
        self.layer_in = self.lin_one(pos)
        self.act = torch.sigmoid(self.layer_in)
        self.layer_out = self.lin_two(self.act)
        y_pred = torch.sigmoid(self.lin_two(self.act))
        return y_pred

def get_bev_corners(y_pred, y): #[x_min, y_min, x_max, y_max] preds, target
    out = -1 * torch.mean(y * torch.log(y_pred) + (1 - y) * torch.log(1 - y_pred))
    return out

    

if __name__ == '__main__':
    model = SHIFT_XYZ(3, 3, 3)
    gt_bev = []
    dt_bev = []
    loss = giou(gt_bev, dt_bev, reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)