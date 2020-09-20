import torch
from torch.autograd import Variable
import numpy as np
from input_data import seg_num
from utils import xy_get_char

class P300:

    def __init__(self, model):
        self.model = model

    def _get_one_target(self, light, event):
        round_num = 5
        round_len = 12
        votes = np.zeros(12).astype(int)
        for i in range(round_num):
            s = round_len * i
            e = round_len * (i+1)
            li = light[s: e]
            ev = event[s: e]
            li_ev = ev[li==1] - 1
            votes[li_ev] += 1
            # print(li, li.sum())

        x = np.argmax(votes)
        votes[x] = -1
        y = np.argmax(votes)
        ret = xy_get_char(x+1, y+1)
        return ret

    def get_target(self, data, event):
        input = Variable(torch.from_numpy(data).cuda(0))
        pred = self.model(input).cpu().detach().numpy()
        light = np.argmax(pred, axis=1)
        
        targets = []
        num_target = data.shape[0] // seg_num
        for i in range(num_target):
            this_light = light[i*seg_num: (i+1)*seg_num]
            this_event = event[i*seg_num: (i+1)*seg_num]
            this_target = self._get_one_target(this_light, this_event)
            targets.append(this_target)
        
        # import pdb; pdb.set_trace()
        return targets