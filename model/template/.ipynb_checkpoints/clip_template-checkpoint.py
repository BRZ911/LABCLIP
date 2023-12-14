import os
import clip
import torch
from torch import nn
from PIL import Image
from torch.nn.functional import one_hot, cross_entropy


class ClipTemplate(nn.Module):
    def __init__(self, args, labels, label_desc):
        super(ClipTemplate, self).__init__()
        self.args = args
        self.dataset = args.dataset
        self.labels = labels
        self.label_desc = label_desc
        self.num_labels = len(labels)
        self.ensemble_size = args.ensemble_size
        

        self.model, self.preprocess = clip.load(args.clip_dir, device='cpu')  # use float32 instead of float16
        self.label_img = nn.Parameter(self._get_label_img())

    def _get_label_img(self):
        label_img = list()
        img_dir = os.path.join(self.args.data_dir, self.args.dataset, 'ensemble')
        for label in self.labels:
            if self.dataset == 'emotion':
                if label != 'noemo':
                    idx = self.ensemble_size
                    image = Image.open(os.path.join(img_dir, f'{label}.jpeg'))
                    label_img.append(self.preprocess(image))
            elif self.dataset == 'situation':
                if label != 'out-of-domain':
                    idx = self.ensemble_size
                    image = Image.open(os.path.join(img_dir, f'{label}.jpeg'))
                    label_img.append(self.preprocess(image))
            else:
                idx = self.ensemble_size
                image = Image.open(os.path.join(img_dir, f'{label}.jpeg'))
                label_img.append(self.preprocess(image))

        return torch.stack(label_img, dim=0).clone().detach()

    def forward(self, input_ids, attention_mask, golds=None):
        _, logits = self.model(self.label_img, input_ids)
        if self.dataset == 'emotion' or self.dataset == 'situation':
            logits = torch.mean(logits.reshape(-1, self.num_labels - 1, 1), dim=-1)
            # logits = logits[:[0,1,2,3,4,5,6,7,8,8]] 
            # logits[:,9] = 0
        else:
            logits = torch.mean(logits.reshape(-1, self.num_labels, 1), dim=-1)

        if golds is not None:
            loss = self._get_loss(logits, golds)
        else:
            loss = None

        return {
            'logits': logits,
            'loss': loss
        }

    def _get_loss(self, logits, golds):
        return cross_entropy(input=logits, target=golds)

    def predict(self, input_ids, attention_mask):
        # bsz x num_labels
        logits = self(input_ids, attention_mask)['logits']
        preds = self._decode(logits)
        onehot_preds = one_hot(preds, num_classes=self.num_labels)
        return onehot_preds

    def _decode(self, logits):
        raise NotImplementedError()

    def plm_parameters(self):
        params = list()
        for name, param in self.model.named_parameters():
            if 'learned_embedding' not in name:
                params.append(param)
        return params
