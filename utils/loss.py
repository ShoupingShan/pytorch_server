import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SoftCrossEntropyLoss(nn.NLLLoss):
    def __init__(self, label_smoothing=0, num_classes=45, **kwargs):
        assert label_smoothing >= 0 and label_smoothing <= 1
        super(SoftCrossEntropyLoss, self).__init__(**kwargs)
        self.confidence = 1 - label_smoothing
        self.other      = label_smoothing * 1.0 / (num_classes - 1)
        self.criterion  = nn.KLDivLoss(reduction='batchmean')
        print('using soft celoss!!!, label_smoothing = ', label_smoothing)

    def forward(self, input, target):
        one_hot = torch.zeros_like(input)
        one_hot.fill_(self.other)
        one_hot.scatter_(1, target.unsqueeze(1).long(), self.confidence)
        input   = F.log_softmax(input, 1)
        return self.criterion(input, one_hot)

class FocalLoss_weight(nn.NLLLoss):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num=54, alpha=None, gamma=2, size_average=True):
        super(FocalLoss_weight, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.) #onehot
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        temp =ids.data.view(-1)
        alpha = self.alpha[temp]
        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha

    def forward(self, input, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = self.alpha * (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight)
        return loss

class OHEM(torch.nn.NLLLoss):
    """ Online hard example mining.
    Needs input from nn.LogSotmax() """
                                                                                   
    def __init__(self, ratio=0.5):
        super(OHEM, self).__init__(None, True)
        self.ratio = ratio
                                                                                   
    def forward(self, inputs, targets, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_batch = inputs.size(0)
        num_class = inputs.size(1)
        num_bad = int(self.ratio * num_batch)
        inputs_clone = inputs.clone()
        batch_losses = Variable(torch.zeros(num_batch))
        if inputs.is_cuda and not batch_losses.is_cuda:
            batch_losses = batch_losses.cuda()
        for idx, label in enumerate(targets.data):
            batch_losses[idx] = -inputs_clone.data[idx, label]
        #loss_incs = -x_.sum(1)
        _, idxs = batch_losses.topk(num_bad)
        input_bad = inputs.index_select(0, idxs)
        target_index = targets.index_select(0, idxs)
        # target_bad = inputs.data.new(num_bad, num_class).fill_(0)
        # target_bad = Variable(target_bad)
        # target_bad.scatter_(1, target_index.view(num_bad, -1), 1.)
        # target_bad = torch.tensor(target_bad,  dtype=torch.long)
        loss = torch.nn.CrossEntropyLoss()
        return loss(input_bad, target_index)

if __name__ == '__main__':
    outputs = torch.tensor([[0.45, 0.7, 0.21, -0.2], [0.45, 0.1, 0.9, -0.2], [0.1, 0.7, -0.31, -0.2]])
    targets = torch.tensor([1, 2, 0])
    loss = FocalLoss(gamma=2).forward(outputs, targets)
    print(loss)
    loss = OHEM(0.9).forward(outputs, targets)
    print(loss)