import torch
import numpy as np

def getCosDist(codea, codeb):
    #print(codea.size())
    #print(codeb.size())
    #codeb = codeb.view(len(codeb),2048)
    matrix1_matrix2 = codea.mm(codeb.t())
    matrix1_norm = torch.sqrt((codea * codea).sum(dim=1))
    matrix2_norm = torch.sqrt((codeb * codeb).sum(dim=1))
    matrix1_norm = matrix1_norm.view(1, -1)
    matrix2_norm = matrix2_norm.view(1, -1)
    dist = matrix1_matrix2.div(matrix1_norm.t().mm(matrix2_norm))
    return dist


def get_similarity(ham_martix):
    ham_max = ham_martix.max(dim=1)[0]
    ham_min = ham_martix.min(dim=1)[0]
    filter_threshold = 0.0
    # else:
    #     filter_threshold = float(sys.argv[1])
    #     print('Debug: filter_threshold is %f'%(filter_threshold) )
    false = torch.lt(ham_max, filter_threshold)
    # false = t.lt(ham_max, 0.85)
    max_ = ham_max.repeat(ham_martix.size()[1], 1)
    min_ = ham_min.repeat(ham_martix.size()[1], 1)
    ham_ = ham_martix.t()
    similarity_matrix = ((ham_ - min_).div(max_ - min_)).t()
    similarity_matrix[false] = 0
    return similarity_matrix

if __name__ == '__main__':
    a = np.random.rand(1,5)
    b = np.random.rand(10, 5)
    a = torch.Tensor(a)
    b = torch.Tensor(b)
    cos_matrix = getCosDist(a, b)
    similarity = get_similarity((cos_matrix))
    print('Done')
