import torch

def dot_product(query, key):
    ''' 
    query: (B, Head, Q, C) or (..., Q, C)
    key:   (B, Head, K, C) or (..., K, C)
    '''
    return query @ key.transpose(-1, -2)

def neg_l2_norm(query, key):
    '''
    query: (B, Head, Q, C) or (..., Q, C)
    key:   (B, Head, K, C) or (..., K, C)
    '''

    diff = query.unsqueeze(-2) - key.unsqueeze(-3)
    dist = torch.norm(diff, dim=-1)
    return - dist
def inverse_l2_norm(query, key, eps=1e-6):
    '''
    query: (B, Head, Q, C) or (..., Q, C)
    key:   (B, Head, K, C) or (..., K, C)
    '''
    diff = query.unsqueeze(-2) - key.unsqueeze(-3)
    dist = torch.norm(diff, dim=-1)
    return 1.0 / (dist + eps)

def neg_log_l2_norm(query, key, eps=1e-6):
    '''
    query: (B, Head, Q, C) or (..., Q, C)
    key:   (B, Head, K, C) or (..., K, C)
    '''
    diff = query.unsqueeze(-2) - key.unsqueeze(-3)
    dist = torch.norm(diff, dim=-1)
    return - torch.log(dist + eps)

def cosine_similarity(query, key):
    query = query / query.norm(dim=-1, keepdim=True)
    key = key / key.norm(dim=-1, keepdim=True)
    return query @ key.transpose(-1, -2)


def l2_norm_onehot(query, key):
    '''
    query: (B, Head, Q, C) or (..., Q, C)
    key:   (B, Head, K, C) or (..., K, C)
    '''
    diff = query.unsqueeze(-2) - key.unsqueeze(-3)
    dist = torch.norm(diff, dim=-1)
    pass

COST_METRIC_FN ={
    "dot_product": dot_product,
    "neg_l2": neg_l2_norm,
    "inverse_l2": inverse_l2_norm,
    "neg_log_l2": neg_log_l2_norm,
    "cosine_similarity": cosine_similarity,
}

