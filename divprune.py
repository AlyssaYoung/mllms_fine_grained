import torch

def pairwise_cosine_similarity(matrix):
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    return cosine_similarity

def DivPrune(visual_feature_vectors, image_feature_length, cosine_matrix=None, threshold_ratio=0.1):            
    threshold_terms = int(round(threshold_ratio*image_feature_length))
    if cosine_matrix is None:
        cosine_matrix = 1.0 - (pairwise_cosine_similarity(visual_feature_vectors))

    s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(threshold_terms):
        if i==0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s,0,torch.arange(0,i,device=cosine_matrix.device)))

        if i==0:
            scores = torch.topk(m2, 2,dim=0,largest=False).values[1,:] #for distance
        else:
            scores = torch.min(m2, dim=0).values #for distance 

        phrase_to_add_idx = torch.argmax(scores)
        s[i] = phrase_to_add_idx
    return s, cosine_matrix

def greedy_diverse_subset(dist_matrix, k):
    N = dist_matrix.size(0)
    selected = torch.empty(k, dtype=torch.long, device=dist_matrix.device)

    for i in range(k):
        if i == 0:
            scores = torch.topk(dist_matrix, 2, dim=0, largest=True).values[1]
        else:
            dist_to_selected = dist_matrix[selected[:i]]
            scores = torch.min(dist_to_selected, dim=0).values

        scores[selected[:i]] = -float('inf')
        selected[i] = torch.argmax(scores)

    return torch.sort(selected).values

def DivPrune2D(visual_feature_vectors_2d, threshold_ratio=0.5):
    H, W, D = visual_feature_vectors_2d.shape
    device = visual_feature_vectors_2d.device

    # --- Row Selection ---
    row_vectors = visual_feature_vectors_2d.reshape(H, -1)  # [H, W*D]
    row_sim = pairwise_cosine_similarity(row_vectors)
    row_dist = 1.0 - row_sim  
    row_k = int(round(threshold_ratio * H))
    selected_rows = greedy_diverse_subset(row_dist, row_k)

    # --- Column Selection ---
    col_vectors = visual_feature_vectors_2d.permute(1, 0, 2).reshape(W, -1)  # [W, H*D]
    col_sim = pairwise_cosine_similarity(col_vectors)
    col_dist = 1.0 - col_sim 
    col_k = int(round(threshold_ratio * W))
    selected_cols = greedy_diverse_subset(col_dist, col_k)

    return selected_rows, selected_cols
