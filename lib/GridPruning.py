import numpy as np
import torch

def index_select(inputs, indices, dim):
    outputs = inputs.index_select(dim, indices.view(-1))
    if indices.dim() > 1:
        if dim < 0:
            dim += inputs.dim()
        output_shape = inputs.shape[:dim] + indices.shape + inputs.shape[dim + 1 :]
        outputs = outputs.view(*output_shape)
    return outputs

def tetra_subdivide(cur_verts_features, next_parents, device,any=True):
    next_parents = next_parents.to(device)
    cur_verts_features = cur_verts_features.to(device)
    mask = next_parents == -1
    next_parents[mask] = len(cur_verts_features[0])
    cur_verts_features = torch.nn.functional.pad(cur_verts_features, (0, 0, 0, 1, 0, 0), value=0)
    neighbors = index_select(cur_verts_features, next_parents, dim=1)
    if any:
        return neighbors.mean(2)
    else:
        num_nans = torch.sum(torch.isnan(neighbors),2)
        neighbors[0,num_nans.squeeze()<5,:,:] = 0
        return neighbors.mean(2)


def mark_neighbors_as_deleted(neighbors,mask):
    #1 step mark all neighbors as deleted:
    delete_me = mask.squeeze(1) == -1
    keep_me = mask.squeeze(1) != -1
    neighbors[delete_me,0] = -1 # mark all neighbors to be deleted as -1

    for i in range(len(neighbors)):
        c_neighbors = neighbors[i]
        if c_neighbors[0] == -1 : #skip if marked as deleted
            continue

        for j in range(1,len(c_neighbors)):
            if c_neighbors[j] != -1: # it has a neighbor
                if neighbors[c_neighbors[j],0] == -1: # is the neighbor deleted?
                    neighbors[i,j] = -1 # remove it from the list of neighbors
    # 2 step remove all deleted neighbors:
    return neighbors[keep_me,:]


def _probe_signals(mask,parents_global,num_verts,cuda_device):
    parents = []
    for i in range(len(parents_global)):
        parents.append(parents_global[i].clone())

    indices_to_delete = []
    mask = mask.cuda()

    for i in range(num_verts):
        #create lowest dim cube
        low_d_cube = torch.zeros((1,num_verts, 1))

        low_d_cube[0,i] = torch.nan
        # propagate signal
        for j in range(0,len(parents)):
            low_d_cube = tetra_subdivide(low_d_cube,parents[j],cuda_device)

        #lookup all nans in highest dim
        high_res = torch.nan_to_num(low_d_cube,-1)
        idx = torch.where(high_res[0]==-1)
        uniq = torch.unique(mask[idx])

        if len(uniq)==1 and uniq[0]==-1: # if all values in the mask are -1, its save to delete
            indices_to_delete.append(i)

    return indices_to_delete

def get_actual_masks(mask,upsample,downsample,cuda_device,any):
    """
       :param mask:  N' x 1 , -1 marks values to be removed 0 marks data
       :return: list of masks for every level containing save to remove vertices.
       """
    remove_me = []

    num_verts,_ = downsample[0].shape

    delete_me = _probe_signals(mask,upsample,num_verts,cuda_device)

    low_d_cube = torch.zeros((1,num_verts, 1))
    low_d_cube[0,delete_me] = torch.nan

    remove_me.append(torch.nan_to_num(low_d_cube,-1)[0])
    for j in range(0, len(upsample)):
        low_d_cube = tetra_subdivide(low_d_cube, upsample[j].clone(), "cpu",any=any)
        remove_me.append(torch.nan_to_num(low_d_cube,-1)[0])

    return remove_me


def mapping_index_function(keep_me,vert_indices):
    #creates an array of the original length of vertices -
    #containing the new index positions (after cropping the data) of the vertices
    mapping_function = torch.full_like(keep_me,1e12,dtype=torch.long)
    mapping_function[vert_indices] = torch.arange(0, len(vert_indices))
    return mapping_function


def crop_all_data(vert_list,tetra_list,neighbor_list,upsample_list,downsample_list,mask_list):#
    #1 Step remove all unused vertices:

    c = 0
    cropped_vertics = []
    index_mapping_list = []### helper list to update the indices in the following loops
    for i in range(0,len(vert_list)):

        idx = mask_list[c].squeeze(1) != -1 ## keep all non -1
        vert_indices = torch.arange(0,len(vert_list[i]))

        index_mapping_list.append(vert_indices[idx])
        cropped_vertics.append(vert_list[i][idx,:])
        c += 1

    #
    #2 Step update tetra list
    c = 0
    cropped_tetrahedra = []
    for i in range(0,len(tetra_list)):
        keep_me = mask_list[c].squeeze(1) !=-1 ## keep all non -1
        vert_indices = index_mapping_list[c]
        mapping_function = mapping_index_function(keep_me,vert_indices)
        current_tets = tetra_list[i]
        mask = torch.isin(current_tets, vert_indices).all(dim=1) # remove all tetrahedra where not ALL indices are in the too keep list
        leftover_tets = current_tets[mask].long()
        leftover_tets = mapping_function[leftover_tets] # map all tetrahedra to their new index
        cropped_tetrahedra.append(leftover_tets)
        c+= 1


    #3 step update neighbors
    c = 0
    cropped_neighbors = []
    for i in range(0, len(neighbor_list)):
        keep_me = mask_list[c].squeeze(1) != -1  ## keep all non -1
        vert_indices = index_mapping_list[c]
        mapping_function = mapping_index_function(keep_me, vert_indices)

        cN = mark_neighbors_as_deleted(neighbor_list[i],mask_list[c]).long()
        no_neighbor = cN == -1
        cN = mapping_function[cN]
        cN[no_neighbor] = -1
        cropped_neighbors.append(cN)
        c+=1

    c = 1
    cropped_upsample = []
    for i in range(0, len(upsample_list)):
        keep_me = mask_list[c].squeeze(1) != -1  ## keep all non -1 - True is
        only_upsample_to_that = upsample_list[i][keep_me]

        iml = index_mapping_list[c - 1]
        for j in range(len(only_upsample_to_that)):
            c_upsample = only_upsample_to_that[j]
            mask = ~torch.isin(c_upsample, iml)
            c_upsample[mask] = -1
            only_upsample_to_that[j] = c_upsample

        keep_me_before = mask_list[c-1].squeeze(1) != -1
        vert_indices_before = index_mapping_list[c-1]
        mapping_function = mapping_index_function(keep_me_before, vert_indices_before)

        no_neighbor = only_upsample_to_that == -1

        only_upsample_to_that = mapping_function[only_upsample_to_that]

        only_upsample_to_that[no_neighbor] = -1



        cropped_upsample.append(only_upsample_to_that)
        c+=1

    # 4 step update downsample - indices of high - number of low
    c = 0
    cropped_downsample = []

    for i in range(0, len(downsample_list)):
        keep_me = mask_list[c].squeeze(1) != -1  ## keep all non -1 - True is

        only_downsample_to_that = downsample_list[i][keep_me]

        keep_me_after = mask_list[c+1].squeeze(1) != -1

        iml = index_mapping_list[c+1]
        for j in range(len(only_downsample_to_that)):
            c_downsample = only_downsample_to_that[j]
            mask = ~torch.isin(c_downsample,iml)
            c_downsample[mask] = -1
            only_downsample_to_that[j] = c_downsample

        vert_indices_after = index_mapping_list[c+1]
        mapping_function = mapping_index_function(keep_me_after, vert_indices_after)

        no_neighbor = only_downsample_to_that == -1

        only_downsample_to_that = mapping_function[only_downsample_to_that]
        only_downsample_to_that[no_neighbor] = -1

        cropped_downsample.append(only_downsample_to_that)
        c+=1

    return cropped_vertics,cropped_tetrahedra,cropped_neighbors,cropped_upsample,cropped_downsample



def tetra_subdivide2(cur_verts_features, next_parents, device,any=True):
    next_parents = next_parents.to(device)
    cur_verts_features = cur_verts_features.to(device)


    mask = next_parents == -1

    next_parents[mask] = torch.max(next_parents)+1
    cur_verts_features = torch.nn.functional.pad(cur_verts_features, (0, 0, 0, 1, 0, 0), value=0)
    neighbors = index_select(cur_verts_features, next_parents, dim=1)

    return neighbors.mean(2)

def mask_cube(mask,vertices, tetra_cubes, neighbors, upsample,downsample,cuda_device,any=True):


    mask = torch.where(mask==1,0,-1).unsqueeze(-1) # make it compatible with the notation here

    mask = torch.where(mask==-1,0,torch.nan)[None,...]
    mask = tetra_subdivide2(mask, neighbors[-1].clone(), cuda_device) #v25
    mask = tetra_subdivide2(mask, neighbors[-1].clone(), cuda_device) #v25
    mask_list = []
    mask_list.append(torch.where(torch.nan_to_num(mask, -1) == -1, 0, -1)[0].cpu())
    for i in reversed(range(len(downsample))):
        mask = tetra_subdivide2(mask,downsample[i].clone(),cuda_device)
        mask_list.append(torch.where(torch.nan_to_num(mask,-1)==-1,0,-1)[0].cpu())

    mask_list = mask_list[::-1]

    return crop_all_data(vertices, tetra_cubes, neighbors, upsample,downsample,mask_list),mask_list[-1]

if __name__ == "__main__":
    mask,vertices, tetra_cubes, neighbors, upsample, downsample, cuda_device = torch.load("/scratch2/samples/TetraDiffusion_HyperDrive/trial.pt")
    mask_cube(mask,vertices, tetra_cubes, neighbors, upsample, downsample, cuda_device)
