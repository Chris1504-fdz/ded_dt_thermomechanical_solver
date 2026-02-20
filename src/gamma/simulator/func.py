import cupy as cp
import cupyx.scipy.sparse as cusparse

def elastic_stiff_matrix(elements, nodes, Bip_ele, shear, bulk):
    n_n = nodes.shape[0]
    n_e = elements.shape[0]
    n_p = elements.shape[1]
    n_q = 8
    n_int = n_e*n_q
    nodes_pos = nodes[elements]
    Jac = cp.matmul(Bip_ele,nodes_pos[:,cp.newaxis,:,:].repeat(8,axis=1)) # J = B*x [B:8(nGP)*3(dim)*8(nN), x:nE*8*8*3]
    ele_detJac = cp.linalg.det(Jac)
    iJac = cp.linalg.inv(Jac) #inv J (nE*nGp*dim*dim)
    ele_gradN = cp.matmul(iJac,Bip_ele) # dN/dx = inv(J)*B

    ele_B = cp.zeros([n_e,n_q,6,n_p*3])
    ele_B[:,:,0,0:24:3] = ele_gradN[:,:,0,:]
    ele_B[:,:,1,1:24:3] = ele_gradN[:,:,1,:]
    ele_B[:,:,2,2:24:3] = ele_gradN[:,:,2,:]
    ele_B[:,:,3,0:24:3] = ele_gradN[:,:,1,:]
    ele_B[:,:,3,1:24:3] = ele_gradN[:,:,0,:]
    ele_B[:,:,4,1:24:3] = ele_gradN[:,:,2,:]
    ele_B[:,:,4,2:24:3] = ele_gradN[:,:,1,:]
    ele_B[:,:,5,2:24:3] = ele_gradN[:,:,0,:]
    ele_B[:,:,5,0:24:3] = ele_gradN[:,:,2,:]

    temp = cp.array([[0,1,2]]).repeat(n_p,axis=0).flatten()
    jB = 3*cp.tile(elements[:,cp.newaxis,cp.newaxis,:],(1,n_q,6,1)).repeat(3,axis=3) + temp
    vB = ele_B.reshape(-1,n_p*3)
    jB = jB.reshape(-1,n_p*3)
    iB = cp.arange(0,jB.shape[0])[:,cp.newaxis].repeat(n_p*3,axis=1)
    B = cusparse.csr_matrix((cp.ndarray.flatten(vB),(cp.ndarray.flatten(iB), cp.ndarray.flatten(jB))), shape = (6*n_int, 3*n_n), dtype = cp.float_)

    IOTA = cp.array([[1],[1],[1],[0],[0],[0]]) 
    VOL = cp.matmul(IOTA,IOTA.transpose()) 
    DEV = cp.diag([1,1,1,1/2,1/2,1/2])-VOL/3

    ELASTC = 2*DEV*shear[:,:,cp.newaxis,cp.newaxis] + VOL*bulk[:,:,cp.newaxis,cp.newaxis]
    ele_D = ele_detJac[:,:,cp.newaxis,cp.newaxis]*ELASTC
    temp = cp.arange(0,n_e*n_q*6).reshape(n_e,n_q,6)
    iD = temp[:,:,cp.newaxis,:].repeat(6,axis = 2)
    jD = temp[:,:,:,cp.newaxis].repeat(6,axis = 3)

    D = cusparse.csr_matrix((cp.ndarray.flatten(ele_D),(cp.ndarray.flatten(iD), cp.ndarray.flatten(jD))), shape = (6*n_int, 6*n_int), dtype = cp.float_)
    # ele_K =  ele_B.transpose([0,1,3,2])@ele_D@ele_B
    # ele_K = ele_K.sum(axis = 1)

    # K = B.transpose()*D*B 
    K = None
    return K,B,D,ele_B,ele_D,iD,jD,ele_detJac

def constitutive_problem(E, Ep_prev, Hard_prev, shear, bulk, a, Y, T_anneal = None, T = None):
    
    # anneal temperature that sets previously accumulated plastic strain values to zero at any intpt with T > T_anneal
    if T_anneal and (T is not None):
        Ep_prev[T > T_anneal, :] = 0.0
        Hard_prev[T > T_anneal, :] = 0.0
        
    # Defined as a flat 1D array for clean outer product
    IOTA = cp.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  
    VOL = cp.outer(IOTA, IOTA)
    DEV = cp.diag(cp.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])) - VOL / 3.0
    
    E_tr = E - Ep_prev  

    ELASTC = 2.0 * DEV * shear[:, :, cp.newaxis, cp.newaxis] + VOL * bulk[:, :, cp.newaxis, cp.newaxis]
    
    # CRITICAL FIX: squeeze(-1) strictly removes only the trailing matrix-vector axis
    S_tr = (ELASTC @ E_tr[..., cp.newaxis]).squeeze(-1)
    SD_tr = (2.0 * DEV * shear[:, :, cp.newaxis, cp.newaxis] @ E_tr[..., cp.newaxis]).squeeze(-1) - Hard_prev
    
    # Axis=-1 is safer than axis=2 for generic broadcasting
    norm_SD = cp.sqrt(cp.sum(SD_tr[..., 0:3]**2, axis=-1) + 2.0 * cp.sum(SD_tr[..., 3:6]**2, axis=-1))

    CRIT = norm_SD - Y
    IND_p = CRIT > 0 

    # Ensure zero-overhead copies are made before the early return
    S = S_tr.copy()
    DS = ELASTC.copy()
    Ep = Ep_prev.copy()
    Hard = Hard_prev.copy()

    # Fast boolean check (much faster than checking .shape[0])
    if not cp.any(IND_p):
        return S, DS, IND_p, Ep, Hard   

    # Extract states strictly for the yielded points
    SD_p = SD_tr[IND_p]
    norm_SD_p = norm_SD[IND_p]
    shear_p = shear[IND_p]
    a_p = a[IND_p]
    CRIT_p = CRIT[IND_p]
    Y_p = Y[IND_p]

    # Flow direction normal tensor (Broadcasting replaces .repeat)
    N_hat = SD_p / norm_SD_p[:, cp.newaxis]  
    
    denom = 2.0 * shear_p + a_p 
    Lambda = CRIT_p / denom

    # Correct the Stress
    S[IND_p] -= 2.0 * (shear_p * Lambda)[:, cp.newaxis] * N_hat  
    
    # Outer product N x N (Broadcasting replaces @)
    NN_hat = N_hat[:, :, cp.newaxis] * N_hat[:, cp.newaxis, :]
    const = 4.0 * shear_p**2 / denom

    # Algorithmic Tangent Modulus update (Broadcasting replaces double .repeat)
    factor1 = const[:, cp.newaxis, cp.newaxis]
    factor2 = (const * Y_p / norm_SD_p)[:, cp.newaxis, cp.newaxis]
    
    DS[IND_p] -= factor1 * DEV 
    DS[IND_p] += factor2 * (DEV - NN_hat)

    # Correct Plastic Strain (Direct broadcasting replaces matmul & transpose)
    voigt_corr = cp.array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    Ep[IND_p] += (Lambda[:, cp.newaxis] * N_hat) * voigt_corr

    # Correct Kinematic Backstress 
    Hard[IND_p] += (a_p * Lambda)[:, cp.newaxis] * N_hat
    
    return S, DS, IND_p, Ep, Hard

def transformation(Q_int, active_elements, ele_detJac, n_n_save):
    """
    Vectorized nodal averaging (L2 Projection) using bincounting.
    Eliminates sparse matrix assembly overhead entirely.
    """
    # CRITICAL FIX: Cast active_elements to CuPy array. 
    # domain.elements is stored as a NumPy array, so we must push it to the GPU.
    elem = cp.array(active_elements).T  # Shape: (8, n_e)
    
    # Ensure all inputs are strictly CuPy arrays
    weight = cp.asarray(ele_detJac).flatten()
    Q_int_flat = cp.asarray(Q_int).flatten()
    
    # Repeat the values for the 8 vertices of each element
    # cp.repeat is mathematically equivalent to cp.kron(..., cp.ones) but 100x faster
    vF1 = cp.repeat((weight * Q_int_flat).reshape(1, -1), 8, axis=0).flatten()
    vF2 = cp.repeat(weight.reshape(1, -1), 8, axis=0).flatten()
    jF = cp.repeat(elem, 8, axis=1).flatten()
    
    # cp.bincount instantly sums duplicate indices without needing sparse matrices
    F1_dense = cp.bincount(jF.astype(cp.int32), weights=vF1)
    F2_dense = cp.bincount(jF.astype(cp.int32), weights=vF2)
    
    # Ensure the arrays are at least n_n_save long (pad with zeros if necessary)
    if len(F1_dense) < n_n_save:
        F1_dense = cp.pad(F1_dense, (0, n_n_save - len(F1_dense)))
        F2_dense = cp.pad(F2_dense, (0, n_n_save - len(F2_dense)))
        
    # Safe division: Only divide where F2 > 0 to prevent NaN propagation
    Q_node = cp.ones(n_n_save, dtype=cp.float_)
    valid_mask = F2_dense[:n_n_save] > 1e-12
    
    Q_node[valid_mask] = F1_dense[:n_n_save][valid_mask] / F2_dense[:n_n_save][valid_mask]
    
    return Q_node
def disp_match(nodes, U, n_n_old, n_n):
    """
    Vectorized displacement matching. Finds the node on the previous top surface
    that shares the same X, Y coordinates as the newly activated node.
    """
    # 1. Identify candidate source nodes on the old top surface
    zel_prev = float(cp.max(nodes[0:n_n_old, 2]))
    old_top_mask = cp.abs(nodes[0:n_n_old, 2] - zel_prev) < 1e-5
    old_top_indices = cp.where(old_top_mask)[0]
    old_top_xy = nodes[old_top_indices, 0:2]
    
    # 2. Extract target nodes (the newly born ones)
    new_xy = nodes[n_n_old:n_n, 0:2]
    
    # 3. Match X-Y coordinates using broadcasted distance matrix
    # Shape: (N_new, N_old_top)
    dist_sq = (new_xy[:, 0:1] - old_top_xy[:, 0])**2 + (new_xy[:, 1:2] - old_top_xy[:, 1])**2
    
    # 4. Find the global index of the closest old node for each new node
    best_match_local = cp.argmin(dist_sq, axis=1)
    best_match_global = old_top_indices[best_match_local]
    
    # 5. Assign displacements instantly
    U[n_n_old:n_n, :] = U[best_match_global, :]
    
    return U