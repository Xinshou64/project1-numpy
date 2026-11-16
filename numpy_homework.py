import numpy as np

#np_e1
A=np.array([[0.20, -0.50, 1.10, 0.40],
              [0.80, 0.30, -0.70, 0.60],
              [-0.40, 0.90, 0.10, -0.20]])
print(A.ndim,A.shape,A.size)
print(A[1],A[:,2:4],A[1].shape,A[:,2:4].shape)
B=A.copy()
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        if B[i,j]<0:
            B[i,j]=0
print(B)
print(A)
print()

#np_e2
a = np.array([0.80, -0.60, 0.70])
b = np.array([-0.20, 0.90, 0.40])
print(a*b,a@b,(a*b).shape,(a@b).shape)
length_a=np.sqrt(np.sum(a**2))
length_b=np.sqrt(np.sum(b**2))
print(length_a,length_b)
print(np.linalg.norm(a),np.linalg.norm(b))
print()

#np_e3
def angle_deg(a,b):
    c=np.linalg.norm(a)*np.linalg.norm(b)
    if c==0:
        raise ValueError("请输入非零向量")
    cos_theta=(a@b)/c
    cos_theta=np.clip(cos_theta,-1,1)
    theta_rad=np.arccos(cos_theta)
    theta_deg=np.degrees(theta_rad)
    return theta_deg
a = np.array([1.00, -1.00])
b = np.array([1.00, 0.00])
print(angle_deg(a,b))
print()

#np_e4
p = np.array([-0.50, 1.20, 2.00])
q = np.array([0.80, -0.40, 1.10])
diff=p-q
length_diff=np.sqrt(np.sum((diff)**2))
distance_diff=np.sum(np.abs(diff))
print(diff)
print(length_diff)
print(distance_diff)
print()

#np_e5
def unit_vector(a):
    length_a=np.linalg.norm(a)
    if length_a==0:
        raise ValueError("请输入非零向量")
    e_a=a/length_a
    return e_a
a = np.array([-1.20, 0.40, 0.80])
e_a=unit_vector(a)
print(e_a)
print(np.allclose(np.linalg.norm(e_a),1.0))
print()

#np.e6
A = np.array([[0.20, -0.50, 1.10],
              [0.80, 0.40, -0.30]])
print(A.sum(axis=0),A.sum(axis=0).shape)
print(A.mean(axis=1),A.mean(axis=1).shape)
print(A.max())
print()
#axis=0的结果依次对应A的第0，1，2列
#axis=1的结果依次对应A的第0，1行

#np_e7
X = np.array([[0.80, -0.60, 0.70],
              [-0.40, 0.50, 0.20]])
scale = np.array([1.50, 0.50, -1.00])
bias = np.array([-0.10, 0.20, 0.30])
Y=X*scale+bias
print(Y,Y.shape)
print()
#scale和bias的每个元素分别对应X的一列，
#广播时，它们会作用于X的每一行。

#np_e8
a = np.array([[[0.20, -0.40], [0.70, 0.10]],
              [[0.30, 0.80], [-0.20, 0.50]],
              [[-0.10, 0.60], [0.40, -0.30]]])
b = a + 0.10
batch=np.stack([a,b])
print(batch.shape)
blue=batch[:,2]
print(blue)
print(np.transpose(batch,(0,2,3,1)).shape)
print()

#np_m1
def angles_to(query,X):
    length=np.linalg.norm(query)*np.linalg.norm(X,axis=1)
    valid=length>0
    cos_theta=np.zeros(X.shape[0])
    if np.linalg.norm(query)==0:
        raise ValueError("query不能是零向量")
    cos_theta[valid]=(X[valid]@query)/length[valid]
    cos_theta[~valid]=np.nan
    cos_theta = np.clip(cos_theta, -1, 1)#防止因为浮点数导致越界
    theta_rad=np.arccos(cos_theta)
    theta_deg=np.degrees(theta_rad)
    return theta_deg

query = np.array([0.60, -0.80, 0.00])
X = np.array([[0.30, -0.40, 0.00],
              [-0.60, 0.80, 0.00],
              [0.80, 0.60, 0.00],
              [0.00, 0.00, 0.00]])
angle=angles_to(query,X)
print(angle.shape)
print(angle)
print()

#np_m2
def top_k_cosine(query,gallery,k):
    if not isinstance(k,int) or  k<=0:
        raise  ValueError("K不为正整数")
    lengths=np.linalg.norm(gallery,axis=1)
    valid=lengths>0
    valid_gallery=gallery[valid]
    valid_lengths=lengths[valid]
    query_length=np.linalg.norm(query)
    if k > valid_gallery.shape[0]:
        raise ValueError("k 不能超过有效行数")
    if valid_gallery.shape[0]==0:
        raise ValueError("gallery中都是零向量")
    if query_length==0:
        raise ValueError("query不能是零向量")
    similarities=(valid_gallery@query)/(valid_lengths*query_length)
    valid_indices=np.flatnonzero(valid)
    order=np.argsort(-similarities,kind="stable")
    top_positions=order[:k]
    top_indices=valid_indices[top_positions]
    top_similarties=similarities[top_positions]
    return top_indices,top_similarties
query = np.array([0.60, -0.80, 0.00])
gallery = np.array([[0.60, -0.80, 0.00],
                    [0.30, -0.40, 0.00],
                    [0.80, 0.60, 0.00],
                    [-0.60, 0.80, 0.00],
                    [0.00, 0.00, 0.00]])
ids = np.array(["img_a", "img_b", "img_c", "img_d", "img_zero"])
k=3
indices,similarties=top_k_cosine(query,gallery,k)
print(ids[indices],similarties)
print()

#np_m3
A = np.array([[0.20, -0.40],
              [0.80, 0.30],
              [-0.50, 0.70]])
B = np.array([[0.10, 0.20],
              [-0.60, -0.20]])
diff=A[:,None,:]-B[None,:,:]
D=np.sqrt(np.sum(diff**2,axis=2))
print(D,D.shape)
nearest=np.argmin(D,axis=1)
print(nearest)
print()

#np_m4
X = np.array([[0.80, -0.60, 0.70],
              [-0.40, -0.50, 0.90]])
W1 = np.array([[-0.10, 0.20, -0.80],
               [0.30, -0.50, 0.10]])
b1 = np.array([-0.15, 0.05])
W2 = np.array([[0.20, -0.60], [-0.80, 0.52]])
b2 = np.array([0.10, -0.20])
Z = X @ W1.T + b1
H=np.maximum(Z,0.0)
Y = H @ W2.T + b2
print(Z,Z.shape)
print(H,H.shape)
print(Y,Y.shape)