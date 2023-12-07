import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.sparse import csr_matrix, csc_matrix, find, coo_matrix
from scipy.sparse import eye, spdiags
from scipy.sparse.linalg import spsolve
from scipy.linalg import eigh
from scipy.io import  savemat


from IPython.display import display, HTML


def read_off(filepath):
    """
    Reads a simple .off file
    
    Input
    --------------
    filepath : str - path to the .off file
    
    Output
    --------------
    vertices : (n,3) array of vertex coordinates (float)
    faces    : (m,3) array of faces defined by vertices index (integers)
    """
    
    with open(filepath, 'r') as f:
        f.readline() # First line
        n_vertices, n_triangles, _ = f.readline().strip().split() # nb of vertices triangles and 0
        
        vertices, faces = [], []

        for i in range(int(n_vertices)):
            x, y, z = f.readline().strip().split()
            vertices.append([float(x), float(y), float(z)])

        for j in range(int(n_triangles)):
            _, i, j , k = f.readline().strip().split()
            faces.append([int(i), int(j), int(k)])
    
    return np.array(vertices), np.array(faces)



class Mesh3D:
    def __init__(self, path):
        """
        Initialize the mesh from a path
        """
        self.vertices, self.faces = read_off(path)
        self.A = None
        self.W = None


    def compute_faces_areas(self):
        """
        Compute the area of each face
        
        Input
        --------------
        vertices : (n,3) - vertex coordinates
        faces    : (m,3) - faces defined by vertex indices
        
        Output
        --------------
        faces_areas : (m,) - area of each face
        """
        v1 = self.vertices[self.faces[:,0]]  # (m,3)
        v2 = self.vertices[self.faces[:,1]]  # (m,3)
        v3 = self.vertices[self.faces[:,2]]  # (m,3)
        faces_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m,)

        return faces_areas
    
    def area_matrix(self):
        """
        Compute the diagonal area matrix
        
        Input
        --------------
        vertices : (n,3) - vertex coordinates
        faces    : (m,3) - faces defined by vertex indices
        
        Output
        --------------
        A : (n,n) sparse matrix in DIAgonal format
        """

        N = self.vertices.shape[0]

        # Compute face area

        v1 = self.vertices[self.faces[:,0]]  # (m,3)
        v2 = self.vertices[self.faces[:,1]]  # (m,3)
        v3 = self.vertices[self.faces[:,2]]  # (m,3)
        faces_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m,)

        I = np.concatenate([self.faces[:,0], self.faces[:,1], self.faces[:,2]])
        J = np.zeros_like(I)
        V = np.concatenate([faces_areas, faces_areas, faces_areas])/3

        # Get array of vertex areas
        vertex_areas = np.array(scipy.sparse.coo_matrix((V, (I, J)), shape=(N, 1)).todense()).flatten()

        A = scipy.sparse.dia_matrix((vertex_areas,0), shape=(N, N))
        self.A = A
        return A
    
    def cotan_matrix(self):
        """
        Compute the stiffness matrix
        
        Input
        --------------
        vertices : (n,3) - vertex coordinates
        faces    : (m,3) - faces defined by vertex indices
        
        Output
        --------------
        W : (n,n) sparse matrix in CSC format
        """
        N = self.vertices.shape[0]

        v1 = self.vertices[self.faces[:,0]]  # (m,3)
        v2 = self.vertices[self.faces[:,1]]  # (m,3)
        v3 = self.vertices[self.faces[:,2]]  # (m,3)

        # Edge lengths indexed by opposite vertex
        u1 = v3 - v2
        u2 = v1 - v3
        u3 = v2 - v1

        L1 = np.linalg.norm(u1,axis=1)  # (m,)
        L2 = np.linalg.norm(u2,axis=1)  # (m,)
        L3 = np.linalg.norm(u3,axis=1)  # (m,)

        # Compute cosine of angles
        A1 = np.einsum('ij,ij->i', -u2, u3) / (L2*L3)  # (m,)
        A2 = np.einsum('ij,ij->i', u1, -u3) / (L1*L3)  # (m,)
        A3 = np.einsum('ij,ij->i', -u1, u2) / (L1*L2)  # (m,)

        # Use cot(arccos(x)) = x/sqrt(1-x^2)
        I = np.concatenate([self.faces[:,0], self.faces[:,1], self.faces[:,2]])
        J = np.concatenate([self.faces[:,1], self.faces[:,2], self.faces[:,0]])
        S = np.concatenate([A3,A1,A2])
        S = 0.5 * S / np.sqrt(1-S**2)

        In = np.concatenate([I, J, I, J])
        Jn = np.concatenate([J, I, I, J])
        Sn = np.concatenate([-S, -S, S, S])

        W = scipy.sparse.coo_matrix((Sn, (In, Jn)), shape=(N, N)).tocsc()
        self.W = W
        return W
    
    def compute_laplacian(self):
        self.A = self.area_matrix()
        self.W = self.cotan_matrix()

    def compute_eigendecomposition(self, K:int):
        if self.W is None or self.A is None:
            self.compute_laplacian()
        self.eigenvalues, self.eigenvectors = scipy.sparse.linalg.eigsh(self.W, M=self.A,
                                                                  k=K, sigma=-0.01)

