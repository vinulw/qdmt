'''
Helper functions to do Riemannian optimisation.

- Projecting onto tangent space.
- Performing a retraction along a tangent space vector.
- Parallell transport of vectors along manifold.
'''

import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group


def projectTangentStiefel(D, W):
    '''
    Project $D$ onto the tangent space of the Stiefel manifold at $W$. Use the
    Euclidean metric to perform the projection.

    $$
    G = D - \frac{1}{2} W (W^\dagger D + D^\dagger W)
    $$
    '''
    return D - 0.5*W @ (W.conj().T @ D+D.conj().T @ W)


def projectTangentGrassmann(D, W):
    '''
    Project $D$ onto the tangent space of the Grassmann manifold at $W$. Use
    the Euclidean metric to perform the prjection.
    '''
    return D - W @ W.conj().T @ D


def retractionStiefel(W, G, α=1, opt='euclidean'):
    '''
    Peform a retraction on the Stiefel manifold based on a Euclidean metric.

    Options include retraction based on a `euclidean` or `canonical` metrix.
    '''

    opts = ['euclidean', 'canonical']
    assert opt in opts

    if opt == 'euclidean':
        return _retractionStiefelEuclidean(W, G, α)
    else:
        return _retractionStiefelCanonical(W, G, α)


def _retractionStiefelEuclidean(W, G, α=1):
    '''
    Peform a retraction on the Stiefel manifold based on a Euclidean metric.
    '''

    A = W.conj().T @ G
    m = A.shape[0]
    I = np.eye(m)

    a = np.block([W, α*G])
    b = np.block([
        [α*A, -α**2 * G.conj().T @ G],
        [I, α*A]
    ])
    b = expm(b)[..., :m]
    c = expm(-α*A)

    return a @ (b @ c)


def _retractionStiefelCanonical(W, G, α=1):
    '''
    Peform a retraction on the Stiefel manifold based on a canonical metric.
    '''
    n, m = W.shape

    Q, R = np.linalg.qr((np.eye(n) - W @ W.conj().T) @ G, mode='reduced')
    A = W.conj().T @ G

    a = np.block([W, Q])
    b = np.block([
        [α*A, -α*R.conj().T],
        [α*R, np.zeros((m, m))]
    ])
    b = expm(b)[..., :m]

    return a @ b

def isSkewHermitian(W):
    return np.allclose(W, -1*W.conj().T)


def testProjectTangentStiefel():
    d = 4
    W = unitary_group.rvs(d)

    I = np.eye(d)

    # print('W in Stiefel: ', np.allclose(I, W.conj().T @ W))
    assert np.allclose(I, W.conj().T @ W)

    # Generate random update tensor
    D = np.random.randn(d, d) * 1j*np.random.randn(d, d)

    # # Check if D in tangent space
    # WdagD = W.conj().T @ D
    # # Should be false
    # print('D in Tangent: ', isSkewHermitian(WdagD))

    G = projectTangentStiefel(D, W)
    WdagG = W.conj().T @ G
    # print('G in Tangent: ', isSkewHermitian(WdagG))
    assert isSkewHermitian(WdagG)


def testProjecTangentGrassmann():
    d = 4
    W = unitary_group.rvs(d)

    # Generate random update tensor
    D = np.random.randn(d, d) * 1j*np.random.randn(d, d)

    G = projectTangentGrassmann(D, W)
    WdagG = W.conj().T @ G

    assert np.allclose(WdagG, np.zeros((d, d)))


def testRetractionStiefel():
    # Prepare isometry
    n, m = 6, 4
    W = unitary_group.rvs(n)
    W = W[:n, :m]

    print('W isometry: W†W == I')
    print('\t...', np.allclose(W.conj().T @ W, np.eye(m)))

    # Generate random update tensor
    D = np.random.randn(n, m) * 1j*np.random.randn(n, m)
    D = D / np.linalg.norm(D)

    G = projectTangentStiefel(D, W)
    print('Created gradient')

    print('Testing retraction Euclidean')
    Wprime_e = retractionStiefel(W, G, opt='euclidean')
    print('\t...Ran')
    print('\t...Isometry: ',
          np.allclose(Wprime_e.conj().T @ Wprime_e, np.eye(m)))
    assert np.allclose(Wprime_e.conj().T @ Wprime_e, np.eye(m))\
        , 'Stiefel Euclidean Retraction failing'

    print('Testing retraction Canonical')
    Wprime_c = retractionStiefel(W, G, opt='canonical')
    print('\t...Ran')
    print('\t...Isometry: ',
          np.allclose(Wprime_c.conj().T @ Wprime_c, np.eye(m)))
    assert np.allclose(Wprime_c.conj().T @ Wprime_c, np.eye(m))\
        , 'Stiefel Canonical Retraction failing'


if __name__ == "__main__":
    testRetractionStiefel()


