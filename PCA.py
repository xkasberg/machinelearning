import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from six.moves import urllib
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA as SKPCA


def normalize(X):
    """Normalize the given dataset X
    
    Args:
        X: ndarray, dataset
    
    Returns:
        (Xbar, mean, std): tuple of ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the 
        mean and standard deviation respectively.
    
    Note:
        You will encounter dimensions where the standard deviation is
        zero, for those when you do normalization the normalized data
        will be NaN. Handle this by setting using `std = 1` for those 
        dimensions when doing normalization.
    """
    mu = np.zeros(X.shape[1])
    mu = np.mean(X, axis =0)
    std = np.std(X, axis=0)
    std_filled = std.copy()
    std_filled[std==0] = 1.
    Xbar = (X - mu) / std_filled
    return Xbar, mu, std



def eig(S):
    """Compute the eigenvalues and corresponding eigenvectors 
        for the covariance matrix S.
    Args:
        S: ndarray, covariance matrix
    
    Returns:
        (eigvals, eigvecs): ndarray, the eigenvalues and eigenvectors

    Note:
        the eigenvals and eigenvecs should be sorted in descending
        order of the eigen values
    """
    #computes 'right' eigenvalues, eigenvectors of an mxm matrix 
    eigenvalues, eigenvectors = np.linalg.eig(S)
    desc_order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[desc_order], eigenvectors[:, desc_order]
    return (eigenvalues, eigenvectors)

def projection_matrix(B):
    """Compute the projection matrix onto the space spanned by `B`
    Args:
        B: ndarray of dimension (D, M), the basis for the subspace
    
    Returns:
        P: the projection matrix
    """
    P = B @ (np.linalg.inv(B.T @ B)) @ B.T
    return P

def PCA(X, num_components):
    """
    Args:
        X: ndarray of size (N, D), where D is the dimension of the data,
           and N is the number of datapoints
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: ndarray of the reconstruction
        of X from the first `num_components` principal components.
    """
    #Normalize Data
    #Xbar, mu, std = normalize(X)
    
    #Compute the covariance Matrix S, and find its Eigenvalues and their corresponding Eigenvectors
    S = np.cov(Xbar, rowvar=False, bias=True)
    eigenvalues, eigenvectors = eig(S)
    basis = eigenvectors[:, :num_components]

    P = projection_matrix(basis)

    X_reconstruct = (P @ X.T).T
    return X_reconstruct


def mse(predict, actual):
    """Helper function for computing the mean squared error (MSE)"""
    return np.square(predict - actual).sum(axis=1).mean()


def show_pca_digits(i):
    """Show the i th digit and its reconstruction"""
    plt.figure(1, figsize=(3,3))
    actual_sample = images[i]
    reconst_sample = (reconst[i, :] * std + mu).reshape(8, 8)
    plt.imshow(np.hstack([actual_sample, reconst_sample]), cmap='gray')
    plt.show()



digits = load_digits()

images, labels = digits.images, digits.target
#Uncomment to view graphic
# print("Image 0 in Images, with color map gray: ")
# plt.figure(1, figsize=(3, 3))
# plt.imshow(images[1], cmap='gray') #interpolation='nearest')
# plt.show()

X = images.reshape(-1, 8*8)
Xbar, mu, std = normalize(X)

loss = []
reconstructions = []
# iterate over different number of principal components, and compute the MSE
for num_component in range(1, 100):
    #print("Component "+str(num_component))
    reconst = PCA(Xbar, num_component)
    error = mse(reconst, Xbar)
    reconstructions.append(reconst)
    # print('n = {:d}, reconstruction_error = {:f}'.format(num_component, error))
    loss.append((num_component, error))


reconstructions = np.asarray(reconstructions)
reconstructions = reconstructions * std + mu # "unnormalize" the reconstructed image
loss = np.asarray(loss)

#Uncomment to view graphic
#show_pca_digits(1)


pd.DataFrame(loss).head()
fig, ax = plt.subplots()
ax.plot(loss[:,0], loss[:,1]);
ax.axhline(100, linestyle='--', color='r', linewidth=2)
ax.xaxis.set_ticks(np.arange(1, 100, 5));
ax.set(xlabel='num_components', ylabel='MSE', title='MSE vs number of principal components');
plt.show()


##Sci-Kit Learn Implementation###
# for num_component in range(1, 8):
#     pca = SKPCA(n_components=num_component, svd_solver='full')
#     sklearn_reconst = pca.inverse_transform(pca.fit_transform(Xbar))
#     reconst = PCA(Xbar, num_component)
#     np.testing.assert_almost_equal(reconst, sklearn_reconst)
#     print(np.square(reconst - sklearn_reconst).sum())
