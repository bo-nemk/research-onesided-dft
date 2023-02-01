import numpy as np


np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)


def real_vector(x):
    return np.concatenate([np.real(x), np.imag(x)])

def rfft_matrix(N):
    W = np.fft.rfft(np.eye(N), axis=0)
    V = np.vstack([np.real(W), np.imag(W)])
    return V

def irfft_matrix(N):
    W = np.fft.ifft(np.eye(N), axis=1)
    M = N // 2 + 1
    S = np.bmat([
        [np.eye(M), 1j * np.eye(M)],
        [np.zeros((M - 2, 1)), \
                np.fliplr(np.eye(M - 2)), np.zeros((M-2, 2)), np.fliplr(-1j * np.eye(M - 2)), np.zeros((M - 2, 1))]
        ])

    print(np.imag(W@S))
    return W @ S

if __name__ == "__main__":
    x = np.array([1,2,3,4,5,6])
    x = np.random.randn(1024)

    print(f"x:\n{x}")
    print(f"real vector rfft x:\n{real_vector(np.fft.rfft(x))}")
    print(f"real vector rfft_matrix x:\n{rfft_matrix(x.size) @ x}")
    print(f"\n")
    print(f"real vector rfft x:\n{real_vector(np.fft.rfft(x))}")
    print(f"irfft_matrix x:\n{np.real(irfft_matrix(x.size) @ real_vector(np.fft.rfft(x)))}")

