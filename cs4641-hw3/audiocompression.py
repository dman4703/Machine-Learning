from typing import Tuple

import numpy as np


class AudioCompression(object):

    def __init__(self):
        pass

    def svd(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform SVD. You should use numpy's SVD.
        Your function should be able to handle single channel
        audio ((1, F, N) arrays) as well as stereo audio ((2, F, N) arrays)
        In this audio compression, we assume that each column of the spectrogram is a feature. Perform SVD on the channels of
        the audio (1 channel for single channel, 2 for stereo)
        The audio is the matrix X.

        Hint: np.linalg.svd by default returns the transpose of V. We want you to return the transpose of V, not V.

        Args:
            X: (C, F, N) numpy array corresponding to the audio

        Return:
            U: (C, F, F) numpy array
            S: (C, min(F, N)) numpy array
            V^T: (C, N, N) numpy array
        """
        if X.ndim == 2:
            U, S, Vt = np.linalg.svd(X, full_matrices=True)
            return U, S, Vt
        # end if
        C = X.shape[0]
        uChannels = []
        sChannels = []
        vtChannels = []
        for c in range(C):
            uChannel, sChannel, vtChannel = np.linalg.svd(X[c], full_matrices=True)
            uChannels.append(uChannel)
            sChannels.append(sChannel)
            vtChannels.append(vtChannel)
        # end for
        U = np.stack(uChannels, axis=0)
        S = np.stack(sChannels, axis=0)
        Vt = np.stack(vtChannels, axis=0)
        return U, S, Vt
    # end svd

    def compress(
        self, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compress the SVD factorization by keeping only the first k components

        Args:
            U (np.ndarray): (C, F, F) numpy array
            S (np.ndarray): (C, min(F, N)) numpy array
            V (np.ndarray): (C, N, N) numpy array
            k (int): int corresponding to number of components to keep

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                U_compressed: (C, F, k) numpy array
                S_compressed: (C, k) numpy array
                V_compressed: (C, k, N) numpy array
        """
        if U.ndim == 2:
            kActual = min(k, S.shape[0])
            uCompressed = U[:, :kActual]
            sCompressed = S[:kActual]
            vCompressed = V[:kActual, :]
            return uCompressed, sCompressed, vCompressed
        # end if
        kActual = min(k, S.shape[1])
        uCompressed = U[:, :, :kActual]
        sCompressed = S[:, :kActual]
        vCompressed = V[:, :kActual, :]
        return uCompressed, sCompressed, vCompressed
    # end compress

    def rebuild_svd(
        self,
        U_compressed: np.ndarray,
        S_compressed: np.ndarray,
        V_compressed: np.ndarray,
    ) -> np.ndarray:
        """
        Rebuild original matrix X from U, S, and V which have been compressed to k componments.

        Args:
            U_compressed: (C,F,k) numpy array
            S_compressed: (C,k) numpy array
            V_compressed: (C,k,N) numpy array

        Return:
            Xrebuild: (C,F,N) numpy array

        Hint: numpy.matmul may be helpful for reconstructing stereo audio
        """
        if U_compressed.ndim == 2:
            uScaled = U_compressed * S_compressed[None, :]
            return np.dot(uScaled, V_compressed)
        # end if
        uScaled = U_compressed * S_compressed[:, None, :]
        return np.matmul(uScaled, V_compressed)
    # end rebuild_svd

    def compression_ratio(self, X: np.ndarray, k: int) -> float:
        """
        Compute the compression ratio of a sample: (num stored values in compressed)/(num stored values in original)
        Refer to https://timbaumann.info/svd-image-compression-demo/
        Args:
            X: (C,F,N) numpy array
            k: int corresponding to number of components

        Return:
            compression_ratio: float of proportion of storage used by compressed audio
        """
        if X.ndim == 2:
            F, N = X.shape
            actualK = min(k, min(F, N))
            original = F * N
            compressed = actualK * (F + N + 1)
            return compressed / original
        # end if
        C, F, N = X.shape
        actualK = min(k, min(F, N))
        original = C * F * N
        compressed = C * actualK * (F + N + 1)
        return compressed / original
    # end compression_ratio

    def recovered_variance_proportion(self, S: np.ndarray, k: int) -> float:
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: (C, min(F,N)) numpy array
           k: int, rank of approximation

        Return:
           recovered_var: C floats corresponding to proportion of recovered variance for each channel
        """
        if S.ndim == 1:
            kActual = min(k, S.shape[0])
            total = np.sum(S ** 2)
            topK = np.sum(S[:kActual] ** 2)
            return float(topK / total) if total != 0 else 0.0
        # end if
        kActual = min(k, S.shape[1])
        total = np.sum(S ** 2, axis=1)
        topK = np.sum(S[:, :kActual] ** 2, axis=1)
        # Avoid divide-by-zero
        with np.errstate(divide='ignore', invalid='ignore'):
            recoveredVar = np.true_divide(topK, total)
            recoveredVar[~np.isfinite(recoveredVar)] = 0.0
        return recoveredVar
    # end recovered_variance_proportion

    def memory_savings(
        self, X: np.ndarray, U: np.ndarray, S: np.ndarray, V: np.ndarray, k: int
    ) -> Tuple[int, int, int]:
        """
        PROVIDED TO STUDENTS

        Returns the memory required to store the original audio X and
        the memory required to store the compressed SVD factorization of X

        Args:
            X (np.ndarray): (C,F,N) numpy array
            U (np.ndarray): (C,F,F) numpy array
            S (np.ndarray): (C,min(F,N)) numpy array
            V (np.ndarray): (C,N,N) numpy array
            k (int): integer number of components

        Returns:
            Tuple[int, int, int]:
                original_nbytes: number of bytes that numpy uses to represent X
                compressed_nbytes: number of bytes that numpy uses to represent U_compressed, S_compressed, and V_compressed
                savings: difference in number of bytes required to represent X
        """
        original_nbytes = X.nbytes
        U_compressed, S_compressed, V_compressed = self.compress(U, S, V, k)
        compressed_nbytes = (
            U_compressed.nbytes + S_compressed.nbytes + V_compressed.nbytes
        )
        savings = original_nbytes - compressed_nbytes
        return original_nbytes, compressed_nbytes, savings

    def nbytes_to_string(self, nbytes: int, ndigits: int = 3) -> str:
        """
        PROVIDED TO STUDENTS

        Helper function to convert number of bytes to a readable string

        Args:
            nbytes (int): number of bytes
            ndigits (int): number of digits to round to

        Returns:
            str: string representing the number of bytes
        """
        if nbytes == 0:
            return "0B"
        units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
        scale = 1024
        units_idx = 0
        n = nbytes
        while n > scale:
            n = n / scale
            units_idx += 1
        return f"{round(n, ndigits)} {units[units_idx]}"
