""" Implements Dynamic Mode Decomposition, kernel DMD, exact DMD,
    and related algorithms given sets of data
"""


struct DMD
    """ Dynamic Mode Decomposition (DMD)

    Dynamically relevant dimensionality reduction using the Proper Orthogonal
    Decomposition (POD) in conjunction with a least-squares solver.  This
    approach extracts a set of modes with a fixed temporal behavior (i.e.,
    exponential growth/decay).

    This implementation uses the numpy implementation of the singular value
    decomposition (SVD), and therefore requires the entire data set to fit in
    memory.  This algorithm runs in O(NM^2) time, where N is the size of a
    snapshot and M is the number of snapshots assuming N>M.

    Due to the similarities in implementation, this code can compute the modes
    associated with three variants of DMD that have appeared in the literature,
    and a forth that is a logical combination of existing approaches:

    1) Projected DMD (see Tu et al., 2014)
    2) Exact DMD (see Tu et al., 2014)
    3) Total least squares DMD (see Hemati & Rowley, 2015)
    4) Projected total least squares DMD (not published, but a logical
        combination of the Tu and Hemati papers)


    Parameters
    ----------
    n_rank : int or None, optional
        Number of POD modes to retain in the when performing DMD.  n_rank is
        an upper bound on the rank of the resulting DMD matrix.

        If n_rank is None (default), then all of the POD modes will
        be retained.

    exact : bool, optional
        If false (default), compute the DMD modes using projected DMD
        If true, compute the DMD modes using exact DMD

        See Tu et al., 2014 for details.

    total : bool, optional
        If false (default), compute the standard DMD modes
        If true, compute the total least squares DMD modes

        See Hemati & Rowley, 2015 for details.

    Attributes
    ----------
    evals : array, shape (n_rank,) or None
       The eigenvalues associated with each mode (None if not computed)

    modes: array, shape (n_dim, n_rank) or None
       The DMD modes associated with the eigenvalues in evals

    basis : array, shape (n_dim, n_rank) or None
       The basis vectors used to construct the modes.  If exact=False,
       these are the POD modes ordered by energy.

    Atilde : array, shape (n_rank, n_rank) or None
       The DMD matrix used in mode computation

    Notes
    -----
    Implements the DMD algorithms as presented in:

    Tu et al. On Dynamic Mode Decomposition: Theory and Applications,
        Journal of Computational Dynamics 1(2), pp. 391-421 (2014).

    Total least squares DMD is defined in:

    Hemati and Rowley, De-biasing the dynamic mode decomposition
        for applied Koopman spectral analysis, arXiv:1502.03854 (2015).

    For projected DMD as defined in Tu et al., exact=False and total=False
    For exact DMD as defined in Tu et al., exact=True and total=False
    For total least squares DMD in Hemati & Rowley, exact=True and total=True

    """
    n_rank::Int64
    exact::Bool
    total::Bool
    _basis
    _Ã
    _w
    _λ
end

function DMD(n_rank=0, exact=false, total=false)

    # Internal variables
    _basis = nothing  # spatial basis vectors
    _Ã = nothing  # The full DMD matrix
    _w = nothing  # DMD mode coefficients
    _λ = nothing  # DMD eigenvalues
    

    return DMD(n_rank, exact, total, _basis, _Ã, _w, _λ)
end
    

function modes(self::DMD)
"""
Koopman mode for observable x (identity map)
w -> λw under the map F
"""
    return self._basis * self._w
end

function evals(self::DMD)
    return self._λ
end
function basis(self::DMD)
    return self._basis
end

function Atilde(self::DMD)
    return self._Ã
end
function fit!(self::DMD, X, Y=nothing)
    """ Fit a DMD model with the data in X (and Y)

    Parameters
    ----------
    X : array, shape (n_dim, n_snapshots)
        Data set where n_snapshots is the number of snapshots and
        n_dim is the size of each snapshot.  Note that spatially
        distributed data should be "flattened" to a vector.

        If Y is None, then the columns of X must contain a time-series
        of data taken with a fixed sampling interval.

    Y : array, shape (n_dim, n_snapsots)
        Data set containing the updated snapshots of X after a fixed
        time interval has elapsed.

    Returns
    -------
    self : object
        Returns this object containing the computed modes and eigenvalues
    """

    if Y === nothing
        Y = X[:, 2:end]
        X = X[:, 1:end-1]
    end

    #  Max rank is either the specified value or determined by matrix size
    if self.n_rank == 0
        n_rank = min(self.n_rank, size(X, 1), size(X, 2))
    else
        n_rank = min(size(X))
    end

    # ====== Total Least Squares DMD: Project onto shared subspace ========
    # project both XY on the joint 
    if self.total
        # Compute V using the method of snapshots
        U, S, _ = svd(X'*X + Y'*Y, full=false)
        
        V_stacked = U[:, 1:n_rank]  # truncate to n_rank

        # Compute the "clean" data sets
        proj_Vh = V_stacked * V_stacked'
        X = X * proj_Vh
        Y = Y * proj_Vh
    end

    # ===== Dynamic Mode Decomposition Computation ======
    # AX    = Y
    # AUΣVᵀ = Y  (X = UΣVᵀ)
    # AU = YVΣ⁻¹
    # Ã := UᵀAU  = UᵀYVΣ⁻¹
    # Ãw = λw, 
    #
    # Standard DMD:
    # Any eigen pair of A with the form (λ, Uw), 
    # then (λ, w) is the eigen pair of Ã, Ãw = UᵀAUw = λUᵀUw = λw
    # Any eigen pair of Ã with the form (λ, w), 
    # then Uᵀ(AUw - λw) = 0,  AUw = λUw + w' with w' ∈ U⟂ 
    #
    # Exact DMD:
    # Let denote A = YVΣ⁻¹Uᵀ (AX = Y)
    # Any eigen pair of Ã with the form (λ, w), 
    # then A(YVΣ⁻¹w) = YVΣ⁻¹UᵀYVΣ⁻¹w = YVΣ⁻¹Ãw = YVΣ⁻¹λw = λ(YVΣ⁻¹w)
    # =====
    U, S, Vᵀ = svd(X, full=false)

    if self.n_rank > 0
        U = U[:, 1:n_rank]
        S = S[1:n_rank]
        Vᵀ = Vᵀ[1:n_rank, :]
    end

    # Compute the DMD matrix using the pseudoinverse of X
    self._Ã = U'*Y*Vᵀ'/S

    # Eigensolve gives modes and eigenvalues
    self._λ, self._w = eig(self._Ã)

    # Two options: exact modes or projected modes
    if self.exact
        self._basis = Y*Vᵀ'/S
    else
        self._basis = U
    end

end


struct KDMD
    """ Kernel Dynamic Mode Decomposition (KDMD)

    Dynamically relevent dimensionality reduction using kernel-based methods
    to implicitly choose a larger subset of observable space than used by
    standard DMD.  This approach extracts a set of modes with fixed temoral
    behaviors (i.e., exponential growth or decay) and embeds the data
    in an approximate Koopman eigenfunction coordinate system.

    This implementation uses the numpy implementation of the singular value
    decomposition (SVD), and therefore requires the entire data set to fit in
    memory.  This algorithm runs in O(NM^2) time, where N is the size of a
    snapshot and M is the number of snapshots assuming N>M.

    Due to the similarities in implementation, this code computes four
    variants of Kernel DMD, only one of which has appeared in the literature.

    1) Kernel DMD (see Williams, Rowley, & Kevrekidis, 2015)
    2) Exact Kernel DMD (modes are based on the Y data rather than the X data)
    3) Total least squares kernel DMD (a combination of Williams 2014 and
        Hemati 2015)
    4) Exact, TLS, kernel DMD (a combination of Williams 2014 and Hemati 2015)


    Parameters
    ----------
    kernel_fun : function or functor (array, array) -> square array
        A kernel function that computes the inner products of data arranged
        in an array with snapshots along each *COLUMN* when the __call__
        method is evaluated.

    n_rank : int or None, optional
        Number of features to retain in the when performing DMD.  n_rank is
        an upper bound on the rank of the resulting DMD matrix.

        If n_rank is None (default), then n_snapshot modes will be retained.

    exact : bool, optional
        If false (default), compute the KDMD modes using the X data
        If true, compute the KDMD modes using the Y data

        See Tu et al., 2014 and Williams, Rowley, & Kevrekidis 2014
        for details.

    total : bool, optional
        If false (default), compute the standard KDMD modes
        If true, compute the total least squares KDMD modes

        See Hemati & Rowley, 2015 and Williams, Rowley,
        & Kevrekidis, 2015 for details.

    Attributes
    ----------
    evals : array, shape (n_rank,) or None
       The eigenvalues associated with each mode (None if not computed)

    modes: array, shape (n_dim, n_rank) or None
       The DMD modes associated with the eigenvalues in evals

    Phi : array, shape (n_rank, n_snapshots) or None
       An embedding of the X data

    Atilde : array, shape (n_rank, n_rank) or None
       The "KDMD matrix" used in mode computation

    Notes
    -----
    Implements the DMD algorithms as presented in:

    Williams, Rowley, and Kevrekidis.  A Kernel-Based Approach to
        Data-Driven Koopman Spectral Analysis, arXiv:1411.2260 (2014)

    Augmented with ideas from:

    Hemati and Rowley, De-biasing the dynamic mode decomposition
        for applied Koopman spectral analysis, arXiv:1502.03854 (2015).

    For kernel DMD as defined in Williams, exact=False and total=False
    """

    kernel_fun::Function
    n_rank::Int
    exact::Bool
    total::Bool

    _basis
    _Ã
    _w
    _λ

    self._modes = None
    self._evals = None
    self._Phi = None
    self._Ã = None
    self._G = None
    self._A = None
end

function KDMD(kernel_fun::Function, n_rank=0, exact=false, total=false)
    
end

function modes(self::KDMD)
        return self._modes

function evals(self::KDMD)
    return self._λ
end

function basis(self::KDMD)
    return self._basis
end

function Atilde(self::KDMD)
    return self._Ã
end

function fit!(self::KDMD, X, Y=nothing)
    """ Fit a DMD model with the data in X (and Y)

    Parameters
    ----------
    X : array, shape (n_dim, n_snapshots)
        Data set where n_snapshots is the number of snapshots and
        n_dim is the size of each snapshot.  Note that spatially
        distributed data should be "flattened" to a vector.

        If Y is None, then the columns of X must contain a time-series
        of data taken with a fixed sampling interval.

    Y : array, shape (n_dim, n_snapsots)
        Data set containing the updated snapshots of X after a fixed
        time interval has elapsed.

    Returns
    -------
    self : object
        Returns this object containing the computed modes and eigenvalues
    """

    # Ψ(x) = [ψ(x₁)ᵀ; ψ(x₂)ᵀ; ..., ψ(xₙ)ᵀ]
    # Ψ(y) = [ψ(y₁)ᵀ; ψ(y₂)ᵀ; ..., ψ(yₙ)ᵀ]
    # Ψ(x) K = Ψ(y)
    # G = Ψ(x)ᵀΨ(x)
    # A = Ψ(x)ᵀΨ(y)
    # K = G⁻¹ A
    # Instead of computing G or A (the dimension of feature space is very high), we compute
    # Ψ(x) = QΣZᵀ
    # K := ZΣ⁻¹QᵀΨ(y)
    # μv = Kv, we consider v = Zv̂
    # μZv̂ = KZv̂ = ZΣ⁻¹QᵀΨ(y)Zv̂ = ZΣ⁻¹QᵀΨ(y)Ψ(x)ᵀQΣ⁻¹v̂
    # μv̂ = Σ⁻¹QᵀÂQΣ⁻¹v̂ where Â = Ψ(x)Ψ(y)ᵀ   Âᵢⱼ= K(xᵢ,yⱼ) 
    # Then (μ, v̂) is the eigen pair of K̂ = Σ⁻¹QᵀÂQΣ⁻¹

    if Y === nothing
        # Efficiently compute Â = ψ(x)ᵀψ(y), Ĝ = ψ(x)ᵀψ(y), and optionally Ĝy = ψ(y)ᵀψ(y)
        # given a time series of data

        Ĝfull = inner_product(self.kernel_fun, X, X)
        Ĝ = Ĝfull[1:end-1, 1:end-1]
        Â = Ĝfull[2:end, 1:end-1]

        if self.total
            Ĝy = Ĝfull[2:end, 2:end]
        end

        Y = X[:, 2:end]
        X = X[:, 1:end-1]
    else  
        # Paired data
        # ψ(x) = [ψ₁(x); ψ₂(x); ...; ψₖ(x)] 
        gram_tuple = compute_products(self.kernel_fun, X, Y, self.total)

        if self.total
            Ĝ, Â, Ĝy = gram_tuple
        else
            Ĝ, Â = gram_tuple
        end
    end
    # Rank is determined either by the specified value or
    # the number of snapshots
    if self.n_rank == 0
        n_rank = min(self.n_rank, size(X,1))
    else
        n_rank = size(X,1)
    end

    # ====== Total Least Squares DMD: Project onto shared subspace ========
    if self.total
        # Compute V using the method of snapshots
        U, S, _ = svd(G + Gy)
        V_stacked = U[:, 1:n_rank]  # truncate to n_rank

        # Compute the "clean" data sets
        proj_Vh = V_stacked * V_stacked'
        Ĝ = proj_Vh * G * proj_Vh
        Â = proj_Vh * A * proj_Vh
        X = X * proj_Vh
        Y = Y * proj_Vh
    end

    # ===== Kernel Dynamic Mode Decomposition Computation ======
    # Ψ(x) = [ψ(x₁)ᵀ; ψ(x₂)ᵀ; ..., ψ(xₙ)ᵀ]
    # Ψ(y) = [ψ(y₁)ᵀ; ψ(y₂)ᵀ; ..., ψ(yₙ)ᵀ]
    # Ψ(x) K = Ψ(y)
    # G = Ψ(x)ᵀΨ(x)
    # A = Ψ(x)ᵀΨ(y)
    # K = G⁻¹ A
    # Instead of computing G or A (the dimension of feature space is very high), we compute
    # Ψ(x) = QΣZᵀ
    # K := ZΣ⁻¹QᵀΨ(y)
    # μv = Kv, we consider v = Zv̂
    # μZv̂ = KZv̂ = ZΣ⁻¹QᵀΨ(y)Zv̂ = ZΣ⁻¹QᵀΨ(y)Ψ(x)ᵀQΣ⁻¹v̂
    # μv̂ = Σ⁻¹QᵀÂQΣ⁻¹v̂ where Â = Ψ(x)Ψ(y)ᵀ   Âᵢⱼ= K(xᵢ,yⱼ) 
    # Then (μ, v̂) is the eigen pair of K̂ = Σ⁻¹QᵀÂQΣ⁻¹

    self._Â = Â
    self._Ĝ = Ĝ
    Q, S2, _ = svd(Ĝ, full=false)
    Q = Q[:, 1:n_rank]
    S2 = S2[1:n_rank]
    S = sqrt.(S2)
    self._Ã = (Q'*Â*Q)/S2  #S\(Q'*Â*Q)/S

    # Eigensolve gives modes and eigenvalues
    self._λ, self._w = eig(self._Ã)
    # self._evals, vecs = np.linalg.eig(self._Ã)
    self._PhiX = (U*self._w)'

    # Two options: exact modes or projected modes
    if self.exact
        PhiY = (Â*U/S2 * self._w)'
        self._modes = Y*pinv(PhiY)
    else
        self._modes = X*pinv(self._PhiX)
    end
end

struct PolyKernel
    """ Implements a simple polynomial kernel

    This class is meant as an example for implementing kernels.

    Parameters
    ----------
    alpha : int
        The power used in the polynomial kernel
    epsilon : double, optional
        Scaling parameter in the kernel, default is 1.

    Notes
    -----
    We refer to the transformation from state space to feature space a f 
    in all that follows.
    """
    α::Int
    ϵ::FT

function PolyKernel(α::Int, ϵ::FT=1.0)
    return PolyKernel(α, ϵ)
end

function inner_product(self::PolyKernel, X, Y)
        """
        Compute the inner products (in feature space) of f(X)^T*f(Y)

        Parameters
        ----------
        X : array, shape (n_dim, n_snapshots)
            Data set where n_snapshots is the number of snapshots and
            n_dim is the size of each snapshot.  Note that spatially
            distributed data should be flattened to a vector.

        Y : array, shape (n_dim, n_snapsots)
            Data set containing the updated snapshots of X after a fixed
            time interval has elapsed.

        Returns
        -------
        self : array, shape (n_snapsots, n_snapshots)
            Returns the matrix of inner products in feature space
        """

        return (1.0 .+ X'*Y/self.ϵ).^self.α

function compute_products(self::PolyKernel, X, Y, Gy=false)
    """
    Compute the inner products f(X)^T*f(X), f(Y)^T*f(X), and if needed f(Y)^T*f(Y).

    For a polynomial kernel, this code is no more efficient than
    computing the terms individually.  Other kernels require
    knowledge of the complete data set, and must use this.

    Note: If this method is not implemented, the KDMD code will
    manually compute the inner products using the __call__ method.
    """

    if Gy
        return inner_product(self, X, X), inner_product(self, Y, X), inner_product(self, Y, Y)
    else
        return inner_product(self, X, X), inner_product(self, Y, X)
    end
end

function sort_modes_evals(λ, modes, k=0, sortby="LM", target=None)
    """ Sort and return the DMD or KDMD modes and eigenvalues

    Parameters
    ----------
    dmd_class : object
       A DMD-like object with evals and modes properties

    k : int, optional
        The number of DMD mode/eigenvalue pairs to return.
        None (default) returns all of them.

    sortby : string, optional
       How to sort the eigenvalues and modes.  Options are

       "LM"   : Largest eigenvalue magnitudes come first
       "closest" : Sort by distance from argument target

    target : complex double, optional
       If "closest" is chosen, sort by distance from this eigenvalue
    """

    if k == 0
        k = len(λ)
    end

    if sortby == "LM"
        inds = sortperm(abs.(evals), rev=true)
    elseif sortby == "closest"
        inds = sortperm(abs.(evals - target), rev=true)
    else
        NotImplementedError("Cannot sort by " + sortby)
    end

    λ = λ[inds]
    modes = modes[:, inds]

    return λ[:k], modes[:, :k]

    end
end