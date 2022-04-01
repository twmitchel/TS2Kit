# _TS2Kit_: Differentiable Spherical Harmonic Transforms in PyTorch
_TS2Kit_ (**Version 1.0**) is a self-contained PyTorch library which computes auto-differentiable forward and inverse discrete Spherical Harmonic Transforms (**SHTs**). The routines in _TS2Kit_ are based on the seminal _S2Kit_ and _SOFT_ packages, but are designed for evaluation on a GPU.  Specifically, the Discrete Legendre Transform (**DLT**) is computed via sparse matrix multiplication in what is essentially a tensorized version of the so-called "semi-naive" algorithm. This enables parallelization while keeping memory footprint small, and the end result are auto-differentiable forward and inverse SHTs that are fast and efficient in a practical sense. For example, given a spherical signal (tensor) taking values on a 128 X 128 spherical grid with b = 4096 batch dimensions,  _TS2Kit_ computes a forward SHT followed by an inverse SHT in approximately tens of milliseconds at floating precision.

## Dependencies
- [PyTorch >= 1.10](https://pytorch.org)

## Conventions and Implementation
Please see `TS2Kit.pdf` for a detailed review of the chosen conventions and implementation details.

## Set up
To use _TS2Kit_, simply copy the `TS2Kit` folder into your project directory. 

### Setting the cache path
Several tensors are pre-computed at initialization and at higher bandlimits (B <= 64) this can take some time. To avoid re-computing these quantities every initialization, the modules will check if the tensors have been saved in a cache directory and either A). load the tensors directly from the cache; or B). compute the tensors and save them to the cache directory so they can be loaded next time the modules are initialized. 

To enable caching, choose a directory on your machine to serve as the cache folder and set the variable `cacheDir` at the top of the `ts2kit.py` file to the absolute path of the directory, _e.g._
```python
cacheDir = '/absolute/path/to/cache'
```
The cache directory can be cleared (of `.pt` files) at any time by importing and running the `clearCache` function:
```python
from TS2Kit.ts2kit import clearCache

clearCache()
```

## The Forward and Inverse SHTs
The front-end of _TS2Kit_ consists of the `torch.nn.Module` classes `FTSHT` and `ITSHT`, corresponding to the forward and inverse SHT, respectively. At initialization, the modules are passed an integer argument B which determines the bandlimit of the forward and inverse SHT, _e.g._
```python
from ts2kit.ts2kit import FTSHT, ITSHT

## Bandlimit
B = 64

## Initialize the (B-1)-bandlimited forward SHT
FT = FTSHT(B)

## Initialize the (B-1)-bandlimited inverse SHT
IT = ITSHT(B)
```

### `FTSHT`: The Forward SHT
Initialized with bandlimit B, calling the `FTSHT` module applies the forward SHT to a spherical signal composed with several batch dimensions. Specifically, inputs are b X 2B X 2B real or complex `torch` tensors, where b is the batch dimension and the second and third dimensions increment over the values in the 2B X 2B Driscoll-Healy spherical grid (see `TS2Kit.pdf`). For example, given a tensor `psi` of size 100 X 128 X 128 (b = 100, B = 64), the element `psi[26, 47, 12]` is the value of the spherical signal in batch dimension 26 at coordinates (theta_46, phi_11) in the DH spherical grid. To assist in sampling to a DH grid, the user can import the `gridDH` function, which takes as input a fixed bandlimit B and returns two 2B X 2B tensors `theta` and `phi` giving the spherical coordinates of the corresponding DH grid indices. 

The forward call returns a b X (2B-1) X B complex `torch` tensor giving the array of SH coefficients --  with m and l incremented along the second and third dimensions, respectively -- of spherical signals for each batch dimension of the input tensor. For example, passing the real or complex 100 X 128 X 128 tensor `psi` to the module returns the complex 100 X 127 X 64 tensor of SH coefficients:
```python
F = FTSHT(B)
psiCoeff = F(psi)
```
The (l, m)-th SH coefficients in batch dimension c can be accessed via `psiCoeff[c, m+B, l]` _e.g._ for ; l = 5, m = -5, and c = 12, the corresponding SH coefficient is `psiCoeff[12, 59, 5]`. For l < |m|, the values in `psiCoeff` will be zero.

### `ITSHT`: The Inverse SHT
Initialized with bandlimit B, calling the `ITSHT` module applies the inverse SHT to a signal composed of several arrays of SH coefficients. Inputs are b X (2B - 1) X B complex `torch` tensors consisting of b channels of SH coefficent arrays, structured in exactly the same way as the output of the `FTSHT` module. The forward call returns a b X 2B X 2B` _complex_ `torch` tensor corresponding to the spherical signals reconstructed from the SH coefficients in each batch dimension:
```python
I = ITSHT(B)
psi = I(psiCoeff)
```
The output tensor is complex-valued, so if the input SH coefficient tensor corresponds to a real-valued signal then the imaginary part of the output tensor will be zero and it can be cast to a real tensor (_e.g._ by calling `psi.real`) without loss of information. 

### Double Vs. Floating Precision
The `FTSHT` and `ITSHT` modules are initialized at double precision. That is, the forward call of `FTSHT` maps tensors of type `torch.double` (real-valued) or `torch.cdouble` (complex-valued) to tensors of type `torch.cdouble`. Similarly, the forward call of `ITSHT` maps tensors of type `torch.cdouble` to tensors of the same type. 

The modules can also be cast to floating precision at initialization, _e.g._ via `FTSHT(B).float()` and `ITSHT(B).float()`.  In this case, the forward call of `FTSHT` maps tensors of type `torch.float` and `torch.cfloat` to tensors of type `torch.cfloat` and that of `ITSHT` maps tensors of type `torch.cfloat` to tensors of the same type.

Casting to floating precision results in half the memory overhead and about an order of magnitude decrease in run-time at the cost of several orders of magnitude in accuracy. For example, given a tensor of double-precision SH coefficients on the GPU
```python
device = torch.device('cuda')

Psi = torch.view_as_complex(2*(torch.rand(b, 2*B -1, B, 2).double() - 0.5)).to(device)

for m in range(-(B-1), B):
    for l in range(0, B):
        if (l * l < m * m):
            Psi[:, m + (B-1), l] = 0.0;

```
one can expect the following error to be very, very small
```python
F = FTSHT(B).to(device)
I = ITSHT(B).to(device)

Psi2 = F(I(Psi))

## This error should be very, very  small
error = torch.sum(torch.abs(Psi-Psi2)) / torch.sum(torch.abs(Psi))
```
Casting to floating precision will result in a significant speed up and less overhead, but a larger error:
```python
Psi_f = Psi.cfloat();
F_f = FTSHT(B).to(device).float()
I_f = ITSHT(B).to(device).float()

## This should run about an order of magnitude faster
Psi2_f = F_f(I_f(Psi_f))

## This error will be much larger
error = torch.sum(torch.abs(Psi-Psi2)) / torch.sum(torch.abs(Psi))
```
This does not imply that the `FSHT` and `ISHT` modules are "slow" at double precision nor "inaccurate" at floating precision. Rather, it all depends on the application. The `test_ts2kit.ipynb` notebook can be used to compare the transforms at different precisions and bandlimits to see what makes sense for your use case. 

