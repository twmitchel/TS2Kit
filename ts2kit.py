import os
import os.path as osp
import glob
import numpy as np
import torch
import torch.nn as nn
from scipy.special import gammaln
from math import pi as PI

sqrt2 = np.sqrt(2.0);

##########################
######## Cache ###########
##########################

## If you'd like to save cached files in a different directory, then change this
## to the absolute path of said directory
defaultCacheDir = 'cache'

try:
    ################################################
    ### For integration with Mobius Convolutions ###
    ################################################
    
    from cache.cache import cacheDir
    
except:
 
    #####################################
    ### For use as standalone package ###
    #####################################
    
    cacheDir = defaultCacheDir
    


def clearTS2KitCache(cacheDir=cacheDir):

    cFiles = glob.glob(osp.join(cacheDir, '*.pt'));

    for l in range(len(cFiles)):

        os.remove(cFiles[l]);


    return 1;


#############################
######## Utilities ##########
#############################


## Driscoll-Healy Sampling Grid ##
# Input: Bandlimit B
# Output: "Meshgrid" spherical coordinates (theta, phi) of 2B X 2B Driscoll-Healy spherical grid
# These correspond to a Y-Z spherical coordinate parameterzation:
# [X, Y, Z] = [cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)]

def gridDH(B):
    
    k = torch.arange(0, 2*B).double();
    
    theta = 2*PI*k / (2*B)
    phi = PI*(2*k + 1) / (4*B);
    
    theta, phi = torch.meshgrid(theta, phi, indexing='ij');
    
    return theta, phi;
    
    
###########################################################
################ Discrete Legendre Transform ##############
###########################################################

## Recursive computation of d^l_mn ( pi/2) 

def triHalfRecur(l, m, n):
    
    denom = (-1 + l)*np.sqrt( (l-m)*(l+m)*(l-n)*(l+n) );
    
    c1 = (1 - 2*l)*m*n/denom;
    
    c2 = -1.0*l*np.sqrt( ( (l-1)*(l-1) - m*m ) * ( (l-1)*(l-1) - n*n ) ) / denom;
    
    return c1, c2;

def generateLittleHalf(B):
    
    fName = osp.join(cacheDir, 'littleHalf_{}.pt'.format(B));
    
    if (osp.isfile(fName) == False):
                
        #m, n -> m + (B-1), n + (B-1)
        
        d = torch.empty( B, 2*B - 1, 2*B - 1).double().fill_(0);
        
        # Fill first two levels (l = 0, l = 1)
        d[0, B-1, B-1] = 1
        
        d[1, -1 +(B-1), -1 + (B-1)] = 0.5;
        d[1, -1 +(B-1), B-1] = 1.0 / sqrt2;
        d[1, -1 +(B-1), B] = 0.5;
        
        d[1, (B-1), -1 + (B-1)] = -1.0 / sqrt2;
        d[1, (B-1), B] = 1.0 / sqrt2;
        
        d[1, B, -1 + (B-1)] = 0.5;
        d[1, B, (B-1)] = -1.0 / sqrt2;
        d[1, B, B] = 0.5;
        
        ## Fill rest of values through Kostelec-Rockmore recursion
        
        for l in range(2, B):
            for m in range(0, l):
                for n in range(0, l):
                    
                    if ( (m == 0) and (n == 0) ):
                        
                        d[l, B-1, B-1] = -1.0*( (l-1)/l )*d[l-2, B-1, B-1];
                        
                    else:
                        c1, c2 = triHalfRecur(l, m, n);
                        
                        d[l, m + (B-1), n + (B-1)]= c1 * d[l-1, m + (B-1), n + (B-1)] + c2 * d[l-2, m+(B-1), n+(B-1)];
            
            for m in range(0, l+1):
                
                lnV = 0.5*( gammaln(2*l + 1) - gammaln(l+m +1) - gammaln(l-m + 1) ) - l*np.log(2.0);
                
                d[l, m+(B-1), l+(B-1)] = np.exp(lnV);
                d[l, l+(B-1), m+(B-1)] = np.power(-1.0, l - m) * np.exp(lnV);
                
            
            for m in range(0, l+1):
                for n in range(0, l+1):
                    
                    val = d[l, m+(B-1), n+(B-1)]
                    
                    if ( (m != 0) or (n != 0) ):
                        
                        d[l, -m + (B-1), -n + (B-1)] = np.power(-1.0, m-n)*val;
                        d[l, -m + (B-1), n + (B-1)] = np.power(-1.0, l-n)*val;
                        d[l, m+(B-1), -n + (B-1) ] = np.power(-1.0, l+m)*val;
                         
                
            
        torch.save(d, fName)  
        
        print('Computed littleHalf_{}'.format(B), flush=True);
    
    else:
        
        d = torch.load(fName);
        
 
    return d;

def dltWeightsDH(B):
    
    fName = osp.join(cacheDir, 'dltWeights_{}.pt'.format(B));
    
    if (osp.isfile(fName) == False):
        
        W = torch.empty(2*B).double().fill_(0);
        
        for k in range(0, 2*B):
            
            C = (2.0/B)*np.sin( PI*(2*k + 1) / (4.0*B) );
            
            wk = 0.0;
            
            for p in range(0, B):
                
                wk += (1.0 / (2*p + 1) ) * np.sin( (2*k + 1)*(2*p + 1) * PI / (4.0 * B));
                
            W[k] = C * wk;
        
        torch.save(W, fName);
        
        print('Computed dltWeights_{}'.format(B), flush=True);
               
    else:
    
        W = torch.load(fName);
        
    return W;

## Inverse (orthogonal) DCT Matrix of dimension N x N
def idctMatrix(N):
    
    fName = osp.join(cacheDir, 'idctMatrix_{}.pt'.format(N));
    
    if (osp.isfile(fName) == False):
        
        DI = torch.empty(N, N).double().fill_(0);
        
        for k in range(0, N):
            for n in range(0, N):
                
                DI[k, n] = np.cos(PI*n*(k + 0.5)/N)
        
        DI[:, 0] = DI[:, 0] * np.sqrt(1.0 / N);
        DI[:, 1:] = DI[:, 1:] * np.sqrt(2.0 / N);
        
        torch.save(DI, fName);
        
        print('Computed idctMatrix_{}'.format(N), flush=True);

        
    else:
        
        DI = torch.load(fName);
        
    return DI;

## Inverse (orthogonal) DST Matrix of dimension N x N
def idstMatrix(N):
    
    fName = osp.join(cacheDir, 'idstMatrix_{}.pt'.format(N));
    
    if (osp.isfile(fName) == False):
        
        DI = torch.empty(N, N).double().fill_(0);
        
        for k in range(0, N):
            for n in range(0, N):
                
                if (n == (N-1) ):
                    DI[k, n] = np.power(-1.0, k);
                else:
                    DI[k, n] = np.sin(PI*(n+1)*(k + 0.5)/N);

        DI[:, N-1] = DI[:, N-1] * np.sqrt(1.0 / N);
        DI[:, :(N-1)] = DI[:, :(N-1)] * np.sqrt(2.0 / N);
        
        torch.save(DI, fName);
        
        print('Computed idstMatrix_{}'.format(N), flush=True);

        
    else:
        
        DI = torch.load(fName);
        
    return DI;

# Normalization coeffs for m-th frequency (C_m)
def normCm(B):
        
    fName = osp.join(cacheDir, 'normCm_{}.pt'.format(B));
    
    if (osp.isfile(fName) == False):
        
        Cm = torch.empty(2*B - 1).double().fill_(0);
        
        for m in range(-(B-1), B):
            Cm[m + (B-1)] = np.power(-1.0, m) * np.sqrt(2.0 * PI);

        torch.save(Cm, fName);
        
        print('Computed normCm_{}'.format(B), flush=True);

        
    else:
        Cm = torch.load(fName);
        
    return Cm;

# Computes sparse matrix of Wigner-d function cosine + sine series coefficients
def wignerCoeffs(B):
    
    fName = osp.join(cacheDir, 'wignerCoeffs_{}.pt'.format(B));
    
    if (osp.isfile(fName) == False):
        
        d = generateLittleHalf(B).cpu().numpy();
        
        H = 0;
        W = 0;
        
        indH = [];
        indW = [];
        val = [];
        
        N = 2*B;
        
        for m in range(-(B-1), B):
            
            for l in range(np.absolute(m), B):
                
                for n in range(0, l+1):
                    
                    iH = l + H;
                    iW = n + W;
                    
                    # Cosine series
                    if ( (m % 2) == 0 ):
                        
                        if (n == 0):
                            c = np.sqrt( (2*l + 1)/2.0 ) * np.sqrt( N );
                        else:
                            c = np.sqrt( (2*l + 1)/2.0 ) * np.sqrt( 2.0*N );
                            
                        if ( (m % 4) == 2 ):
                            
                            c *= -1.0;
                        
                        coeff = c * d[l, n + (B-1), -m + (B-1)] * d[l, n+(B-1), B-1];
                    
                    # Sine series
                    else:
                        
                        if (n == l):
                            coeff = 0.0;
                            
                        else:
                            
                            c = np.sqrt( (2*l + 1) / 2.0 ) * np.sqrt( 2.0 * N);
                            
                            if ( (m % 4) == 1 ):
                                c *= -1.0;
  
                            coeff = c * d[l, (n+1) + (B-1), -m + (B-1)] * d[l, (n+1) + (B-1), B-1];
        
        
                    if ( np.absolute(coeff) > 1.0e-15 ):
                
                        indH.append(iH);
                        indW.append(iW);
                        val.append(coeff);
                        
            
            H += B;
            W += N;
            
        # Cat indices, turn into sparse matrix
        ind = torch.cat( (torch.tensor(indH).long()[None, :], torch.tensor(indW).long()[None, :]), dim=0);
        val = torch.tensor( val, dtype=torch.double );
        
        D = torch.sparse_coo_tensor(ind, val, [B*(2*B - 1), 2*B*(2*B - 1)])
        
        torch.save(D, fName);    
        
        print('Computed wignerCoeffs_{}'.format(B), flush=True);

                                
        
    else:
        
        D = torch.load(fName);
        
    return D;

# Weighted DCT and DST implemented as linear layers
# Adapted from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py
class weightedDCST(nn.Linear):
    '''DCT or DST as a linear layer'''
    
    def __init__(self, B, xform):
        self.xform = xform
        self.B = B
        super(weightedDCST, self).__init__(2*B, 2*B, bias=False)

    def reset_parameters(self):
        B = self.B;
        
        if (self.xform == 'c'): 
            W = torch.diag(dltWeightsDH(B))
            XF = torch.matmul(W, idctMatrix(2*B))
            
        elif(self.xform == 'ic'):         
            XF = idctMatrix(2*B).t()
          
        elif(self.xform == 's'): 
            W = torch.diag(dltWeightsDH(B))
            XF = torch.matmul(W, idstMatrix(2*B));

        elif(self.xform == 'is'):
            XF = idstMatrix(2*B).t()
        
        self.weight.data = XF.t().data;

        self.weight.requires_grad = False # don't learn this! 

        
# Forward Discrete Legendre Transform      
class FDLT(nn.Module):
    
    def __init__(self, B):
        super(FDLT, self).__init__()
        
        self.B = B;
        
        self.dct = weightedDCST(B, 'c');
        self.dst = weightedDCST(B, 's');
        
        if ( ((B-1)%2) == 1 ):
            
            cInd = torch.arange(1, 2*B-1, 2);
            sInd = torch.arange(0, 2*B-1, 2);
        
        else:
            
            sInd = torch.arange(1, 2*B-1, 2);
            cInd = torch.arange(0, 2*B-1, 2);
        
        self.register_buffer('cInd', cInd);
        self.register_buffer('sInd', sInd);

        self.register_buffer('Cm', normCm(B));
        
        self.register_buffer('D', wignerCoeffs(B));
        
        
    def forward(self, psiHat):
        
        # psiHat = b x M x phi
        
        B, b = self.B, psiHat.size()[0]
         
        # Multiply by normalization coefficients
        psiHat = torch.mul(self.Cm[None, :, None], psiHat);

        # Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :]);
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :]);
        
        # Reshape for sparse matrix multiplication
        psiHat = torch.transpose(torch.reshape(psiHat, (b, 2*B*(2*B - 1) ) ), 0, 1);
        
        # Psi =  b x M x L 
        return torch.permute(torch.reshape(torch.mm(self.D, psiHat), (2*B - 1, B, b)), (2, 0, 1));


# Inverse Discrete Legendre Transform      
class IDLT(nn.Module):
    
    def __init__(self, B):
        super(IDLT, self).__init__()
        
        self.B = B;
        
        self.dct = weightedDCST(B, 'ic');
        self.dst = weightedDCST(B, 'is');
        
        if ( ((B-1)%2) == 1 ):
            
            cInd = torch.arange(1, 2*B-1, 2);
            sInd = torch.arange(0, 2*B-1, 2);
        
        else:
            
            sInd = torch.arange(1, 2*B-1, 2);
            cInd = torch.arange(0, 2*B-1, 2);
                                
        self.register_buffer('cInd', cInd);
        self.register_buffer('sInd', sInd);

        self.register_buffer('iCm', torch.reciprocal(normCm(B)));
        
        self.register_buffer('DT', torch.transpose(wignerCoeffs(B), 0, 1));
        
    def forward(self, Psi):
        
        # Psi: b x M x L
        
        B, b = self.B, Psi.size()[0]
        
        psiHat = torch.reshape(torch.transpose(torch.mm(self.DT, torch.transpose(torch.reshape(Psi, (b, (2*B - 1)*B)), 0, 1)), 0, 1), (b, 2*B - 1, 2*B))

         #Apply DCT + DST to even + odd indexed m
        psiHat[:, self.cInd, :] = self.dct(psiHat[:, self.cInd, :]);
        psiHat[:, self.sInd, :] = self.dst(psiHat[:, self.sInd, :]);
 
        # f: b x theta x phi
        return torch.mul(self.iCm[None, :, None], psiHat);


#############################################################
################ Spherical Harmonic Transforms ##############
#############################################################


class FTSHT(nn.Module):
    
    '''
    The Forward "Tensorized" Discrete Spherical Harmonic Transform
    
    Input:
    
    B: (int) Transform bandlimit
    
    '''
    def __init__(self, B):
        super(FTSHT, self).__init__()
        
        self.B = B;
        
        self.FDL = FDLT(B);
        
    def forward(self, psi):
        
        '''
        Input:
        
        psi: ( b x 2B x 2B torch.double or torch.cdouble tensor ) 
             Real or complex spherical signal sampled on the 2B X 2B DH grid with b batch dimensions
           
        Output:
        
        Psi: (b x (2B - 1) x B torch.cdouble tensor)
             Complex tensor of SH coefficients over b batch dimensions
            
        '''
        
        # psi: b x theta x phi (real or complex) 
        B, b = self.B, psi.size()[0]
        
        ## FFT in polar component
        # psiHat: b x  M x Phi
        
        psiHat = torch.fft.fftshift(torch.fft.fft( psi, dim=1, norm='forward'), dim=1)[:, 1:, :]

        ## Convert to real representation
        psiHat = torch.reshape(torch.permute(torch.view_as_real(psiHat), (0, 3, 1, 2)), (2*b, 2*B - 1, 2*B));
        
        # Forward DLT
        Psi = self.FDL(psiHat); 
        
        # Convert back to complex and return
        # Psi: b x M x L (complex)
        
        return torch.view_as_complex(torch.permute(torch.reshape(Psi, (b, 2, 2*B-1, B)), (0, 2, 3, 1)));
                                     
        
class ITSHT(nn.Module):
    
    '''
    The Inverse "Tensorized" Discrete Spherical Harmonic Transform
    
    Input:
    
    B: (int) Transform bandlimit
    
    '''
        
    def __init__(self, B):
        super(ITSHT, self).__init__()
        
        self.B = B;
        
        self.IDL = IDLT(B);

    def forward(self, Psi):
        
        ''' 
        Input:
        
        Psi: (b x (2B - 1) x B torch.cdouble tensor)
             Complex tensor of SH coefficients over b batch dimensions
             
        Output:
        
        psi: ( b x 2B x 2B torch.cdouble tensor ) 
             Complex spherical signal sampled on the 2B X 2B DH grid with b batch dimensions
            
        '''
        
        # Psi: b x  M x L (complex)
        B, b = self.B, Psi.size()[0];
        
        # Convert to real
        Psi = torch.reshape(torch.permute(torch.view_as_real(Psi), (0, 3, 1, 2)), (2*b, 2*B-1, B));
        
        # Inverse DLT
        psiHat = self.IDL(Psi);
        
        # Convert back to complex
        psiHat = torch.view_as_complex(torch.permute(torch.reshape(psiHat, (b, 2, 2*B - 1, 2*B)), (0, 2, 3, 1)));
        
        ## Set up for iFFT
        psiHat = torch.cat( (torch.empty(b, 1, 2*B, device=psiHat.device).float().fill_(0), psiHat), dim=1);
        
        # Inverse FFT and return
        # psi: b x theta x phi (complex)
        
        return torch.fft.ifft( torch.fft.ifftshift(psiHat, dim=1), dim=1, norm='forward');


        
      
          
