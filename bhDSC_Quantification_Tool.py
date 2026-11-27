##Modified from https://github.com/JSchul1998/bhDSC_Scripts (Schulman et al., 2024, Sci Rep 14:7322)
import argparse
import os
import numpy as np 
import nibabel as nib
from sklearn import metrics
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import circulant
import matplotlib.pyplot as plt

##Command-line arguments (replace the hardcoded file paths and scan parameters; defaults reproduce the original script)
parser = argparse.ArgumentParser(description="bhDSC CBV/CBF/MTT quantification (Schulman et al., 2024)")
parser.add_argument("--relax", required=True, help="Average_Relaxation.nii.gz from the averaging tool")
parser.add_argument("--aif", required=True, help="AIF mask NIfTI (voxels > 0 are averaged)")
parser.add_argument("--vof", required=True, help="VOF mask NIfTI (voxels > 0 are averaged)")
parser.add_argument("--gm", required=True, help="GM mask NIfTI (voxels > 0 are used)")
parser.add_argument("--wm", required=True, help="WM mask NIfTI (voxels > 0 are used)")
parser.add_argument("--outdir", required=True, help="output folder")
parser.add_argument("--tr", type=float, default=2.0, help="repetition time (s)")
parser.add_argument("--thresh", type=float, default=0.2, help="SVD threshold (standard 0.2 for sSVD; 0.1 for cSVD)")
parser.add_argument("--art-time", type=float, nargs=6, default=[2, 4, 6, 8, 10, 12], metavar="S",
                    help="artery bounds in the averaged window (s): pre-baseline start/end, bolus start/end, post-baseline start/end")
parser.add_argument("--tis-time", type=float, nargs=4, default=[2, 4, 10, 12], metavar="S",
                    help="tissue baseline bounds (s): pre-baseline start/end, post-baseline start/end (bolus = artery bounds)")
parser.add_argument("--ven-time", type=float, nargs=6, default=[2, 4, 6, 8, 10, 12], metavar="S",
                    help="vein bounds (s): pre-baseline start/end, bolus start/end, post-baseline start/end")
parser.add_argument("--no-show", action="store_true", help="save the QC plot as PNG in --outdir instead of showing it")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

def show_or_save(name):
    if args.no_show:
        plt.savefig(os.path.join(args.outdir, name))
        plt.close()
    else:
        plt.show()

TR = args.tr
thresh = args.thresh

##Load the relaxation file (output from bolus averager file), AIF mask, VOF mask, GM mask, and WM mask
relax_n = nib.load(args.relax)
relax = relax_n.get_fdata()
aif_data = nib.load(args.aif)
aif_data = aif_data.get_fdata()
vof_data = nib.load(args.vof)
vof_data = vof_data.get_fdata()
gm_data = nib.load(args.gm)
gm_data = gm_data.get_fdata()
wm_data = nib.load(args.wm)
wm_data = wm_data.get_fdata()

##File co-ordinates 
x = relax.shape[0]
y = relax.shape[1]
z = relax.shape[2]
t = relax.shape[3]
T = range(relax.shape[3])

##Bolus timebounds are given in seconds and converted to volumes of the averaged window with the TR (defaults = original volumes at TR = 2 s)
##Artery (first two (pre-baseline); second two (bolus); final two (post-baseline))
art_time = np.array([int(round(s / TR)) for s in args.art_time])
##Tissue (first two (pre-baseline); second two (post-baseline)) --> bolus timebounds are same as those for artery above.
tis_time = np.array([int(round(s / TR)) for s in args.tis_time])
##Vein (first two (pre-baseline); second two (bolus); final two (post-baseline))
ven_time = np.array([int(round(s / TR)) for s in args.ven_time])


##Obtain average time-course fofr AIF and VOF
aif = []
vof = []
for i in range(x):
    print("Progression of Fitting ", (i/relax.shape[0])*100,"%")
    for j in range(y):
        for k in range(z):
            mask = aif_data[i,j,k]
            mask1 = vof_data[i,j,k]
            data = relax[i,j,k]
            if mask > 0:
                aif.append(data)
            if mask1 > 0:
                vof.append(data)
                
##Takes average of AIF and VOF voxels --> Vertically shifts AIF and VOF by the baseline --> Scales AIF integral by VOF integral --> Plots scaled AIF and VOF
AIFav = (np.array(aif)).mean(0)
AIF = ((AIFav[art_time[2]:art_time[3]] - (np.mean(AIFav[art_time[0]:art_time[1]]) + np.mean(AIFav[art_time[4]:art_time[5]]))/2))
##IF YOU ARE NOT USING NOVEL AIF (Schulman et al., 2023) with positive relaxation change, UNCOMMENT the next line
#AIF = -((AIFav[art_time[2]:art_time[3]] - (np.mean(AIFav[art_time[0]:art_time[1]]) + np.mean(AIFav[art_time[4]:art_time[5]]))/2))
VOFav = (np.array(vof)).mean(0)
VOF = -((VOFav[ven_time[2]:ven_time[3]] - (np.mean(VOFav[ven_time[0]:ven_time[1]]) + np.mean(VOFav[ven_time[4]:ven_time[5]]))/2))
AIFv = AIF*((metrics.auc(np.array(range(len(VOF))), VOF))/(metrics.auc(np.array(range(len(AIF))), AIF)))
plt.plot(np.array(range(len(AIF))),AIFv, color='red')
plt.plot(np.array(range(len(VOF))),VOF)
show_or_save('AIF_VOF.png')

##Function to transform AIF into appropriate matrix form for deconvolution
def AIFmatrixMaker(aif):
    mat = []
    for l in range(0,len(range(art_time[2],art_time[3]))):
        a = aif
        A = np.delete(a, np.s_[len(range(art_time[2],art_time[3]))-l:len(range(art_time[2],art_time[3]))])
        b = np.pad(A, (l, 0), 'constant', constant_values=(0, 0))
        B = np.array(b)
        mat.append(B)
    mat = np.transpose(mat)
    return mat
AIFmatfilt = AIFmatrixMaker(AIFv)

##Performs SVD with threshold (defined above) and outputs U,S,and Vt SVD matrices
b1, b2, b3 = randomized_svd(AIFmatfilt, n_components = len(range(art_time[2],art_time[3])))
truncation1 = np.max(np.where(b2 >= thresh*np.max(b2))) + 1
c1, c2, c3 = randomized_svd(AIFmatfilt, n_components = truncation1)
C1, C2, C3 = c1.T, np.diag(c2), c3.T
C2i = np.linalg.inv(C2)

##Performs iterative CBV/CBF/MTT quantification for all voxels in nii relaxation file
A=[]
B=[]
C=[]
cbv_gm=[]
cbv_wm=[]
cbf_gm=[]
cbf_wm=[]
mtt_gm=[]
mtt_wm=[]
for i in range(x):
    print("Progression of CBV/CBF/MTT Calculation ", (i/relax.shape[0])*100,"%")
    for j in range(y):
        for k in range(z):
            a = -((relax[i, j, k, art_time[2]:art_time[3]] - (np.mean(relax[i, j, k, tis_time[0]:tis_time[1]]) + np.mean(relax[i, j, k, tis_time[2]:tis_time[3]]))/2))
            gm_mask = gm_data[i,j,k]
            wm_mask = wm_data[i,j,k]
            
            ##Hematocrit correction factor when using dOHb contrast (different for Gd contrast)
            hema = 1/0.69

            ##For all voxels with no data
            if np.mean(a) == 0:
                A.append(0)
                B.append(0)
                C.append(0)
            
            else:
                ##Calculates CBV (refer to Schulman et al., 2023)
                CBV = (metrics.auc(np.array(range(len(a))), a))/(metrics.auc(np.array(range(len(a))), AIFv))              
                if CBV > 0:
                    cbv = ((100*hema)/1.05)*CBV
                    A.append(cbv)
                    if gm_mask > 0:
                        cbv_gm.append(cbv)
                    if wm_mask > 0:
                        cbv_wm.append(cbv)
                else:
                    A.append(0)
                    
                ##Calculates residue function and CBF via deconvolution (refer to Schulman et al., 2023)
                Residue = np.transpose([np.matmul(C3, np.matmul(C2i, np.matmul(C1, np.transpose(a))))])/TR
                max_value = (max([max(n) for n in Residue]))
                if CBV > 0:
                    cbf = ((((100*hema)/1.05)*max_value)*60)
                    B.append(cbf)
                    if gm_mask > 0:
                        cbf_gm.append(cbf)
                    if wm_mask > 0:
                        cbf_wm.append(cbf)
                else:
                    B.append(0)

                ##Calculates MTT via central volume principle (refer to Schulman et al., 2023)
                if max_value == 0 or CBV <= 0:
                    C.append(0)
                else:
                    mtt = (CBV/max_value)
                    C.append(mtt)
                    if gm_mask > 0:
                        mtt_gm.append(mtt)
                    if wm_mask > 0:
                        mtt_wm.append(mtt)
A = np.array(A)
B = np.array(B)
C = np.array(C)

##Outputs the CBV, CBF, and MTT Files into --outdir
A = A.reshape(x, y, z)
image = nib.Nifti1Image(A, relax_n.affine)
nib.save(image, os.path.join(args.outdir, 'CBV.nii.gz'))
B = B.reshape(x, y, z)
image = nib.Nifti1Image(B, relax_n.affine)
nib.save(image, os.path.join(args.outdir, 'CBF.nii.gz'))
C = C.reshape(x, y, z)
image = nib.Nifti1Image(C, relax_n.affine)
nib.save(image, os.path.join(args.outdir, 'MTT.nii.gz'))

##Calculates the average GM and WM CBV, CBF, and MTT for the given file
cbv_gm = np.array(cbv_gm)
cbv_wm = np.array(cbv_wm)
cbf_gm = np.array(cbf_gm)
cbf_wm = np.array(cbf_wm)
mtt_gm = np.array(mtt_gm)
mtt_wm = np.array(mtt_wm)
print("CBV GM = ", round(np.mean(cbv_gm),2))
print("CBV WM = ", round(np.mean(cbv_wm),2))
print("CBF GM = ", round(np.mean(cbf_gm),2))
print("CBF WM = ", round(np.mean(cbf_wm),2))
print("MTT GM = ", round(np.mean(mtt_gm),2))
print("MTT WM = ", round(np.mean(mtt_wm),2))
