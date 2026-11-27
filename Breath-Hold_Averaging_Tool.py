##Modified from https://github.com/JSchul1998/bhDSC_Scripts (Schulman et al., 2024, Sci Rep 14:7322)
import argparse
import os
import numpy as np 
import nibabel as nib
from sklearn.utils.extmath import randomized_svd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks

##Command-line arguments (replace the hardcoded file paths and scan parameters; defaults reproduce the original script)
parser = argparse.ArgumentParser(description="bhDSC breath-hold averaging (Schulman et al., 2024)")
parser.add_argument("--signal", required=True, help="pre-processed 4D signal NIfTI")
parser.add_argument("--gm", required=True, help="GM mask NIfTI (voxels > 0 are used)")
parser.add_argument("--wm", required=True, help="WM mask NIfTI (voxels > 0 are used)")
parser.add_argument("--outdir", required=True, help="output folder")
parser.add_argument("--tr", type=float, default=2.0, help="repetition time (s)")
parser.add_argument("--te", type=float, default=0.02, help="echo time (s)")
parser.add_argument("--bound", type=float, default=40.0, help="half window around each bolus peak (s)")
parser.add_argument("--times", type=float, nargs=6, default=[0, 10, 20, 60, 60, 70], metavar="S",
                    help="t1..t6 in the averaged window (s): pre-baseline start/end, bolus start/end, post-baseline start/end")
parser.add_argument("--dist", type=float, default=40.0, help="minimum distance between breath-hold peaks (s)")
parser.add_argument("--cutoff", type=float, default=0.01, help="high-pass cut-off (Hz)")
parser.add_argument("--no-show", action="store_true", help="save the QC plots as PNG in --outdir instead of showing them")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

def show_or_save(name):
    if args.no_show:
        plt.savefig(os.path.join(args.outdir, name))
        plt.close()
    else:
        plt.show()

###Load the pre-processed signal file, GM mask, and WM mask
data_n = nib.load(args.signal)
data = data_n.get_fdata()
gm = nib.load(args.gm)
gm = gm.get_fdata()
wm = nib.load(args.wm)
wm = wm.get_fdata()

##File Co-ordinates
x = data.shape[0]
y = data.shape[1]
z = data.shape[2]
t = data.shape[3]
X = np.array(range(t))

##Window parameters are given in seconds and converted to volumes with the TR (defaults = original volumes at TR = 2 s)
TR, TE = args.tr, args.te
##Bound specifies half of the window size that is centred around the bolus peak
bound = int(round(args.bound / TR))
##These are key timepoints of the final averaged bolus (first two: pre-baseline; middle two: Bolus; final two: post-baseline)
t1, t2, t3, t4, t5, t6 = [int(round(s / TR)) for s in args.times]
##Minimum distance between breath-holds
dist_param = int(round(args.dist / TR))

##Parameters needed for the initial high-pass filter
T = t*TR
##fs is the sampling frequency in your scan (1/TR)
fs = 1 / TR
##High-pass filter frequency cut-off (to remove low-frequency signal variations, and in effect, detrend the data)
cutoff1 = args.cutoff
nyq = 0.5*fs
order = 2
##Defines high pass filter function
def butter_highpass_filter(data, cutoff1, fs, order):
    normal_cutoff = cutoff1 / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='hp', analog=False)
    out = filtfilt(b, a, data)
    return out

##This iterates through the signal time course map and applies the high-pass filter
Y = []
for i in range(x):
    print("Progression of Filtering", (i/data.shape[0])*100,"%")
    for j in range(y):
        for k in range(z):
            Data = data[i, j, k]
            Data_mean = np.mean(Data)
            if Data_mean == 0:
                Y.append(np.zeros(t))
            else:
                yy = butter_highpass_filter(Data, cutoff1, fs, order) + Data_mean
                Y.append(yy)
Y = np.array(Y)
Y = Y.reshape(x, y, z, t)

##Next, this iterates through high-pass filtered map and obtains an average GM time-course for reference (in order to identify breath-hold peaks)
GM = []
for i in range(x):
    print("Progression of Averaging", (i/data.shape[0])*100,"%")
    for j in range(y):
        for k in range(z):
            GMm = gm[i, j, k]
            a = Y[i, j, k]
            a_mean = np.mean(a)
            if GMm > 0 and a_mean != 0:
                GM.append(a)
GM = np.array(GM)
GM_av = GM.mean(0)

##Displays the average GM time-course and the associated peaks using the find_peaks tool
plt.plot(X, GM_av)
plt.hlines(np.mean(GM_av)+0.75*np.std(GM_av), 0, len(GM_av),linestyles='dashed', colors='black')
GM_av_p = find_peaks(GM_av, height=(np.mean(GM_av)+0.75*np.std(GM_av)), distance=dist_param)
GM_av_p = GM_av_p[0]
plt.scatter(GM_av_p, GM_av[GM_av_p], color='red', marker='x')
show_or_save('GM_peaks.png')

##Obtains the breath-hold averaged GM time course to determine where bolus and baseline are located
GM_new = []
for l in GM_av_p:
    if bound < l < (t - bound):
        GM_new.append(GM_av[l-bound-1:l+bound-1])
    else:
        pass
GM_new = (np.array(GM_new)).mean(0)

##Displays the breath-hold averaged GM time course and baseline bounds (can be changed at line 27)
plt.plot(np.array(range(len(GM_new))),GM_new)
plt.vlines(t1,np.mean(GM_new)-2*(np.std(GM_new)),np.mean(GM_new)+2*(np.std(GM_new)), linestyles='dashed', colors='black')
plt.vlines(t2,np.mean(GM_new)-2*(np.std(GM_new)),np.mean(GM_new)+2*(np.std(GM_new)), linestyles='dashed', colors='black')
plt.vlines(t5,np.mean(GM_new)-2*(np.std(GM_new)),np.mean(GM_new)+2*(np.std(GM_new)), linestyles='dashed', colors='black')
plt.vlines(t6,np.mean(GM_new)-2*(np.std(GM_new)),np.mean(GM_new)+2*(np.std(GM_new)), linestyles='dashed', colors='black')
show_or_save('GM_average_bolus.png')
print(len(GM_new))
print(np.argmax(GM_new))


##Finally, this iterates through the entire nii file to generate breath-hold averaged maps, CNR, and dS files
S = []
R = []
cnr = []
gmcnr = []
wmcnr = []
ds_gm = []
ds_wm = []
ds = []
for i in range(x):
    print("Progression of Iterating", (i/data.shape[0])*100,"%")
    for j in range(y):
        for k in range(z):
            Data = data[i, j, k]
            Gm = gm[i,j,k]
            Wm = wm[i,j,k]
            data_mean = np.mean(Data)    

            if data_mean == 0:
                S.append(np.zeros(2*bound))
                R.append(np.zeros(2*bound))
                cnr.append(0)
                ds.append(0)
            else:
                data_new = []
                for l in GM_av_p:
                    if bound < l < (t - bound):
                        data_new.append(Data[l-bound-1:l+bound-1])
                    else:
                        pass
                data_new = (np.array(data_new)).mean(0)
                S.append(data_new)
                Average_sig = (np.mean(data_new[t1:t2]) + np.mean(data_new[t5:t6]))/2

                ##Calculate CNR for voxel (in GM and WM)
                Peak = -(np.mean(Average_sig) - max(data_new[t3:t4]))
                dS = ((Peak)/np.mean(Average_sig))*100 
                std = (np.std(data_new[t1:t2]) + np.std(data_new[t5:t6]))/2
                CNR = Peak/std
                cnr.append(CNR)
                ds.append(dS)
                if Gm > 0:
                    gmcnr.append(CNR)
                    ds_gm.append(dS)
                if Wm > 0:
                    wmcnr.append(CNR)
                    ds_wm.append(dS)
                r = (-1/TE)*np.log(data_new/Average_sig)
                R.append(r)

S = np.array(S)
R = np.array(R)
cnr = np.array(cnr)
ds = np.array(ds)
GM_CNR = np.array(gmcnr)
WM_CNR = np.array(wmcnr)
dS_GM = np.array(ds_gm)
dS_WM = np.array(ds_wm)


##Outputs the breath-hold averaged time courses, CNR, and dS Maps into --outdir
newarr = S.reshape(x, y, z, len(GM_new))
new_image = nib.Nifti1Image(newarr, data_n.affine)
nib.save(new_image, os.path.join(args.outdir, 'Average_Signal.nii.gz'))
newarr = R.reshape(x, y, z, len(GM_new))
new_image = nib.Nifti1Image(newarr, data_n.affine)
nib.save(new_image, os.path.join(args.outdir, 'Average_Relaxation.nii.gz'))
newarr = cnr.reshape(x, y, z)
new_image = nib.Nifti1Image(newarr, data_n.affine)
nib.save(new_image, os.path.join(args.outdir, 'CNR.nii.gz'))
newarr = ds.reshape(x, y, z)
new_image = nib.Nifti1Image(newarr, data_n.affine)
nib.save(new_image, os.path.join(args.outdir, 'dS.nii.gz'))

##This final code computes the average GM and WM CNR and delta S values, along with the delta CNR (GM-WM)
dSg_Mean = np.mean(dS_GM)
dSw_Mean = np.mean(dS_WM)
CNRg_Mean = np.mean(GM_CNR)
CNRw_Mean = np.mean(WM_CNR)
CNRw_std = np.std(WM_CNR)
print("GM dS = ", dSg_Mean)
print("WM dS = ", dSw_Mean)
print("GM CNR = ", CNRg_Mean)
print("WM CNR = ", CNRw_Mean)
print("dCNR = ", (CNRg_Mean-CNRw_Mean)/(CNRw_std))
