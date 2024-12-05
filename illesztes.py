import uproot
import numpy as np
from numpy import emath as em
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm, crystalball
from scipy.special import erf
from scipy.signal.windows import gaussian as sc_gauss

def gaussian(x, A, xbar, sigma):
    return  A * np.exp(-(x - xbar)**2 / (2 * sigma**2))

def tail(x,alpha=-2.2,n=0.7,xbar = 5280,sigma=15):
    n_over_alpha = n/abs(alpha)
    exp = np.exp(-0.5* (alpha**2))
    A = (np.float_power(n_over_alpha,n))*exp
    B =  n_over_alpha - abs(alpha)
    N = 1
    powerlaw = N*A / np.float_power(B-(x-xbar)/sigma,n)
    return powerlaw

def crystal_ball(x, alpha, xbar, sigma):
    n=1
    n_over_alpha = n/abs(alpha)
    A = (np.float_power(n_over_alpha,n))*np.exp(-0.5*(alpha**2))
    B =  n_over_alpha - abs(alpha)
    #N = 1

    masking = (x - xbar)/15 > -alpha

    gaussian = np.exp(-0.5*((x[masking]-xbar)/sigma)**2)
    powerlaw = A*0.5 / np.float_power(B-(x[~masking]-xbar)/sigma,n)

    result = np.zeros_like(x)
    result[masking] = gaussian
    result[~masking] = powerlaw

    return result

def crystalb(x, alpha, n, xbar, sigma):
    return crystalball.pdf(x,alpha,n,xbar,sigma)

datax = np.array([5005.2841626,  5011.47651123, 5017.66885986, 5023.8612085 , 5030.05355713,
 5036.24590576, 5042.43825439, 5048.63060303, 5054.82295166, 5061.01530029,
 5067.20764893, 5073.39999756, 5079.59234619, 5085.78469482, 5091.97704346,
 5098.16939209, 5104.36174072, 5110.55408936, 5116.74643799, 5122.93878662,
 5129.13113525, 5135.32348389, 5141.51583252, 5147.70818115, 5153.90052979,
 5160.09287842, 5166.28522705, 5172.47757568, 5178.66992432, 5184.86227295,
 5191.05462158, 5197.24697021, 5203.43931885, 5209.63166748, 5215.82401611,
 5222.01636475, 5228.20871338, 5234.40106201, 5240.59341064, 5246.78575928,
 5252.97810791, 5259.17045654, 5265.36280518, 5271.55515381, 5277.74750244,
 5283.93985107, 5290.13219971, 5296.32454834, 5302.51689697, 5308.70924561,
 5314.90159424, 5321.09394287, 5327.2862915 , 5333.47864014, 5339.67098877,
 5345.8633374,  5352.05568604, 5358.24803467, 5364.4403833 , 5370.63273193,
 5376.82508057, 5383.0174292,  5389.20977783, 5395.40212646, 5401.5944751,
 5407.78682373, 5413.97917236, 5420.171521  , 5426.36386963, 5432.55621826,
 5438.74856689, 5444.94091553, 5451.13326416, 5457.32561279, 5463.51796143,
 5469.71031006, 5475.90265869, 5482.09500732, 5488.28735596, 5494.47970459,
 5500.67205322, 5506.86440186, 5513.05675049, 5519.24909912, 5525.44144775,
 5531.63379639, 5537.82614502, 5544.01849365, 5550.21084229, 5556.40319092,
 5562.59553955, 5568.78788818, 5574.98023682, 5581.17258545, 5587.36493408,
 5593.55728271, 5599.74963135, 5605.94197998, 5612.13432861, 5618.32667725])

datay = np.array([1,    1,    2,    3,    3,    2,    5,    1,    2,    3,    8,    5,   10,    3,
    5,    9,    9,    4,   11,    7,    4,    9,   10,    8,    8,   10,   17,   12,
   15,    9,   22,   20,   32,   30,   44,   37,   50,   76,  109,  146,  239,  411,
  720, 1009, 1297, 1240,  999,  685,  408,  186,  118,   59,   40,   14,   19,   12,
   10,    4,    5,    4,    2,    4,    1,    3,    5,    3,    3,    1,    0,    3,
    3,    1,    2,    0,    0,    1,    0,    1,    0,    1,    1,    1,    0,    0,
    1,    1,    1,    1,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,
    0,    1])

# Masking and linspace
mask = datay > 1
points = np.linspace(datax[mask][0],datax[mask][-1],1000)
maximum = datay >= max(datay)
half_value = datay >= max(datay)/2
power_mask = points < 5280

# Defining the error and the initial params
yerr = np.sqrt(datay)[mask]
p0 = np.array([datay[maximum],datax[maximum],np.abs(datax[maximum]-datax[half_value][0])])

# Curve fits
popt, _ = curve_fit(gaussian, datax[mask], datay[mask], p0=p0, method = 'lm', absolute_sigma = True, nan_policy = 'omit')
a0 = (np.mean(datax)-popt[1])/popt[2]
print("a0:",a0)
p0 = [2,5280,15] # alpha, xbar, sigma
bounds=([1.7,5270,10],[10,5290,20])
popt2, _ = curve_fit(crystal_ball, datax[mask], datay[mask], p0=p0 , method = 'trf', bounds=bounds, sigma=yerr, absolute_sigma = True, nan_policy = 'omit')

# Defining N for the power tail pdf
alpha, _, sigma = popt2
n = 1
C = n/abs(alpha)/(n-1)*np.exp(-0.5*(alpha**2))
D = np.sqrt(0.5*np.pi)*(1 + erf(abs(alpha)/np.sqrt(2)))
N = 1/(sigma*(C + D))

# Plots
fig, ax = plt.subplots()
ax.errorbar(datax[mask], datay[mask],yerr = yerr,  fmt='+',mfc='black',mec='black' , ecolor='black', capsize=5, label='Entries and error')
print("Az értékek:",popt2)
#plt.plot(points, gaussian(points, *popt), label='Gauss Fit', color='green')
#ax.plot(datax[mask],datay[maximum]*crystal_ball(datax[mask],1000,0,5280), label='Crystal Ball Drawing')
ax.plot(points, datay[maximum]*crystal_ball(points, *popt2), label='Crystal Ball Fit', color='hotpink')
#ax.plot(points[power_mask],datay[maximum]*tail(points[power_mask]), label='Tail drawing')
#plt.plot(sc_gauss(20,2))
ax.legend()
plt.figtext(0.01,0.01, f"paramters of [alpha, n, xbar, sigma]: \n{np.around(popt2,3)} \n initials and bounds:\n {p0}\n{bounds}", va = 'bottom' , wrap = True)
fig.subplots_adjust(bottom=0.25) 
plt.show()
