# In the new hydra IFT cluster run with python3.8
import os
import warnings
import numpy as np
from math import pi
import scipy.constants as conts
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.signal import savgol_filter

# uncomment to get plots displayed in notebook
# %matplotlib inline
#import matplotlib.pyplot as plt

# import classy module
#!pip install classy==2.9.4
#!pip install classy
from classy import Class

def bias(z):
  return 0.648+0.689*np.exp(0.792*z)

# Cosmological parameters and survey specifications 

# The step in the derivative
epsilon_0 = 0.01

#The redshift error
sigma_0 = 0.001 

# de-wiggling parameters
window_size, poly_order = 201, 3

# The cosmo params
cosmo_params = {
    'Omega_b' : 0.05,
    'Omega_m' : 0.32,
    'N_ur': 2.0308,
    'N_ncdm': 1,
    'm_ncdm': 0.06,
    'h' : 0.67,
    #'A_s' : 2.0545e-9, # changing this is equivalent to fixing \sigma_8
    'ln10^{10}A_s' : 3.022617602335373,
    'n_s' : 0.96
}

# This is the fiducial value for the non-linear parameters \sigma_v and \sigma_p at z=0 using Eq. (81) of 1910.09273. 
# It depends on the cosmology, so has to be adjusted (see the calculation below) if the fiducial cosmology changes!
sigma_vp0_fid =  8.425042478765544

# The default fisher params (redshifts and k values for the grids etc)
fisher_params = {'z_min': 0.5, 'z_max':1.0, 'mu_bin':0, 'n_bins': 0.42875e-4, 'A_survey':912.04, 'sigma_v':sigma_vp0_fid, 'sigma_p':sigma_vp0_fid, 'P_s':0}
zed = np.arange(0.0, 4.0, 0.1)
k_max = 0.25
k_grid = np.logspace(-4,np.log10(k_max),1000) # k in 1/Mpc

# The mu grid
mu_grid = np.linspace(-1.0, 1.0, num=21)
print('The \mu grid: ',mu_grid)

# The redshift bins
z_mins = [0.50, 1.00, 1.50, 2.00]
z_maxs = [1.00, 1.50, 2.00, 3.00]

# The mean redshift at the bins
z_mean = []
for ii in range(len(z_mins)):
  z_mean.append(0.5*(z_mins[ii]+z_maxs[ii]))
print('z_mean = ',z_mean)
print('')

# The observed angle and area in every redshift bin, in deg and deg^2 respectively
omega = [30.2, 20.9, 16.9, 14.0]
area_bins = [ang**2 for ang in omega]

# The number of objects
Mpc_o_h= 1.
Gpc_o_h = 1.e3*Mpc_o_h
dN_dim =[8.e3, 1.6e4, 1.2e4, 8.0e3] 
dN = [dd*Gpc_o_h**(-3) for dd in dN_dim]
n_bins = [dN[i]*(1+z_mean[i])**3 for i in range(len(dN))]

# The number of galaxies per unit area and redshift intervals
#dN_dO_dz_bins = [1815.0, 1701.5, 1410.0, 940.97]

# Which parameters to consider in the fisher matrix: 5 cosmo and 3 "nuisance"
deriv_params_all = ['Omega_b','Omega_m','m_ncdm','h','ln10^{10}A_s','n_s','sigma_v','sigma_p','P_s']
labels_all = [r'$\Omega_b$',r'$\Omega_m$',r'$\Sigma m_{\nu}$','$h$','$\ln(10^{10}A_s)$','$n_s$',r'$\sigma_v$',r'$\sigma_p$',r'$P_s$']

# Pick which ones you want from the list above
index = range(len(deriv_params_all))
#index = [1,2]
deriv_params = [deriv_params_all[ii] for ii in index]
labels = [labels_all[ii] for ii in index]
dim = len(deriv_params)

# Print the params chosen:
print('We will use the following',len(deriv_params),'params for the F_{ij}:')
print(deriv_params)

# A quick run to get the fiducial non-linear parameters and to show off the de-wiggling process
LambdaCDM = Class()
LambdaCDM.set(cosmo_params)
LambdaCDM.set({'output':'mPk','lensing':'no','P_k_max_1/Mpc':3.0,'z_max_pk': '5.1'})
LambdaCDM.compute()

# The fiducial value if given by the integral of Eq. (81) of 1910.09273
# Calculate the parameter at z=0 and the redshift bins
for zmi in ([0]+z_mean):
  Pk_dd = [] 
  for k in k_grid:
      Pk_dd.append(LambdaCDM.pk(k, zmi))
  print('[z,sigma_v_fid] = ',[round(zmi,2),np.sqrt(np.trapz(Pk_dd, x = k_grid)/(6.*np.pi**2))])
LambdaCDM.empty()

# This does the de-wiggling as an example. I had to fine-tune the window-size a bit to make it work wel
itp = interp1d(k_grid,Pk_dd, kind='linear')
window_size, poly_order = 201, 3
pkfy_sg = savgol_filter(itp(k_grid), window_size, poly_order)

# A comparison between the wiggly (=with BAO oscillations) and de-wiggled power spectrum 
# plt.figure(1)
# plt.xscale('log');
# plt.yscale('log');
# plt.xlabel(r'$k \,\,\,\, [1/\mathrm{Mpc}]$')
# plt.ylabel(r'$d P(k)\,\,\,\, [\mathrm{Mpc}]^3$')
# plt.plot(k_grid,Pk_dd,label=r'$P_{\delta\delta}(k)$')
# plt.plot(k_grid,pkfy_sg,label=r'$P_{dw}(k)$')
# plt.legend(loc='upper left')
# plt.show()

def cosmo(params_cosmo, params_fisher):
  dA = np.array([], 'float64')
  H = np.array([], 'float64')
  s8 =np.array([], 'float64')
  f =np.array([], 'float64')
  fs8 =np.array([], 'float64')
  # Some params
  z_bin = 0.5*(params_fisher['z_min']+params_fisher['z_max'])
  mu = params_fisher['mu_bin']
  # create an instance of the class "Class"
  LambdaCDM = Class()
  # and pass the input parameters
  LambdaCDM.set(params_cosmo)
  LambdaCDM.set({'output':'mPk','lensing':'no','P_k_max_1/Mpc':3.0,'z_max_pk': '5.1'})
  # run class
  LambdaCDM.compute()
  # Some more params:
  c_over_H_z = 1/LambdaCDM.Hubble(z_bin) #in units of Mpc
  sigma_r_z = c_over_H_z*sigma_0*(1+z_bin)
  # Extract the background values from CLASS
  for zz in zed:
    dA = np.append(dA, LambdaCDM.angular_distance(zz))
    H = np.append(H, LambdaCDM.Hubble(zz)*conts.c /1000.0)
    s8 = np.append(s8, LambdaCDM.sigma(8./LambdaCDM.h(),zz))
    f = np.append(f,LambdaCDM.scale_independent_growth_factor_f(zz))
    fs8 = np.append(fs8,LambdaCDM.scale_independent_f_sigma8(zz))
  # sigma_8(z=0)
  s8_0 = LambdaCDM.sigma(8./LambdaCDM.h(), 0)
  # The values of some params on the z_bin
  s8_z = LambdaCDM.sigma(8./LambdaCDM.h(), z_bin)
  f_z = LambdaCDM.scale_independent_growth_factor_f(z_bin)
  fs8_z = LambdaCDM.scale_independent_f_sigma8(z_bin)
  # The effective volume
  Omega = params_fisher['A_survey']*(np.pi/180)**2
  Delta_z = params_fisher['z_max']-params_fisher['z_min']
  r_min = LambdaCDM.angular_distance(params_fisher['z_min'])*(1+params_fisher['z_min'])
  r_max = LambdaCDM.angular_distance(params_fisher['z_max'])*(1+params_fisher['z_max'])
  V_s = Omega/3*(r_max**3-r_min**3)
  #n = params_fisher['dN_dO_dz']*params_fisher['A_survey']*Delta_z/V_s
  n = params_fisher['n_bins']
  V_eff = []
  # Collect everything that doesn't have a k dependence
  # The non-linear params
  sigma_v_z = params_fisher['sigma_v']*s8_z/s8_0
  sigma_p_z = params_fisher['sigma_p']*s8_z/s8_0
  # The linear prefactor
  prefactor = (bias(z_bin)*s8_z+fs8_z*mu**2)**2/s8_z**2
  # The g_mu function
  g_mu = sigma_v_z**2*(1.-mu**2+mu**2*(1+f_z)**2)

  # Here we get the linear spectrum 
  Pk_dd = [LambdaCDM.pk(k, z_bin) for k in k_grid] 

  # Here we de-wiggle it
  itp = interp1d(k_grid,Pk_dd, kind='linear')
  Pk_nw = savgol_filter(itp(k_grid), window_size, poly_order)

  # The final observed spectrum
  Pk_obs = [] 

  # Loop of the wavenumbers and get the observed P(k)
  for ii in range(len(k_grid)):
    k = k_grid[ii]
    F = np.exp(-k**2*mu**2*sigma_r_z**2)
    non_linear = 1./(1.+(f_z*k*mu*sigma_p_z)**2)

    # This is given by Eq. (83)
    Pk_dw = Pk_dd[ii]*np.exp(-g_mu*k**2)+Pk_nw[ii]*(1.-np.exp(-g_mu*k**2))

    # This is given by Eq. (87)
    Pk_zs = non_linear*prefactor*Pk_dw*F+params_fisher['P_s']

    # Append the values and prepare to wrap up
    Pk_obs.append(Pk_zs) # P(k) in (Mpc)**3
    V_eff.append(V_s*((n*Pk_zs)/(n*Pk_zs+1.))**2)

  LambdaCDM.empty()
  return [[k_grid,Pk_obs,V_eff], [zed, dA, H, s8, f,fs8]]

# This calculates the final quantities for the fiducial cosmology
# Very useful for making plots and to make sure things are ok
data = cosmo(cosmo_params,fisher_params);

# fig, axs = plt.subplots(2, 3)
# fig.set_figheight(10)
# fig.set_figwidth(20)
# fig.tight_layout(pad=3.0)
# axs[0, 0].plot(zed, data[1][1])
# axs[0, 0].set_xlabel(r'$z$')
# axs[0, 0].set_ylabel(r'$d_A(z)~~~[\mathrm{Mpc}]$')
# axs[0, 1].plot(zed, data[1][2], 'tab:orange')
# axs[0, 1].set_xlabel(r'$z$')
# axs[0, 1].set_ylabel(r'$H(z)~~~[\mathrm{km}\mathrm{s}^{-1}\mathrm{Mpc}^{-1}]$')
# axs[1, 0].plot(zed, data[1][3], 'tab:green')
# axs[1, 0].set_xlabel(r'$z$')
# axs[1, 0].set_ylabel(r'$\sigma_8(z)$')
# axs[1, 1].plot(zed, data[1][5], 'tab:red')
# axs[1, 1].set_xlabel(r'$z$')
# axs[1, 1].set_ylabel(r'$f\sigma_8(z)$')
# axs[0, 2].plot(k_grid, data[0][1], 'tab:red')
# axs[0, 2].set_xscale('log')
# axs[0, 2].set_yscale('log')
# axs[0, 2].set_xlabel(r'$k \,\,\,\, [1/\mathrm{Mpc}]$')
# axs[0, 2].set_ylabel(r'$P_{obs}(k) \,\,\,\, [\mathrm{Mpc}]^3$')
# axs[1, 2].plot(k_grid, data[0][2], 'tab:red')
# axs[1, 2].set_xscale('log')
# axs[1, 2].set_yscale('log')
# axs[1, 2].set_xlabel(r'$k \,\,\,\, [1/\mathrm{Mpc}]$')
# axs[1, 2].set_ylabel(r'$V_\mathrm{eff}(k)$');

def calc_derivs(params_cosmo, params_fisher, data_0, epsilon, param):
  P_0 = np.asarray(data_0[0][1])
  # Here put an if statement for P_s as the derivative is given by Eq.(99) in the Euclid paper
  if param == 'P_s':
    dP_dp = 1. 
  elif param == 'sigma_v' or param == 'sigma_p':
    params_m = params_fisher.copy()
    params_p = params_fisher.copy()
    params_m[param]*=(1.-epsilon)
    params_p[param]*=(1.+epsilon)
    data_m = cosmo(params_cosmo,params_m)
    data_p = cosmo(params_cosmo,params_p)
    dP_dp = (np.asarray(data_p[0][1])-np.asarray(data_m[0][1]))/(2*epsilon*params_fisher[param])
  else:
    params_m = params_cosmo.copy()
    params_p = params_cosmo.copy()
    params_m[param]*=(1.-epsilon)
    params_p[param]*=(1.+epsilon)
    data_m = cosmo(params_m,params_fisher)
    data_p = cosmo(params_p,params_fisher)
    dP_dp = (np.asarray(data_p[0][1])-np.asarray(data_m[0][1]))/(2*epsilon*params_cosmo[param])
  return dP_dp/P_0

dPk = [calc_derivs(cosmo_params, fisher_params, data, epsilon_0, params) for params in deriv_params];

# fig=plt.figure(1)
# fig.set_figheight(7)
# fig.set_figwidth(10)
# plt.xscale('log');
# plt.yscale('log');
# plt.xlabel(r'$k \,\,\,\, [1/\mathrm{Mpc}]$')
# plt.ylabel(r'$d \ln P(k)/dp \,\,\,\, [\mathrm{Mpc}]^3$')
# for ii in range(len(deriv_params)):
  # plt.plot(k_grid,np.abs(dPk[ii]),label=labels[ii])
# plt.legend(loc='lower right')
# plt.show()

# Some variables
Veff = data[0][2]

# Initialize some params
Fij_tot = np.asarray([ [0.]*dim for i in range(dim)])
Fij_bin_z = np.asarray([ [0.]*dim for i in range(dim)])
fisher_params_mu = fisher_params.copy()

# Here we do the loop over the redshifts
for zi in range(len(z_mins)):
  fisher_params_mu['z_min'] = z_mins[zi]
  fisher_params_mu['z_max'] = z_maxs[zi]
  fisher_params_mu['n_bins'] = n_bins[zi]
  fisher_params_mu['A_survey'] = area_bins[zi]
  print('Now working on bin:',zi)

  # Loop over indices i,j and mu
  for j in range(dim):
    for i in range(j,dim):  
      print('Now working on Fij:',[i,j])
      Fij_mu = []
      for mu in mu_grid:
        fisher_params_mu['mu_bin'] = mu
        dPks = [calc_derivs(cosmo_params, fisher_params_mu, data, epsilon_0, params) for params in deriv_params];
        integrand = dPks[i]*dPks[j]*Veff*k_grid**2/(8*np.pi**2)
        Fij_mu.append(np.trapz(integrand, x = k_grid))
      Fij_bin_z[i][j] = np.trapz(Fij_mu, x=mu_grid)
      # Make the Fisher matrix symmetric again!
      Fij_bin_z[j][i] = Fij_bin_z[i][j]

  # Add the contribution from the bin to get the total
  Fij_tot += Fij_bin_z

# Print the values to see how it looks
print(Fij_tot)

# Save the Fij
np.savetxt('fij_param_matrix.txt',Fij_tot)

# Read the Fij values from a file for debugging purposes
Fij_tot = np.loadtxt('fij_param_matrix.txt')

# Get the errors in the cosmo parameter plane.
errs = []
for i in range(dim):
  errs.append(np.sqrt(np.linalg.inv(Fij_tot)[i][i]))
print(errs)

def fs8_cosmo(params_cosmo):
  fs8 =np.array([], 'float64')
  # create instance of the class "Class"
  LambdaCDM = Class()
  # pass input parameters
  LambdaCDM.set(params_cosmo)
  LambdaCDM.set({'output':'mPk','lensing':'no','P_k_max_1/Mpc':3.0,'z_max_pk': '5.1'})
  # run class
  LambdaCDM.compute()
  # Extract the background values from CLASS
  for zz in zed:
    fs8 = np.append(fs8,LambdaCDM.scale_independent_f_sigma8(zz))
  LambdaCDM.empty()
  return [zed,fs8]

def calc_fs8_derivs(params_cosmo, epsilon, param):
  # Here put an if statement for P_s as the derivatie is given by (99) in the Euclid paper
  if param == 'P_s':
    dfs8_dp = [0.]*len(zed)
  elif param == 'sigma_v' or param == 'sigma_p':
    dfs8_dp = [0.]*len(zed)
  else:
    params_m = params_cosmo.copy()
    params_p = params_cosmo.copy()
    params_m[param]*=(1.-epsilon)
    params_p[param]*=(1.+epsilon)
    data_m = fs8_cosmo(params_m)
    data_p = fs8_cosmo(params_p)
    dfs8_dp = (np.asarray(data_p[1])-np.asarray(data_m[1]))/(2*epsilon*params_cosmo[param])
  return [zed,dfs8_dp]

# Here we do the actual calculation of the derivatives of fs8
d_fs8_dp = [calc_fs8_derivs(cosmo_params, epsilon_0, params) for params in deriv_params]

# fig=plt.figure(1)
# fig.set_figheight(7)
# fig.set_figwidth(10)
# plt.figure(1)
# plt.xlabel(r'$z$')
# plt.ylabel(r'$df\sigma_8(z)/dp$')
# for ii in range(len(deriv_params)):
  # plt.plot(d_fs8_dp[0][0],d_fs8_dp[ii][1],label=labels[ii])
# plt.legend(loc='lower left')
# plt.show()

# Here we get the derivatives
dfs8_dp_int = []
for ii in range(len(d_fs8_dp)):
  dfs8_dp_int.append(interp1d(zed, d_fs8_dp[ii][1]))

# And here we get the actual jacobian
jacob = np.zeros((len(z_mean),len(dfs8_dp_int)))

for ii in range(len(z_mean)):
  for jj in range(len(dfs8_dp_int)):
    jacob[ii,jj] = dfs8_dp_int[jj](z_mean[ii])
print(jacob)

np.savetxt('jacobian_params_to_fs8_matrix.txt',jacob)

# Here we do the error propagation and get the errors -> \sigma fs8_a = C_{a,a}^{1/2}
Cij = np.linalg.inv(Fij_tot)
Cab = np.dot(np.dot(jacob,Cij),np.transpose(jacob))
errs_fs8 = np.sqrt(np.diagonal(Cab));
print('The errors are:',errs_fs8)
np.savetxt('fs8_errors.txt',errs_fs8)

# The normalized C_ab matrix
# Cab_norm = [ [0.]*len(errs_fs8) for i in range(len(errs_fs8))]
# for ii in range(len(errs_fs8)):
  # for jj in range(len(errs_fs8)):
    # Cab_norm[ii][jj] = Cab[ii][jj]/errs_fs8[ii]/errs_fs8[jj]
  
# print(Cab_norm)

# Plot the C_{ab}

# labels_fs8 = ['','z=1', 'z=1.2', 'z=1.4', 'z=1.65']

# fig = plt.figure()
# axes = fig.add_subplot(111)
 
# # using the matshow() function
# caxes = axes.matshow(Cab_norm, interpolation ='nearest')
# fig.colorbar(caxes)
 
# axes.set_xticklabels(labels_fs8)
# axes.set_yticklabels(labels_fs8)
 
# plt.show()