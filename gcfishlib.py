import numpy as np
import scipy
import camb
import time, datetime

# Cosmology and constants
c = 299792.458 # km/s 

# Distance and cosmology functions stolen from GC-Fish
#     https://github.com/Alkistis/GC-Fish & https://github.com/didamarkovic/GC-Fish 
# Also see https://arxiv.org/pdf/astro-ph/9905116.pdf

# Planck best fit
h = 0.6774
H0 = 100.*h # (km/s) / Mpc

# Densities
Omega_b = 0.0486; Omega_dm = 0.2589
Omega_m = Omega_b + Omega_dm
Omega_de = 1. - Omega_m
#print Omega_m

# Dark Energy parameters
w0 = -1.0
wa = 0.0

# Functions for DE later
def w_integrand(z,Omega_m,Omega_de,w0,wa):
    return (1 + w0+wa*z/(z+1)) / (1+z)
def DE_evo(z,Omega_m,Omega_de,w0,wa): 
    return np.exp(3*scipy.integrate.romberg(lambda x: w_integrand(x,Omega_m,Omega_de,w0,wa), 0, z))
#print DE_evo(0.), DE_evo(1.) # should be 1

# Define E(z) = H(z)/H0
def E(z,Omega_m,Omega_de,w0,wa):
    Omega_k = 1. - Omega_m - Omega_de
    return np.sqrt( Omega_de*DE_evo(z,Omega_m,Omega_de,w0,wa) + Omega_m*(1+z)**3 + Omega_k*(1+z)**2 )
def H(z,Omega_m,Omega_de,w0,wa):
    return E(z,Omega_m,Omega_de,w0,wa)*H0

# Define the cosmological distances
def D_M(z,Omega_m,Omega_de,w0,wa):
    return scipy.integrate.romberg(lambda z: c/H0/E(z,Omega_m,Omega_de,w0,wa),0,z)
def D_A(z,Omega_m,Omega_de,w0,wa):
    return D_M(z,Omega_m,Omega_de,w0,wa) / (1+z)

# Function to calculate distances to convert to a Hubble diagram
def D_V(z,Omega_m,Omega_de,w0,wa):
    return ( c*z*D_A(z,Omega_m,Omega_de,w0,wa)**2 / H(z,Omega_m,Omega_de,w0,wa) )**(1./3.) # missing a factor of (1+z)**2 ??
#print D_A(1.,Omega_m,Omega_de,w0,wa), D_M(1.,Omega_m,Omega_de,w0,wa), D_V(1.,Omega_m,Omega_de,w0,wa)

# Get default distances
def get_DVPlanck(zlist):
    return np.array([D_V(z,Omega_m,Omega_de,w0,wa) for z in zlist])
def get_DAPlanck(zlist,gcfish=True):
    if gcfish: # faster, but slightly different to below
        return np.array([D_A(z,Omega_m,Omega_de,w0,wa) for z in zlist])
    else:
        pars = camb.set_params(H0=H0, ombh2=Omega_b*h**2, omch2=Omega_dm*h**2, mnu=0.06, omk=1.-Omega_de-Omega_dm)
        results = camb.get_results(pars) 
    return results.angular_diameter_distance(zlist)

# Case when you have to run CAMB, do it globally
pars_pl = camb.set_params(H0=H0, ombh2=Omega_b*h**2, omch2=Omega_dm*h**2, mnu=0.06, omk=1.-Omega_de-Omega_dm-Omega_b)
results_pl = camb.get_results(pars_pl)
def get_rd_Planck():
    z_tmp = [0.1]
    BAO = results_pl.get_BAO(z_tmp,pars_pl)
    rs_DV, H, DA, F_AP = BAO[0,0],BAO[0,1],BAO[0,2],BAO[0,3]
    DV = (c*z_tmp[0]*(1.+z_tmp[0])**2 * DA**2 / H )**(1./3.)
    return rs_DV*DV
def get_DMrsPlanck(zlist,gcfish=True):
    try: 
        len(zlist)
        zlist = np.array(zlist)
    except TypeError:
        zlist = np.array([zlist])
    DA = results_pl.angular_diameter_distance(zlist)
    return DA*(1.+zlist)/get_rd_Planck()

# Generate a string timestamp
def get_timestamp():
	# Calculate the number of seconds since the Brexit referrendum polls closed
	t_Brexit = '23/06/2016 21:00' # GMT/UTC, 22:00 BST
	t_Brexit = time.mktime(datetime.datetime.strptime(t_Brexit, "%d/%m/%Y %H:%M").timetuple())
	return '-'+str(int(time.time() - t_Brexit))

# Fisher Matrix manipulation functions originally from "Fisher Matrix to Error.ipynb" from Dida's CosmoDemos

# A function to round to significant figures, not decimal places
def roundsf(x, n):
    mask = np.logical_or(np.equal(np.abs(x),np.inf), np.equal(x,0.0))
    decades = np.power(10.0,n-np.floor(np.log10(np.abs(x)))-1); decades[mask] = 1.0
    return np.round(np.array(x)*decades,0)/decades

# A function to return the Fisher matrix and its indices
def read_fisher(fishfile, marged=True):

    # Read the fisher matrix from the given file
    fm = np.loadtxt(fishfile)
    parameters = None
    with open(fishfile,'r') as f:
        for line in f:
            if line[0]=='#' and parameters==None: parameters = line[1:].split()
            elif line[0]!='#' and parameters==None: raise Exception(line)
            else: break

    # Extract the list of redshifts and indices for the mini-fm for each redshift
    zs = []; indices = {}
    for i, par in enumerate(parameters):
        p = par.split('_')
        if len(p)>1: 
            par = float(p[1])
            if par not in zs: 
                zs.append(par)
                indices[par] = []
            indices[par].append(i)

    return fm, parameters, indices, zs

# If want error on DV, need to propagate errors on both H and DA to DV:
def DV_from_fm(fm, parameters, indices, zs, da='Da', marged=True):

    # Construct the mini-fms
    Cz = []; order = None
    for z in zs:
        fm2 = fm[indices[z],:][:,indices[z]]
        pars = [parameters[i] for i in indices[z]]
        selected = [i for i,s in enumerate(pars) if ('H' in s or da in s)]
        if 1: # This is for ignoring shot noise, which breaks for photo-z (haven't thought yet exactly why)
            ignore = [i for i,s in enumerate(pars) if ('Ps' in s)]
            fm2[ignore,:] = 1.e-16 # not constraining the parameter at all
            fm2[:,ignore] = 1.e-16 # not constraining the parameter at all
        if marged:
            C_tmp = scipy.linalg.inv(fm2)[selected,:][:,selected]
        else:
            C_tmp = scipy.linalg.inv(fm2[selected,:][:,selected])
        Cz.append(C_tmp)
        if order is None: 
            order = [pars[i].split('_')[0] for i in selected]
        elif order != [pars[i].split('_')[0] for i in selected]:
            raise Exception('WTF')
        
    # Derivatives of dlnDV/dlnDA and dlnDV/dlnH
    derivs = {'ln'+da:2./3., 'lnH':-1./3.}

    # Project using the defined derivatives above
    J = np.array([derivs[order[0]], derivs[order[1]]])
    errors = []
    for i, z in enumerate(zs):
        errors.append(np.sqrt(np.dot(J.T,np.dot(Cz[i],J))))
    
    return errors, zs

# If want error on DA alone:
def DA_from_fm(fm, parameters, indices, zs, da='Da', marged=True):
    # Construct the mini-fms
    Cz = []; order = None
    for z in zs:
        fm2 = fm[indices[z],:][:,indices[z]]
        pars = [parameters[i] for i in indices[z]]
        selected = [i for i,s in enumerate(pars) if (da in s)]
        if 1: # This is for ignoring shot noise, which breaks for photo-z (haven't thought yet exactly why)
            ignore = [i for i,s in enumerate(pars) if ('Ps' in s)]
            fm2[ignore,:] = 1.e-16 # not constraining the parameter at all
            fm2[:,ignore] = 1.e-16 # not constraining the parameter at all
        if marged:
            C_tmp = scipy.linalg.inv(fm2)[selected,:][:,selected]
        else:
            C_tmp = scipy.linalg.inv(fm2[selected,:][:,selected])
        Cz.append(C_tmp)
        if order is None: 
            order = [pars[i].split('_')[0] for i in selected]
        elif order != [pars[i].split('_')[0] for i in selected]:
            raise Exception('WTF')
        
    # Derivatives of dlnDA/dlnDA
    derivs = {'ln'+da:1.}

    # Project using the defined derivatives above
    J = np.array([derivs[order[0]]])
    errors = []
    for i, z in enumerate(zs):
        errors.append(np.sqrt(np.dot(J.T,np.dot(Cz[i],J))))
    
    return errors, zs

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Now use CAMB (see below for a small consistency test)
    # https://camb.readthedocs.io/en/latest/results.html
    z = np.arange(0.03,3.03,0.3)
    pars = camb.set_params(H0=H0, ombh2=Omega_b*h**2, omch2=Omega_dm*h**2, mnu=0.06, omk=1.-Omega_de-Omega_dm)
    results = camb.get_results(pars)

    # And a dynamical dark energy
    pars_dde = camb.set_params(H0=H0, ombh2=Omega_b*h**2, omch2=Omega_dm*h**2, mnu=0.06, omk=1.-Omega_de-Omega_dm)
    pars_dde.set_dark_energy(w=-0.55, wa=-1.32, dark_energy_model='ppf')
    results_dde = camb.get_results(pars_dde)
        
    if 0: # GCFish - CAMB consistency check
        #print(z,results.angular_diameter_distance(z))
        #print(results.angular_diameter_distance(z))
        #print(get_DAPlanck(z))
        plt.plot(z,results.angular_diameter_distance(z),label='CAMB')
        plt.plot(z,get_DAPlanck(z),label='GCFish')
        plt.xlabel('z')
        plt.ylabel('D_A')
        plt.legend()
        plt.savefig('test_Dang.png',dpi=300,bbox_inches='tight')
        plt.clf()
        
    if 1: # understanding BAO-related CAMB outputs better
        
        # Plot LCDM
        print('rs/DA at recombination in LCDM',results.cosmomc_theta()) # rs/DA at recombination
        print('LCDM BAO pars:\n',results.get_BAO(z,pars)) # returns rs/DV, H, DA, F_AP
        plt.plot(z,results.sound_horizon(z),label='LCDM')
        
        # Now plot w0waDCM
        print('rs/DA at recombination in w0waCDM',results_dde.cosmomc_theta()) # rs/DA at recombination
        print('w0wa BAO pars:\n',results_dde.get_BAO(z,pars_dde)) # returns rs/DV, H, DA, F_AP
        plt.plot(z,results_dde.sound_horizon(z),label='DESI best-fit dynamical DE')

        plt.xlabel('z')
        plt.ylabel('comoving sound horizon [Mpc]')
        plt.legend()
        plt.savefig('test_rs.png',dpi=300,bbox_inches='tight')
        plt.clf()

    print('comoving sound horizon today in LCDM = ',results.sound_horizon(0.))
    print('comoving sound horizon today in w0wa = ',results_dde.sound_horizon(0.))
