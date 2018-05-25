from pylab import *
import scipy
from scipy import interpolate
from scipy import optimize
import emcee
import scipy.optimize as op
import corner
import pickle
import os
#import pymultinest
from astropy import constants
import matplotlib
import seaborn as sns
from matplotlib.patches import Ellipse

def get_mass(Ms, P, e, K, i):
	if K<0:
		K=1.
	G = constants.G.cgs.value
	Mjup = constants.M_jup.cgs.value
	Msun = constants.M_sun.cgs.value 

	masses = np.arange(0.0001,100.,0.1)*Mjup

	C1 = (2.*np.pi*G/P)**(1./3.)
	C2 = masses * np.sin(i) / ( (Ms * Msun + masses)**(2./3.) )
	C3 = 1./np.sqrt(1.-e**2)
	cte = C1 * C2 * C3 / K
	tck = interpolate.splrep(cte,masses,k=3)
	return interpolate.splev(1.,tck) / Mjup

def get_radius(Rs,p):
	return p * Rs*constants.R_sun.cgs.value / constants.R_jup.cgs.value

def get_teq(a,Rs,teff):
	return teff * np.sqrt(0.5*Rs/a)

def kep_third(Ms,Mp,P):
	G = constants.G.cgs.value
	Ms = Ms*constants.M_sun.cgs.value
	Mp = Mp*constants.M_jup.cgs.value
	aa = (G*P**2*(Ms + Mp)/(4.*np.pi**2))**(1./3.)
	return aa/(1.496e13)

def lin(params):
	ret = get_pars_YY_tracks(params[0],params[1],ret='TG')
	return ret

def res_lin(params,y,yerr):
	return (y-lin(params))/yerr

def end(mm,get='age'):
	a = np.array([ 4.5, 4.1, 4.0, 3.8, 3.6, 3.4, 3.2,3.0,2.9,2.8,2.7,2.6,2.5,2.4,2.3,2.2,2.1,2.0,1.9,1.8,1.7,1.6,1.5,1.4,1.3,1.2,1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5,0.1])
	b = np.array([0.15,0.18,0.19,0.22,0.25,0.29,0.34,0.4,0.5,0.5,0.6,0.65,0.7,0.8,0.9,1.0,1.1,1.3,1.6,1.8,2.2,2.6,3.2,4.0,5.1,6.6,8.9,12.6,18.3,27.0,30.1,30.2,30.3,30.4])

	#a = np.array([5.0,3.0,2.5,2.0,1.5,1.0])
	#b = np.array([0.2,0.5,0.7,1.3,3.5,13.0])
	if get=='age':
		tck = interpolate.splrep(a[::-1],b[::-1],k=1)
		out = interpolate.splev(mm,tck)
		if mm > 5.0:
			out = 0.1
		elif mm < 1.0:
			out = 12.0
	else:
		I = np.argsort(b)
		tck = interpolate.splrep(b[I],a[I],k=1)
		out = interpolate.splev(mm,tck)
		if out < 0.01:
			out = 0.01
		elif out > 4. or np.isnan(out):
			out = 4.

	return out

def downloadYY():
	os.system('wget http://csaweb.yonsei.ac.kr/~kim/YYiso_v2.tar.gz')
	os.system('tar -xf YYiso_v2.tar.gz')
	os.system('mkdir YY')
	os.system('mv V2 YY/')
	os.system('rm YYiso_v2.tar.gz')
	os.system('cp YYmix2.f YY/V2/')
	cwd = os.getcwd()
	os.chdir('YY/V2/')
	os.system('gfortran YYmix2.f')
	os.chdir(cwd)
	print 'done...'


def downloadDartmouth():
	os.system('wget http://stellar.dartmouth.edu/models/isochrones/UBVRIJHKsKp.tgz')
	os.system('tar -xf UBVRIJHKsKp.tgz')
	os.system('rm UBVRIJHKsKp.tgz')
	os.system('mv isochrones/UBVRIJHKsKp .')
	os.system('rm -r isochrones')


def get_vals(vec):
	fvec   = np.sort(vec)

	fval  = np.median(fvec)
	nn = int(np.around(len(fvec)*0.15865))

	vali,valf = fval - fvec[nn],fvec[-nn] - fval
	return fval,vali,valf

def lnlike(theta, y, yerr):
	if lum_ind == 'rho':
		rets = 'TD'
	elif lum_ind == 'aR':
		rets = 'TA'
	elif lum_ind == 'logg':
		rets = 'TG'
	elif lum_ind == 'rstar':
		rets = 'TR'
	if feh_free:
		AGE, MASS, FEH = theta
		if isochrones == 'YY':
			model = get_pars_fehfree(AGE,MASS,FEH)
		elif isochrones == 'Dartmouth':
			model = get_pars_dartmouth_fehfree(AGE,MASS,FEH)
		
	else:
		AGE, MASS = theta
		if isochrones == 'YY':
			model = get_pars_YY_isoc_fortran(AGE,MASS,ret=rets)
			#model = get_pars(AGE,MASS,ret=rets)

		elif isochrones == 'Dartmouth':
			model = get_pars_dartmouth(AGE,MASS)
		elif isochrones == 'Girardi':
			model = get_pars_girardi(AGE,MASS)


	model = np.array([model[0],model[1]])
	inv_sigma2 = 1.0/(yerr**2)
	#print y, model
	#ret = -np.log(2*np.pi) + np.log(np.sum(np.exp(-0.5*((y-model)/yerr)**2)/yerr))
	ret = -0.5*(np.sum(inv_sigma2*(y-model)**2 - np.log(inv_sigma2)))
	if np.isnan(ret):
		return -np.inf
	else:
		return ret
	#return -0.5*(np.sum(inv_sigma2*(y-model)**2 - np.log(inv_sigma2)))

def lnprior(theta):
	if feh_free:
		AGE, MASS, FEH = theta
		if isochrones == 'YY':
			if 0.05 < AGE < 20. and 0.4 < MASS < 4.5 and -1.0 < FEH < 0.6:
				return 0.0
		elif isochrones == 'Dartmouth':
			if 1. < AGE < 15. and 0.2 < MASS < 2.1 and -1.0 < FEH < 0.5:
				return 0.0
	else:
		AGE, MASS = theta
		if isochrones == 'YY':
			if 0.05 < AGE < 20 and 0.4 < MASS < 4.5:
				return 0.0
		elif isochrones == 'Dartmouth':
			if 1. < AGE < 15. and 0.2 < MASS < 2.1:
				return 0.0
		elif isochrones == 'Girardi':
			if 0.07 < AGE < 17. and 0.15 < MASS < 6.3:
				return 0.0
	return -np.inf

def lnprob(theta, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, y, yerr)

def mp(K,Ms,P,i,e):
	G = 6.67259e-8
	q1 = (2*np.pi*G/P)**(1./3.)
	q2 = np.sin(i)/np.sqrt(1-e*e)
	return K*Ms**(2./3.)/(q1*q2)

def interp_simple(x,y,x1,x2,y1,y2,f11,f12,f21,f22):

	if x1 != x2 and y1 != y2:
		fact1 = (x2 - x)/(x2 - x1)
		fact2 = (x - x1) / (x2 - x1)
		fact3 = (y2 - y)/(y2 - y1)
		fact4 = (y - y1) / (y2 - y1)

		fxy1  = fact1 * f11 + fact2 * f21
		fxy2  = fact1 * f12 + fact2 * f22

		fxy   = fact3 * fxy1 + fact4 * fxy2
	elif x1 == x2 and y1 == y2:
		fxy = f11.copy()

	elif x1 == x2:
		pendiente = (f22 - f21) / (y2 - y1)
		poscoef   = f22 - pendiente*y2
		fxy       = pendiente*y + poscoef

	elif y1 == y2:
		pendiente = (f12 - f11) / (x2 - x1)
		poscoef   = f11 - pendiente*x1
		fxy       = pendiente*x + poscoef

	return fxy

def interp_in_age(AGE,lines):

	ags, tfs, lms = [],[],[]
	for line in lines:
		#print line
		cos = line.split()
		ags.append(float(cos[1]))
		tfs.append(float(cos[2]))
		lms.append(float(cos[3]))
	ags, tfs, lms = np.array(ags), np.array(tfs), np.array(lms)
	if AGE < ags[-1]:
		aI = np.argsort(ags)
		ags,tfs,lms = ags[aI],tfs[aI],lms[aI]
		tckt = interpolate.splrep(ags,tfs,k=3) 
		tckl = interpolate.splrep(ags,lms,k=3) 

		lt = interpolate.splev(AGE, tckt)
		ll = interpolate.splev(AGE, tckl)

	else:
		lt = tfs[-1]
		ll = lms[-1]
	return lt,ll

def get_pars_YY_tracks(AGE,MASS,ret='TA'):
	"""
	Performs an interpolation in AGE, MASS and FEH for the YY evolutionary tracks.
	output:
	T = effective temperature [K]
	L = luminosity [solar]
	R = radius [solar]
	G = surface gravity (logg)
	D = density (cgs)
	A = a/Rs
	"""
	fehs = np.array([-1.288247, -0.681060, -0.432835, -0.272683, 0.046320, 0.385695, 0.603848, 0.775363])
	zeta = np.array(['x767z001','x758z004','z749z007','x74z01','x71z02','x65z04','x59z06','x53z08'])

	masses = np.hstack(( np.arange(0.4,3.01,0.1),np.arange(3.2,4.01,0.2),np.array([4.1,4.5]) ))

	dfehs = fehs - FEH
	dfehs = np.around(dfehs,5)

	dmass = masses - MASS
	dmass = np.around(dmass,5)

	I1 = np.where(dfehs<=0)[0]
	I2 = np.where(dfehs>=0)[0]
	feh1 = fehs[I1[-1]]
	feh2 = fehs[I2[0]]

	I1 = np.where(dmass<=0)[0]
	I2 = np.where(dmass>=0)[0]
	mass1 = np.around(masses[I1[-1]],1)
	mass2 = np.around(masses[I2[0]],1)
	sm1 = str(int(mass1*10))
	sm2 = str(int(mass2*10))

	if mass1<1:
		sm1 = '0'+sm1
	if mass2<1:
		sm2 = '0'+sm2

	Z1 = np.where(fehs==feh1)[0]
	Z2 = np.where(fehs==feh2)[0]
	file11 = 'tracks/YY/'+zeta[Z1][0]+'/m'+sm1+zeta[Z1][0]+'.track2'
	file12 = 'tracks/YY/'+zeta[Z1][0]+'/m'+sm2+zeta[Z1][0]+'.track2'
	file21 = 'tracks/YY/'+zeta[Z2][0]+'/m'+sm1+zeta[Z2][0]+'.track2'
	file22 = 'tracks/YY/'+zeta[Z2][0]+'/m'+sm2+zeta[Z2][0]+'.track2'

	f11 = open(file11,'r')
	f12 = open(file12,'r')
	f21 = open(file21,'r')
	f22 = open(file22,'r')
	if mass1 == mass2 and feh1 == feh2:
		lT,lL = interp_in_age(AGE, f11.readlines()[1:])
	elif mass1 != mass2 and feh1 == feh2:
		lT1,lL1 = interp_in_age(AGE, f11.readlines()[1:])
		lT2,lL2 = interp_in_age(AGE, f12.readlines()[1:])
		x = [mass1,mass2]
		y = [lT1,lT2]
		tck = interpolate.splrep(x,y,k=1)
		lT = interpolate.splev(MASS,tck)
		y = [lL1,lL2]
		tck = interpolate.splrep(x,y,k=1)
		lL = interpolate.splev(MASS,tck)
	elif mass1 == mass2 and feh1 != feh2:
		lT1,lL1 = interp_in_age(AGE, f11.readlines()[1:])
		lT2,lL2 = interp_in_age(AGE, f21.readlines()[1:])
		x = [feh1,feh2]
		y = [lT1,lT2]
		tck = interpolate.splrep(x,y,k=1)
		lT = interpolate.splev(FEH,tck)
		y = [lL1,lL2]
		tck = interpolate.splrep(x,y,k=1)
		lL = interpolate.splev(FEH,tck)
	else:
		lT11,lL11 = interp_in_age(AGE, f11.readlines()[1:])
		lT12,lL12 = interp_in_age(AGE, f12.readlines()[1:])
		lT21,lL21 = interp_in_age(AGE, f21.readlines()[1:])
		lT22,lL22 = interp_in_age(AGE, f22.readlines()[1:])

		lT = interp_simple(FEH,MASS,feh1,feh2,mass1,mass2,lT11,lT12,lT21,lT22)
		lL = interp_simple(FEH,MASS,feh1,feh2,mass1,mass2,lL11,lL12,lL21,lL22)

	"Lets work in cgs"
	L,T = 10**lL, 10**lT
	LL  = L*3.839e33
	sbc = 5.6704e-5
	G = 6.6743e-8
	MM = MASS*1.989e33
	RR = np.sqrt(LL/(4*np.pi*sbc*T**4))
	R  = RR/6.955e10
	LOGG  = np.log10(G*MM/(RR*RR))
	RHO   = MM / (4.*np.pi*RR*RR*RR/3.)

	a  = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (MM)**(1./3.)
	a2 = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (mp(K,MM,P,inc,e) + MM)**(1./3.)
	aR = a2/RR
	fout = []
	for out in ret:
		if out == 'T':
			fout.append(T)
		elif out == 'L':
			fout.append(L)
		elif out == 'R':
			fout.append(R)
		elif out == 'G':
			fout.append(LOGG)
		elif out == 'D':
			fout.append(RHO)
		elif out == 'A':
			fout.append(aR)

	return fout

def get_pars_YY_isoc_fortran(AGE,MASS,ret='TA'):
	#FEH = -0.11
	#K = 3900.
	#e=0.
	#inc = 87.*np.pi/180.
	#P = 9.1630947521*24.*3600.
	fin = open('YY/V2/YY.nml')
	lines = fin.readlines()
	fin.close()
	lines[1] = ' AFe=0.00\n'
 	lines[3] = ' FeH='+str(FEH)+'\n'
 	fout = open('YY/V2/YY.nml','w')
 	for line in lines:
 		fout.write(line)
 	fout.close()

 	fout = open('YY/V2/YY.age','w')
 	fout.write(str(AGE))
 	fout.close()

	os.chdir('YY/V2')
	os.system('rm yy00l.fm15a2o2')
 	os.system('./a.out >> tmp.tmp')
 	os.chdir('../../')

 	f = open('YY/V2/yy00l.fm15a2o2','r')
 	lines = f.readlines()[3:]
 	vms,vts,vls,vgs,vvs = [],[],[],[],[]
 	for line in lines:
 		cos = line.split()
 		vms.append(float(cos[0]))
 		vts.append(float(cos[1]))
 		vls.append(float(cos[2]))
 		vgs.append(float(cos[3]))
 		vvs.append(float(cos[4]))
 	vms,vts,vls,vgs,vvs =  	np.array(vms),np.array(vts),np.array(vls),np.array(vgs),np.array(vvs)

 	I = np.argsort(vms)
 	vms,vts,vls,vgs,vvs = vms[I],vts[I],vls[I],vgs[I],vvs[I]
	#print vms,vts
	try:
 		tckt = interpolate.splrep(vms,vts,k=3)
  		tckl = interpolate.splrep(vms,vls,k=3)
 		tckg = interpolate.splrep(vms,vgs,k=3)
 		tckv = interpolate.splrep(vms,vvs,k=3)
	except:
		return np.sqrt(-1), np.sqrt(-1)

 	ot = 10**interpolate.splev(MASS,tckt)
 	ol = 10**interpolate.splev(MASS,tckl)
 	og = interpolate.splev(MASS,tckg)
 	ov = interpolate.splev(MASS,tckv)

	G = constants.G.cgs.value
 	#grav = 10**og
 	oM = MASS * constants.M_sun.cgs.value
 	#oR = np.sqrt(G*oM/grav)
 	logoR = 0.5*(np.log10(G) + np.log10(oM) - og)
 	oR    = 10**logoR
 	RADIUS = oR / constants.R_sun.cgs.value

 	RHO = oM/((4./3.)*np.pi*oR**3)
 	RHOS = constants.M_sun.cgs.value / ((4./3.)*(constants.R_sun.cgs.value)**3)
 	RHO = RHO/RHOS
 	#print RADIUS
 	#print MASS
 	RHO =  MASS / (RADIUS**3)
 	#print og
	a2 = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (mp(K,oM,P,inc,e) + oM)**(1./3.)
	aR = a2/oR
	if ee == 0.:
		Te = e
	else:
		Te = np.random.normal(e,ee)

	TK = np.random.normal(K,eK)
	if TK<0:
		TK=0.
	Ti = np.random.normal(inc,einc)
	Tr = np.random.normal(rat,rat_err)
	#print AGE,MASS, ot, RHO, og
	if 'm' in ret:
		mpl = get_mass(MASS,P,Te,TK,Ti)
	if 'r' in ret:
		rp = get_radius(RADIUS, Tr)
	if 'a' in ret:
		aa = kep_third(MASS,mpl,P)
	if 't' in ret:
		teq = get_teq(aa*constants.au.cgs.value,RADIUS*constants.R_sun.cgs.value,ot)
	if 'd' in ret:
		den = oM / (4*np.pi*oR**3/3.)

	fout = []
	for out in ret:
		if out == 'T':
			fout.append(ot)
		elif out == 'L':
			fout.append(ol)
		elif out == 'R':
			fout.append(RADIUS)
		elif out == 'G':
			fout.append(og)
		elif out == 'D':
			fout.append(RHO)
		elif out == 'A':
			fout.append(aR)
		elif out == 'V':
			fout.append(ov)
		elif out == 'm':
			fout.append(mpl)
		elif out == 'r':
			fout.append(rp)
		elif out == 'a':
			fout.append(aa)
		elif out == 't':
			fout.append(teq)
		elif out == 'd':
			fout.append(den)

	#print fout
	#print AGE,MASS, fout
	return fout


def get_pars_fehfree(AGE,MASS,FEH, ret_all=False):
	G = 6.67259e-8

	dfehs = fehs - FEH
	dages = ages - AGE

	I1 = np.where(dfehs<=0)[0]
	I2 = np.where(dfehs>=0)[0]
	feh1,feh2 = fehs[I1[-1]], fehs[I2[0]]

	I1 = np.where(dages<=0)[0]
	I2 = np.where(dages>=0)[0]
	age1,age2 = ages[I1[-1]], ages[I2[0]]

	#print feh1, feh2
	#print age1, age2

	f = open('feh_files.txt','r')
	lines = f.readlines()
	paths = ['','','','','','','','']
	for line in lines:
		cos = line.split()
		I = np.where(fehs == float(cos[1]))[0][0]
		paths[I] = cos[0]

	I1 = np.where(fehs == feh1)[0][0]
	I2 = np.where(fehs == feh2)[0][0]
	#print paths[I1]
	#print paths[I2]
	f1 = open('YY/V2/Iso/'+paths[I1],'r')
	f2 = open('YY/V2/Iso/'+paths[I2],'r')
	lines1 = f1.readlines()
	lines2 = f2.readlines()
	f1.close()
	f2.close()

	ii = 0
	for line1 in lines1:
		cos = line1.split()
		if len(cos)>0 and 'age' in cos[0]:
			stage = line1.split('age(Gyr)=')
			tage = float(stage[1].split()[0])
			nlines = int(stage[1].split()[1])
			if tage == age1:
				ini11 = ii + 1
				fin11 = ii + nlines + 1
			if tage == age2:
				ini12 = ii + 1
				fin12 = ii + nlines + 1
		ii+=1
	#print fd

	masses11,teffs11,loggs11 = [],[],[]
	lums11, mvs11 = [],[]
	for i in np.arange(ini11,fin11,1):
		cos = lines1[i].split()
		masses11.append(float(cos[0]))
		teffs11.append(float(cos[1]))
		loggs11.append(float(cos[3]))
		if ret_all:
			lums11.append(float(cos[2]))
			mvs11.append(float(cos[4]))
	masses11,teffs11,loggs11 = np.array(masses11),np.array(teffs11),np.array(loggs11)
	lums11, mvs11 = np.array(lums11), np.array(mvs11)

	masses12,teffs12,loggs12 = [],[],[]
	lums12, mvs12 = [],[]
	for i in np.arange(ini12,fin12,1):
		cos = lines1[i].split()
		masses12.append(float(cos[0]))
		teffs12.append(float(cos[1]))
		loggs12.append(float(cos[3]))
		if ret_all:
			lums12.append(float(cos[2]))
			mvs12.append(float(cos[4]))

	masses12,teffs12,loggs12 = np.array(masses12),np.array(teffs12),np.array(loggs12)
	lums12, mvs12 = np.array(lums12), np.array(mvs12)

	ii = 0
	for line2 in lines2:
		cos = line2.split()
		if len(cos)>0 and 'age' in cos[0]:
			stage = line2.split('age(Gyr)=')
			tage = float(stage[1].split()[0])
			nlines = int(stage[1].split()[1])
			if tage == age1:
				ini21 = ii + 1
				fin21 = ii + nlines + 1
			if tage == age2:
				ini22 = ii + 1
				fin22 = ii + nlines + 1
		ii+=1
	#print fd

	masses21,teffs21,loggs21 = [],[],[]
	lums21, mvs21 = [],[]
	for i in np.arange(ini21,fin21,1):
		cos = lines2[i].split()
		masses21.append(float(cos[0]))
		teffs21.append(float(cos[1]))
		loggs21.append(float(cos[3]))
		if ret_all:
			lums21.append(float(cos[2]))
			mvs21.append(float(cos[4]))
	masses21,teffs21,loggs21 = np.array(masses21),np.array(teffs21),np.array(loggs21)
	lums21, mvs21 = np.array(lums21), np.array(mvs21)

	masses22,teffs22,loggs22 = [],[],[]
	lums22, mvs22 = [],[]
	for i in np.arange(ini22,fin22,1):
		cos = lines2[i].split()
		masses22.append(float(cos[0]))
		teffs22.append(float(cos[1]))
		loggs22.append(float(cos[3]))
		if ret_all:
			lums22.append(float(cos[2]))
			mvs22.append(float(cos[4]))
	masses22,teffs22,loggs22 = np.array(masses22),np.array(teffs22),np.array(loggs22)
	lums22, mvs22 = np.array(lums22), np.array(mvs22)

	mmin = np.max([masses11[0],masses12[0],masses21[0],masses22[0]])
	mmax = np.min([masses11[-1],masses12[-1],masses21[-1],masses22[-1]])
	umasses = np.arange(mmin,mmax,0.001)

	tckt11 = interpolate.splrep(masses11,teffs11,k=3)
	tckt12 = interpolate.splrep(masses12,teffs12,k=3)
	tckt21 = interpolate.splrep(masses21,teffs21,k=3)
	tckt22 = interpolate.splrep(masses22,teffs22,k=3)
	uteffs11 = interpolate.splev(umasses,tckt11)
	uteffs12 = interpolate.splev(umasses,tckt12)
	uteffs21 = interpolate.splev(umasses,tckt21)
	uteffs22 = interpolate.splev(umasses,tckt22)
	fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,uteffs11,uteffs12,uteffs21,uteffs22)

	tckt   = interpolate.splrep(umasses,fxy,k=3)
	TEFF  = 10**interpolate.splev(MASS,tckt)

	tckl11 = interpolate.splrep(masses11,loggs11,k=3)
	tckl12 = interpolate.splrep(masses12,loggs12,k=3)
	tckl21 = interpolate.splrep(masses21,loggs21,k=3)
	tckl22 = interpolate.splrep(masses22,loggs22,k=3)
	uloggs11 = interpolate.splev(umasses,tckl11)
	uloggs12 = interpolate.splev(umasses,tckl12)
	uloggs21 = interpolate.splev(umasses,tckl21)
	uloggs22 = interpolate.splev(umasses,tckl22)
	fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,uloggs11,uloggs12,uloggs21,uloggs22)

	tckl  = interpolate.splrep(umasses,fxy,k=3)
	LOGG  = interpolate.splev(MASS,tckl)

	g = 10**LOGG
	RADIUS   = np.sqrt(G*MASS*1.98855e33/g)
	AR    = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (MASS*1.98855e33)**(1./3.) / RADIUS
	AR2   = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (mp(K,MASS*1.98855e33,P,inc,e) + MASS*1.98855e33)**(1./3.) / RADIUS


	if ret_all:
		RAD   = RADIUS / 6.95700e10

		tckL11 = interpolate.splrep(masses11,lums11,k=3)
		tckL12 = interpolate.splrep(masses12,lums12,k=3)
		tckL21 = interpolate.splrep(masses21,lums21,k=3)
		tckL22 = interpolate.splrep(masses22,lums22,k=3)
		ulums11 = interpolate.splev(umasses,tckL11)
		ulums12 = interpolate.splev(umasses,tckL12)
		ulums21 = interpolate.splev(umasses,tckL21)
		ulums22 = interpolate.splev(umasses,tckL22)
		fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,ulums11,ulums12,ulums21,ulums22)

		tckL  = interpolate.splrep(umasses,fxy,k=3)
		LUM   = interpolate.splev(MASS,tckL)

		tckM11 = interpolate.splrep(masses11,mvs11,k=3)
		tckM12 = interpolate.splrep(masses12,mvs12,k=3)
		tckM21 = interpolate.splrep(masses21,mvs21,k=3)
		tckM22 = interpolate.splrep(masses22,mvs22,k=3)
		umvs11 = interpolate.splev(umasses,tckM11)
		umvs12 = interpolate.splev(umasses,tckM12)
		umvs21 = interpolate.splev(umasses,tckM21)
		umvs22 = interpolate.splev(umasses,tckM22)
		fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,umvs11,umvs12,umvs21,umvs22)

		tckM  = interpolate.splrep(umasses,fxy,k=3)
		Mv   = interpolate.splev(MASS,tckM)

		return TEFF, AR2, float(LOGG), RAD, 10**float(LUM), float(Mv)

	return TEFF, AR2

def get_pars(AGE,MASS, ret_all=False, ret='aR'):
	FEH = FEH_val #+ FEH_err * np.random.normal()
	G = 6.67259e-8

	dfehs = fehs - FEH
	dages = ages - AGE

	I1 = np.where(dfehs<=0)[0]
	I2 = np.where(dfehs>=0)[0]
	feh1,feh2 = fehs[I1[-1]], fehs[I2[0]]

	I1 = np.where(dages<=0)[0]
	I2 = np.where(dages>=0)[0]
	age1,age2 = ages[I1[-1]], ages[I2[0]]

	#print feh1, feh2
	#print age1, age2

	f = open('feh_files.txt','r')
	lines = f.readlines()
	paths = ['','','','','','','','']
	for line in lines:
		cos = line.split()
		I = np.where(fehs == float(cos[1]))[0][0]
		paths[I] = cos[0]

	I1 = np.where(fehs == feh1)[0][0]
	I2 = np.where(fehs == feh2)[0][0]
	#print paths[I1]
	#print paths[I2]
	f1 = open('YY/V2/Iso/'+paths[I1],'r')
	f2 = open('YY/V2/Iso/'+paths[I2],'r')
	lines1 = f1.readlines()
	lines2 = f2.readlines()
	f1.close()
	f2.close()

	ii = 0
	for line1 in lines1:
		cos = line1.split()
		if len(cos)>0 and 'age' in cos[0]:
			stage = line1.split('age(Gyr)=')
			tage = float(stage[1].split()[0])
			nlines = int(stage[1].split()[1])
			if tage == age1:
				ini11 = ii + 1
				fin11 = ii + nlines + 1
			if tage == age2:
				ini12 = ii + 1
				fin12 = ii + nlines + 1
		ii+=1

	masses11,teffs11,loggs11 = [],[],[]
	lums11, mvs11 = [],[]
	for i in np.arange(ini11,fin11,1):
		cos = lines1[i].split()
		masses11.append(float(cos[0]))
		teffs11.append(float(cos[1]))
		loggs11.append(float(cos[3]))
		lums11.append(float(cos[2]))
		mvs11.append(float(cos[4]))
	masses11,teffs11,loggs11 = np.array(masses11),np.array(teffs11),np.array(loggs11)
	lums11, mvs11 = np.array(lums11), np.array(mvs11)

	masses12,teffs12,loggs12 = [],[],[]
	lums12, mvs12 = [],[]
	for i in np.arange(ini12,fin12,1):
		cos = lines1[i].split()
		masses12.append(float(cos[0]))
		teffs12.append(float(cos[1]))
		loggs12.append(float(cos[3]))
		lums12.append(float(cos[2]))
		mvs12.append(float(cos[4]))

	masses12,teffs12,loggs12 = np.array(masses12),np.array(teffs12),np.array(loggs12)
	lums12, mvs12 = np.array(lums12), np.array(mvs12)

	ii = 0
	for line2 in lines2:
		cos = line2.split()
		if len(cos)>0 and 'age' in cos[0]:
			stage = line2.split('age(Gyr)=')
			tage = float(stage[1].split()[0])
			nlines = int(stage[1].split()[1])
			if tage == age1:
				ini21 = ii + 1
				fin21 = ii + nlines + 1
			if tage == age2:
				ini22 = ii + 1
				fin22 = ii + nlines + 1
		ii+=1
	#print fd

	masses21,teffs21,loggs21 = [],[],[]
	lums21, mvs21 = [],[]
	for i in np.arange(ini21,fin21,1):
		cos = lines2[i].split()
		masses21.append(float(cos[0]))
		teffs21.append(float(cos[1]))
		loggs21.append(float(cos[3]))
		lums21.append(float(cos[2]))
		mvs21.append(float(cos[4]))
	masses21,teffs21,loggs21 = np.array(masses21),np.array(teffs21),np.array(loggs21)
	lums21, mvs21 = np.array(lums21), np.array(mvs21)

	masses22,teffs22,loggs22 = [],[],[]
	lums22, mvs22 = [],[]
	for i in np.arange(ini22,fin22,1):
		cos = lines2[i].split()
		masses22.append(float(cos[0]))
		teffs22.append(float(cos[1]))
		loggs22.append(float(cos[3]))
		lums22.append(float(cos[2]))
		mvs22.append(float(cos[4]))
	masses22,teffs22,loggs22 = np.array(masses22),np.array(teffs22),np.array(loggs22)
	lums22, mvs22 = np.array(lums22), np.array(mvs22)

	mmin = np.max([masses11[0],masses12[0],masses21[0],masses22[0]])
	mmax = np.min([masses11[-1],masses12[-1],masses21[-1],masses22[-1]])
	umasses = np.arange(mmin,mmax,0.001)

	tckt11 = interpolate.splrep(masses11,teffs11,k=3)
	tckt12 = interpolate.splrep(masses12,teffs12,k=3)
	tckt21 = interpolate.splrep(masses21,teffs21,k=3)
	tckt22 = interpolate.splrep(masses22,teffs22,k=3)
	uteffs11 = interpolate.splev(umasses,tckt11)
	uteffs12 = interpolate.splev(umasses,tckt12)
	uteffs21 = interpolate.splev(umasses,tckt21)
	uteffs22 = interpolate.splev(umasses,tckt22)
	fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,uteffs11,uteffs12,uteffs21,uteffs22)

	tckt   = interpolate.splrep(umasses,fxy,k=3)
	TEFF  = 10**interpolate.splev(MASS,tckt)

	tckl11 = interpolate.splrep(masses11,loggs11,k=3)
	tckl12 = interpolate.splrep(masses12,loggs12,k=3)
	tckl21 = interpolate.splrep(masses21,loggs21,k=3)
	tckl22 = interpolate.splrep(masses22,loggs22,k=3)
	uloggs11 = interpolate.splev(umasses,tckl11)
	uloggs12 = interpolate.splev(umasses,tckl12)
	uloggs21 = interpolate.splev(umasses,tckl21)
	uloggs22 = interpolate.splev(umasses,tckl22)
	fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,uloggs11,uloggs12,uloggs21,uloggs22)

	tckl  = interpolate.splrep(umasses,fxy,k=3)
	LOGG  = interpolate.splev(MASS,tckl)
	#print LOGG
	g = 10**LOGG
	RADIUS   = np.sqrt(G*MASS*constants.M_sun.cgs.value/g)
	AR    = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (MASS*constants.M_sun.cgs.value)**(1./3.) / RADIUS
	AR2   = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (mp(K,MASS*constants.M_sun.cgs.value,P,inc,e) + MASS*constants.M_sun.cgs.value)**(1./3.) / RADIUS
	RAD   = RADIUS / constants.R_sun.cgs.value
	#print RADIUS

	if 'L' in ret:
		tckL11 = interpolate.splrep(masses11,lums11,k=3)
		tckL12 = interpolate.splrep(masses12,lums12,k=3)
		tckL21 = interpolate.splrep(masses21,lums21,k=3)
		tckL22 = interpolate.splrep(masses22,lums22,k=3)
		ulums11 = interpolate.splev(umasses,tckL11)
		ulums12 = interpolate.splev(umasses,tckL12)
		ulums21 = interpolate.splev(umasses,tckL21)
		ulums22 = interpolate.splev(umasses,tckL22)
		fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,ulums11,ulums12,ulums21,ulums22)

		tckL  = interpolate.splrep(umasses,fxy,k=3)
		LUM   = interpolate.splev(MASS,tckL)

	if 'V' in ret:
		tckM11 = interpolate.splrep(masses11,mvs11,k=3)
		tckM12 = interpolate.splrep(masses12,mvs12,k=3)
		tckM21 = interpolate.splrep(masses21,mvs21,k=3)
		tckM22 = interpolate.splrep(masses22,mvs22,k=3)
		umvs11 = interpolate.splev(umasses,tckM11)
		umvs12 = interpolate.splev(umasses,tckM12)
		umvs21 = interpolate.splev(umasses,tckM21)
		umvs22 = interpolate.splev(umasses,tckM22)
		fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,umvs11,umvs12,umvs21,umvs22)

		tckM  = interpolate.splrep(umasses,fxy,k=3)
		Mv   = interpolate.splev(MASS,tckM)

	if 'D' in ret:
		RHO  = MASS*constants.M_sun.cgs.value / ((4./3.)*(RAD*constants.R_sun.cgs.value)**3)
		RHOS = constants.M_sun.cgs.value / ((4./3.)*(constants.R_sun.cgs.value)**3)
		RHO = RHO/RHOS

	if 'm' in ret:
		mpl = get_mass(MASS,P,e,np.random.normal(K,eK),inc)
	if 'r' in ret:
		rp = get_radius(RAD, np.random.normal(rat,rat_err))
	if 'a' in ret:
		aa = kep_third(MASS,mpl,P)
	if 't' in ret:
		teq = get_teq(aa*constants.au.cgs.value,RAD*constants.R_sun.cgs.value,TEFF)

	fout = []
	for out in ret:
		if out == 'T':
			fout.append(TEFF)
		elif out == 'L':
			fout.append(10**float(LUM))
		elif out == 'R':
			fout.append(RAD)
		elif out == 'G':
			fout.append(float(LOGG))
		elif out == 'D':
			fout.append(RHO)
		elif out == 'A':
			fout.append(AR2)
		elif out == 'V':
			fout.append(float(Mv))
		elif out == 'm':
			fout.append(mpl)
		elif out == 'r':
			fout.append(rp)
		elif out == 'a':
			fout.append(aa)
		elif out == 't':
			fout.append(teq)
	#print fout
	return fout

def get_pars_dartmouth(AGE,MASS, ret_all=False):

	FEH = FEH_val #+ FEH_err * np.random.normal()
	G = 6.67259e-8

	dfehs = fehs - FEH
	dages = ages - AGE
	#print AGE, ages
	I1 = np.where(dfehs<=0)[0]
	I2 = np.where(dfehs>=0)[0]
	feh1,feh2 = fehs[I1[-1]], fehs[I2[0]]

	I1 = np.where(dages<=0)[0]
	I2 = np.where(dages>=0)[0]
	age1,age2 = ages[I1[-1]], ages[I2[0]]

	#print feh1, feh2
	#print age1, age2

	f = open('feh_files_dar.txt','r')
	lines = f.readlines()
	paths = ['','','','','','','','']
	for line in lines:
		cos = line.split()
		I = np.where(fehs == float(cos[1]))[0][0]
		paths[I] = cos[0]

	I1 = np.where(fehs == feh1)[0][0]
	I2 = np.where(fehs == feh2)[0][0]
	#print paths[I1]
	#print paths[I2]
	f1 = open('UBVRIJHKsKp/'+paths[I1],'r')
	f2 = open('UBVRIJHKsKp/'+paths[I2],'r')
	lines1 = f1.readlines()
	lines2 = f2.readlines()
	f1.close()
	f2.close()

	ii = 0
	for line1 in lines1:
		cos = line1.split()
		if len(cos)>0 and '#AGE=' in cos[0]:
			stage = line1.split('#AGE=')
			tage = float(stage[1].split()[0])
			nlines = int(line1.split('EEPS=')[1])
			if tage == age1:
				ini11 = ii + 2
				fin11 = ii + nlines + 1
			if tage == age2:
				ini12 = ii + 2
				fin12 = ii + nlines + 1
		ii+=1

	masses11,teffs11,loggs11 = [],[],[]
	lums11, mvs11 = [],[]
	for i in np.arange(ini11,fin11,1):
		cos = lines1[i].split()
		if not float(cos[1]) in np.array(masses11):
			masses11.append(float(cos[1]))
			teffs11.append(float(cos[2]))
			loggs11.append(float(cos[3]))
			if ret_all:
				lums11.append(float(cos[4]))
				mvs11.append(float(cos[7]))
	masses11,teffs11,loggs11 = np.array(masses11),np.array(teffs11),np.array(loggs11)
	lums11, mvs11 = np.array(lums11), np.array(mvs11)

	masses12,teffs12,loggs12 = [],[],[]
	lums12, mvs12 = [],[]
	for i in np.arange(ini12,fin12,1):
		cos = lines1[i].split()
		if not float(cos[1]) in np.array(masses12):
			masses12.append(float(cos[1]))
			teffs12.append(float(cos[2]))
			loggs12.append(float(cos[3]))
			if ret_all:
				lums12.append(float(cos[4]))
				mvs12.append(float(cos[7]))

	masses12,teffs12,loggs12 = np.array(masses12),np.array(teffs12),np.array(loggs12)
	lums12, mvs12 = np.array(lums12), np.array(mvs12)

	ii = 0
	for line2 in lines2:
		cos = line2.split()
		if len(cos)>0 and '#AGE=' in cos[0]:
			stage = line2.split('#AGE=')
			tage = float(stage[1].split()[0])
			nlines = int(line2.split('EEPS=')[1])
			if tage == age1:
				ini21 = ii + 2
				fin21 = ii + nlines + 1
			if tage == age2:
				ini22 = ii + 2
				fin22 = ii + nlines + 1
		ii+=1
	#print fd

	masses21,teffs21,loggs21 = [],[],[]
	lums21, mvs21 = [],[]
	for i in np.arange(ini21,fin21,1):
		cos = lines2[i].split()
		if not float(cos[1]) in np.array(masses21):
			masses21.append(float(cos[1]))
			teffs21.append(float(cos[2]))
			loggs21.append(float(cos[3]))
			if ret_all:
				lums21.append(float(cos[4]))
				mvs21.append(float(cos[7]))
	masses21,teffs21,loggs21 = np.array(masses21),np.array(teffs21),np.array(loggs21)
	lums21, mvs21 = np.array(lums21), np.array(mvs21)

	masses22,teffs22,loggs22 = [],[],[]
	lums22, mvs22 = [],[]
	for i in np.arange(ini22,fin22,1):
		cos = lines2[i].split()
		if not float(cos[1]) in np.array(masses22):
			masses22.append(float(cos[1]))
			teffs22.append(float(cos[2]))
			loggs22.append(float(cos[3]))
			if ret_all:
				lums22.append(float(cos[4]))
				mvs22.append(float(cos[7]))
	masses22,teffs22,loggs22 = np.array(masses22),np.array(teffs22),np.array(loggs22)
	lums22, mvs22 = np.array(lums22), np.array(mvs22)

	mmin = np.max([masses11[0],masses12[0],masses21[0],masses22[0]])
	mmax = np.min([masses11[-1],masses12[-1],masses21[-1],masses22[-1]])

	umasses = np.arange(mmin,mmax,0.001)

	Im11 = np.argsort(masses11)
	Im12 = np.argsort(masses12)
	Im21 = np.argsort(masses21)
	Im22 = np.argsort(masses22)

	tckt11 = interpolate.splrep(masses11[Im11],teffs11[Im11],k=1)
	tckt12 = interpolate.splrep(masses12[Im12],teffs12[Im12],k=1)
	tckt21 = interpolate.splrep(masses21[Im21],teffs21[Im21],k=1)
	tckt22 = interpolate.splrep(masses22[Im22],teffs22[Im22],k=1)
	uteffs11 = interpolate.splev(umasses,tckt11)
	uteffs12 = interpolate.splev(umasses,tckt12)
	uteffs21 = interpolate.splev(umasses,tckt21)
	uteffs22 = interpolate.splev(umasses,tckt22)
	fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,uteffs11,uteffs12,uteffs21,uteffs22)

	tckt   = interpolate.splrep(umasses,fxy,k=3)
	TEFF  = 10**interpolate.splev(MASS,tckt)

	tckl11 = interpolate.splrep(masses11[Im11],loggs11[Im11],k=1)
	tckl12 = interpolate.splrep(masses12[Im12],loggs12[Im12],k=1)
	tckl21 = interpolate.splrep(masses21[Im21],loggs21[Im21],k=1)
	tckl22 = interpolate.splrep(masses22[Im22],loggs22[Im22],k=1)
	uloggs11 = interpolate.splev(umasses,tckl11)
	uloggs12 = interpolate.splev(umasses,tckl12)
	uloggs21 = interpolate.splev(umasses,tckl21)
	uloggs22 = interpolate.splev(umasses,tckl22)
	fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,uloggs11,uloggs12,uloggs21,uloggs22)

	tckl  = interpolate.splrep(umasses,fxy,k=3)
	LOGG  = interpolate.splev(MASS,tckl)

	g = 10**LOGG
	RADIUS   = np.sqrt(G*MASS*1.98855e33/g)
	AR    = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (MASS*1.98855e33)**(1./3.) / RADIUS
	AR2   = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (mp(K,MASS*1.98855e33,P,inc,e) + MASS*1.98855e33)**(1./3.) / RADIUS

	if ret_all:
		RAD   = RADIUS / 6.95700e10

		tckL11 = interpolate.splrep(masses11[Im11],lums11[Im11],k=1)
		tckL12 = interpolate.splrep(masses12[Im12],lums12[Im12],k=1)
		tckL21 = interpolate.splrep(masses21[Im21],lums21[Im21],k=1)
		tckL22 = interpolate.splrep(masses22[Im22],lums22[Im22],k=1)
		ulums11 = interpolate.splev(umasses,tckL11)
		ulums12 = interpolate.splev(umasses,tckL12)
		ulums21 = interpolate.splev(umasses,tckL21)
		ulums22 = interpolate.splev(umasses,tckL22)
		fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,ulums11,ulums12,ulums21,ulums22)

		tckL  = interpolate.splrep(umasses,fxy,k=3)
		LUM   = interpolate.splev(MASS,tckL)

		tckM11 = interpolate.splrep(masses11[Im11],mvs11[Im11],k=1)
		tckM12 = interpolate.splrep(masses12[Im12],mvs12[Im12],k=1)
		tckM21 = interpolate.splrep(masses21[Im21],mvs21[Im21],k=1)
		tckM22 = interpolate.splrep(masses22[Im22],mvs22[Im22],k=1)
		umvs11 = interpolate.splev(umasses,tckM11)
		umvs12 = interpolate.splev(umasses,tckM12)
		umvs21 = interpolate.splev(umasses,tckM21)
		umvs22 = interpolate.splev(umasses,tckM22)
		fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,umvs11,umvs12,umvs21,umvs22)

		tckM  = interpolate.splrep(umasses,fxy,k=3)
		Mv   = interpolate.splev(MASS,tckM)
		return TEFF, AR2, float(LOGG), RAD, 10**float(LUM), float(Mv)

	return TEFF, AR2

def get_pars_girardi(AGE,MASS, ret_all=False):
	FEH = FEH_val #+ FEH_err * np.random.normal()
	G = 6.67259e-8

	dfehs = fehs - FEH
	dages = ages - AGE

	I1 = np.where(dfehs<=0)[0]
	I2 = np.where(dfehs>=0)[0]
	feh1,feh2 = fehs[I1[-1]], fehs[I2[0]]

	I1 = np.where(dages<=0)[0]
	I2 = np.where(dages>=0)[0]
	age1,age2 = ages[I1[-1]], ages[I2[0]]

	#print feh1, feh2
	#print age1, age2

	f = open('feh_files_gir.txt','r')
	lines = f.readlines()
	paths = ['','','','','','']
	for line in lines:
		cos = line.split()
		I = np.where(fehs == float(cos[1]))[0][0]
		paths[I] = cos[0]

	I1 = np.where(fehs == feh1)[0][0]
	I2 = np.where(fehs == feh2)[0][0]
	#print paths[I1]
	#print paths[I2]
	f1 = open('Girardi/'+paths[I1],'r')
	f2 = open('Girardi/'+paths[I2],'r')
	lines1 = f1.readlines()
	lines2 = f2.readlines()
	f1.close()
	f2.close()

	ii = 0
	for line1 in lines1:
		cos = line1.split()
		if len(cos)==9:
			tage = float(cos[7])/(1e9)
			if np.around(tage,2) == np.around(age1,2):
				ini11 = ii + 2
			if np.around(tage,2) == np.around(age2,2):
				ini12 = ii + 2
		ii+=1

	masses11,teffs11,loggs11 = [],[],[]
	lums11, mvs11, acmasses11 = [],[],[]
	i = ini11
	while True:
		cos = lines1[i].split()
		if not float(cos[1]) in np.array(masses11):
			masses11.append(float(cos[1]))
			teffs11.append(float(cos[4]))
			loggs11.append(float(cos[5]))
			if ret_all:
				acmasses11.append(float(cos[2]))
				lums11.append(float(cos[3]))
				mvs11.append(float(cos[9]))
		if i+1==len(lines1) or lines1[i+1][0] == '#':
			break
		else:
			i+=1

	masses11,teffs11,loggs11 = np.array(masses11),np.array(teffs11),np.array(loggs11)
	lums11, mvs11,acmasses11 = 10**np.array(lums11), np.array(mvs11), np.array(acmasses11)

	masses12,teffs12,loggs12 = [],[],[]
	lums12, mvs12,acmasses12 = [],[],[]
	i = ini12
	while True:
		cos = lines1[i].split()
		if not float(cos[1]) in np.array(masses12):
			masses12.append(float(cos[1]))
			teffs12.append(float(cos[4]))
			loggs12.append(float(cos[5]))
			if ret_all:
				acmasses12.append(float(cos[2]))
				lums12.append(float(cos[3]))
				mvs12.append(float(cos[9]))
		if i+1==len(lines1) or lines1[i+1][0] == '#':
			break
		else:
			i+=1

	masses12,teffs12,loggs12 = np.array(masses12),np.array(teffs12),np.array(loggs12)
	lums12, mvs12,acmasses12 = 10**np.array(lums12), np.array(mvs12), np.array(acmasses12)

	ii = 0
	for line2 in lines2:
		cos = line2.split()
		if len(cos)==9:
			tage = float(cos[7])/(1e9)
			if np.around(tage,2) == np.around(age1,2):
				ini21 = ii + 2
			if np.around(tage,2) == np.around(age2,2):
				ini22 = ii + 2
		ii+=1
	#print fd

	masses21,teffs21,loggs21 = [],[],[]
	lums21, mvs21, acmasses21 = [],[],[]
	i=ini21
	while True:
		cos = lines2[i].split()
		if not float(cos[1]) in np.array(masses21):
			masses21.append(float(cos[1]))
			teffs21.append(float(cos[4]))
			loggs21.append(float(cos[5]))
			if ret_all:
				acmasses21.append(float(cos[2]))
				lums21.append(float(cos[3]))
				mvs21.append(float(cos[9]))
		if i+1==len(lines2) or lines2[i+1][0] == '#':
			break
		else:
			i+=1
	masses21,teffs21,loggs21 = np.array(masses21),np.array(teffs21),np.array(loggs21)
	lums21, mvs21, acmasses21 = 10**np.array(lums21), np.array(mvs21), np.array(acmasses21)

	masses22,teffs22,loggs22 = [],[],[]
	lums22, mvs22, acmasses22 = [],[],[]
	i=ini21
	while True:
		cos = lines2[i].split()
		if not float(cos[1]) in np.array(masses22):
			masses22.append(float(cos[1]))
			teffs22.append(float(cos[4]))
			loggs22.append(float(cos[5]))
			if ret_all:
				acmasses22.append(float(cos[2]))
				lums22.append(float(cos[3]))
				mvs22.append(float(cos[9]))
		if i+1==len(lines2) or lines2[i+1][0] == '#':
			break
		else:
			i+=1

	masses22,teffs22,loggs22 = np.array(masses22),np.array(teffs22),np.array(loggs22)
	lums22, mvs22, acmasses22 = 10**np.array(lums22), np.array(mvs22), np.array(acmasses22)

	mmin = np.max([masses11[0],masses12[0],masses21[0],masses22[0]])
	mmax = np.min([masses11[-1],masses12[-1],masses21[-1],masses22[-1]])

	umasses = np.arange(mmin,mmax,0.001)
	Im11 = np.argsort(masses11)
	Im12 = np.argsort(masses12)
	Im21 = np.argsort(masses21)
	Im22 = np.argsort(masses22)
	tckt11 = interpolate.splrep(masses11[Im11],teffs11[Im11],k=1)
	tckt12 = interpolate.splrep(masses12[Im12],teffs12[Im12],k=1)
	tckt21 = interpolate.splrep(masses21[Im21],teffs21[Im21],k=1)
	tckt22 = interpolate.splrep(masses22[Im22],teffs22[Im22],k=1)
	uteffs11 = interpolate.splev(umasses,tckt11)
	uteffs12 = interpolate.splev(umasses,tckt12)
	uteffs21 = interpolate.splev(umasses,tckt21)
	uteffs22 = interpolate.splev(umasses,tckt22)

	fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,uteffs11,uteffs12,uteffs21,uteffs22)

	tckt   = interpolate.splrep(umasses,fxy,k=1)
	TEFF   = 10**interpolate.splev(MASS,tckt)
	#print MASS, AGE
	#plot(umasses,10**fxy)
	#show()

	tckl11 = interpolate.splrep(masses11[Im11],loggs11[Im11],k=1)
	tckl12 = interpolate.splrep(masses12[Im12],loggs12[Im12],k=1)
	tckl21 = interpolate.splrep(masses21[Im21],loggs21[Im21],k=1)
	tckl22 = interpolate.splrep(masses22[Im22],loggs22[Im22],k=1)
	uloggs11 = interpolate.splev(umasses,tckl11)
	uloggs12 = interpolate.splev(umasses,tckl12)
	uloggs21 = interpolate.splev(umasses,tckl21)
	uloggs22 = interpolate.splev(umasses,tckl22)
	fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,uloggs11,uloggs12,uloggs21,uloggs22)

	tckl  = interpolate.splrep(umasses,fxy,k=3)
	LOGG  = interpolate.splev(MASS,tckl)

	tckL11 = interpolate.splrep(masses11[Im11],acmasses11[Im11],k=1)
	tckL12 = interpolate.splrep(masses12[Im12],acmasses12[Im12],k=1)
	tckL21 = interpolate.splrep(masses21[Im21],acmasses21[Im21],k=1)
	tckL22 = interpolate.splrep(masses22[Im22],acmasses22[Im22],k=1)
	uacmasses11 = interpolate.splev(umasses,tckL11)
	uacmasses12 = interpolate.splev(umasses,tckL12)
	uacmasses21 = interpolate.splev(umasses,tckL21)
	uacmasses22 = interpolate.splev(umasses,tckL22)
	fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,ucmasses11,uacmasses12,uacmasses21,uacmasses22)

	tckL  = interpolate.splrep(umasses,fxy,k=3)
	ACMASS   = interpolate.splev(MASS,tckL)
	
	g = 10**LOGG
	RADIUS   = np.sqrt(G*ACMASS*1.98855e33/g)
	AR    = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (ACMASS*1.98855e33)**(1./3.) / RADIUS
	AR2   = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (mp(K,ACMASS*1.98855e33,P,inc,e) + ACMASS*1.98855e33)**(1./3.) / RADIUS
	#print AGE, MASS, TEFF, RADIUS,LOGG
	if ret_all:
		RAD   = RADIUS / 6.95700e10

		tckL11 = interpolate.splrep(masses11[Im11],lums11[Im11],k=1)
		tckL12 = interpolate.splrep(masses12[Im12],lums12[Im12],k=1)
		tckL21 = interpolate.splrep(masses21[Im21],lums21[Im21],k=1)
		tckL22 = interpolate.splrep(masses22[Im22],lums22[Im22],k=1)
		ulums11 = interpolate.splev(umasses,tckL11)
		ulums12 = interpolate.splev(umasses,tckL12)
		ulums21 = interpolate.splev(umasses,tckL21)
		ulums22 = interpolate.splev(umasses,tckL22)
		fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,ulums11,ulums12,ulums21,ulums22)

		tckL  = interpolate.splrep(umasses,fxy,k=3)
		LUM   = interpolate.splev(MASS,tckL)

		tckM11 = interpolate.splrep(masses11[Im11],mvs11[Im11],k=1)
		tckM12 = interpolate.splrep(masses12[Im12],mvs12[Im12],k=1)
		tckM21 = interpolate.splrep(masses21[Im21],mvs21[Im21],k=1)
		tckM22 = interpolate.splrep(masses22[Im22],mvs22[Im22],k=1)
		umvs11 = interpolate.splev(umasses,tckM11)
		umvs12 = interpolate.splev(umasses,tckM12)
		umvs21 = interpolate.splev(umasses,tckM21)
		umvs22 = interpolate.splev(umasses,tckM22)
		fxy = interp_simple(FEH,AGE,feh1,feh2,age1,age2,umvs11,umvs12,umvs21,umvs22)

		tckM  = interpolate.splrep(umasses,fxy,k=3)
		Mv   = interpolate.splev(MASS,tckM)
		return TEFF, AR2, float(LOGG), RAD, 10**float(LUM), float(Mv)

	return TEFF, AR2

def comp(infile, lumi='rho', avoid_plot=False,imass=1.,iage=4.5):

	global K,eK, e, ee, inc,einc, P, FEH, FEH_val, FEH_err, feh_free,isochrones, fehs, ages, rat, rat_err
	global lum_ind
	lum_ind = lumi
	f = open(infile,'r')
	lines = f.readlines()
	for line in lines:
		cos = line.split()
		if cos[0] == 'K':
			K = float(cos[1])*100
			eK = float(cos[2])*100
		elif cos[0] == 'e':
			e = float(cos[1])
			ee = float(cos[2])
		elif cos[0] == 'inc':
			inc = float(cos[1])*np.pi/180.
			einc = float(cos[2])*np.pi/180.
		elif cos[0] == 'P':
			P = float(cos[1]) * 24. * 3600.
		elif cos[0] == 'logg':
			LOGG_val,LOGG_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'feh':
			FEH_val, FEH_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'aR':
			AR_val, AR_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'rat':
			rat, rat_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'teff':
			TEFF_val, TEFF_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'rho':
			RHO_val, RHO_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'rstar':
			Rs_val, Rs_err = float(cos[1]), float(cos[2])

		elif cos[0] == 'isoc':
			isochrones = cos[1]
		elif cos[0] == 'feh_free':
			if cos[1] == 'True':
				feh_free = True
			elif cos[1] == 'False':
				feh_free = False
	if isochrones == 'YY' and os.access('YY/V2/Iso/yy00g.x53z08a0o2v2',os.F_OK)==False:
		print 'Downloading YY isochrones...'
		downloadYY()
	if isochrones == 'Dartmouth' and os.access('UBVRIJHKsKp/fehp00afep0.UBVRIJHKsKp',os.F_OK)==False:
		print 'Downloading Dartmouth isochrones...'
		downloadDartmouth()
	"""
	K   = 124.5
	e   = 0.0
	inc = 89.36*np.pi/180.
	P   = 6.5692203 * 24. * 3600.

	FEH_val, FEH_err = 0.2, 0.056
	AR_val, AR_err     = 14.980, 0.15

	TEFF_val, TEFF_err = 5770., 85.

	isochrones = 'YY'
	feh_free   = False
	"""
	if lum_ind == 'rho':
		y=np.array([TEFF_val,RHO_val])
		yerr = np.array([TEFF_err,RHO_err])
	elif lum_ind == 'aR':
		y=np.array([TEFF_val,AR_val])
		yerr = np.array([TEFF_err,AR_err])
	elif lum_ind == 'logg':
		y=np.array([TEFF_val,LOGG_val])
		yerr = np.array([TEFF_err,LOGG_err])
	elif lum_ind == 'rstar':
		y=np.array([TEFF_val,Rs_val])
		yerr = np.array([TEFF_err,Rs_err])	
	
	nwalkers =10
	FEH = FEH_val
	"""
	print get_pars_YY_isoc_fortran(5.8,1.2003,ret='TD')
	theta = 5.8,1.2003
	print lnlike(theta, y, yerr)
	print get_pars_YY_isoc_fortran(0.21,0.53,ret='TD')
	theta = 0.21,0.53
	print lnlike(theta, y, yerr)
	print gfd
	"""
	if feh_free:
		ndim = 3
	else:
		ndim = 2

	#print get_pars_dartmouth(4.49935553, 0.8916894,ret_all=True)
	#print get_pars(4.73,1.01,ret_all=True)
	#print fds
	"""
	nll = lambda *args: -lnlike(*args)
	result = op.minimize(nll, [1.1, 1.2], args=(y, yerr))
	oage, omass = result["x"]
	print result['x']
	"""

	if isochrones == 'YY':
		vec_ages = np.arange(0.0011,11,0.1)
		vec_mass = np.arange(0.41,4.5,0.1)
		fehs = np.array([-1.288247, -0.681060, -0.432835, -0.272683, 0.046320, 0.385695, 0.603848, 0.775363])
		ages = np.array([0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, \
		        		 0.8, 0.9, 1., 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, \
		        		 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
	elif isochrones == 'Dartmouth':
		vec_ages = np.arange(1.001,15,0.1)
		vec_mass = np.arange(0.21,2.2,0.1)
		fehs = np.array([-1.0, -0.5, 0.0, 0.2, 0.3, 0.5])
		ages = np.array([1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3.,3.25,3.5,3.75,4.,4.25,4.5,4.75,5.,5.5,6.,6.5,\
			7.,7.5,8.,8.5,9.,9.5,10.,10.5,11.,11.5,12.,12.5,13.,13.5,14.,14.5,15.])
	elif isochrones == 'Girardi':
		vec_ages = np.arange(0.07,17,0.1)
		vec_mass = np.arange(0.15,6.5,0.1)
		vec_ages = np.arange(0.07,17,0.5)
		vec_mass = np.arange(0.15,6.5,0.2)
		fehs = np.array([-1.54, -1.14, -0.54, -0.23, 0.17, 0.4])
		ages = (10**np.arange(7.8,10.26,0.05))/(1e9)
	"""
	it=0
	for vec_a in vec_ages:
		for vec_m in vec_mass:
			if feh_free:
				theta = vec_a,vec_m,FEH_val
			else:
				theta = vec_a,vec_m
			if True:
				like = lnlike(theta, y, yerr)
				if np.isinf(like) == False:
					#print vec_a,vec_m, like
					if it == 0 or like > like_max:
						like_max = like
						best_a,best_m = vec_a,vec_m
			else:
				'pass'
			it+=1
	"""
	
	best_a = iage
	best_m = imass

	#print 'Initial Guess:', best_a, best_m, like_max

	if feh_free:
		guess = np.array([best_a,best_m,FEH_val])
		sigvec = np.array([5.,0.5,0.5])
	else:
		guess = np.array([best_a,best_m])
		sigvec = np.array([5.,0.5])

	pos = []
	while len(pos) < nwalkers:
		vala = guess[0] + 2.*np.random.randn()
		valm = guess[1] + 0.1*np.random.randn()
		if vala>vec_ages[0] and vala < vec_ages[-1] and valm>vec_mass[0] and valm<vec_mass[-1]:
			pos.append(np.array([vala,valm]))

	#"""
	#pos = [guess + sigvec*np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(y, yerr))
	sampler.run_mcmc(pos, 300)
	samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

	if not avoid_plot:
		if feh_free:
			fig = corner.corner(samples, labels=["$AGE$", "$MASS$", "$[Fe/H]$"])
		else:
			fig = corner.corner(samples, labels=["$AGE$", "$MASS$"])

		fig.savefig(infile+'.'+isochrones+".png")

	dicti = {'samples':samples}
	pickle.dump( dicti, open( infile+'.'+isochrones+"samples.pkl", 'w' ) )
	#"""
	dicti = pickle.load(open( infile+'.'+isochrones+"samples.pkl", 'r' ))
	samples = dicti['samples']

	fages   = np.sort(samples[:,0])
	fmasses = np.sort(samples[:,1])

	fage,age1,age2 = get_vals(fages)
	fmass,mass1,mass2 = get_vals(fmasses)

	print 'AGE =', fage, '(',age1, age2,')'
	print 'MASS =', fmass, '(',mass1,mass2,')'
	#print gfd
	if feh_free:
		ffehs = np.sort(samples[:,2])
		ffeh,feh1,feh2 = get_vals(ffehs)
		print 'FEH =', ffeh, '(',feh1, feh2,')'

	ftefs, fars, floggs, frads, flums, fmvs,fmps,frps,aas,teqs,rhos = [],[],[],[],[],[],[],[],[],[],[]
	for i in range(len(samples)):
		if isochrones == 'YY':
			if feh_free:
				results = get_pars_YY_isoc_fortran(samples[i,0],samples[i,1],samples[i,2],ret_all=True)
			else:
				results = get_pars_YY_isoc_fortran(samples[i,0],samples[i,1],ret='TAGRLVmratd')
				#results = get_pars(samples[i,0],samples[i,1],ret='TAGRLVmrat')

		elif isochrones == 'Dartmouth':
			if feh_free:
				results = get_pars_dartmouth(samples[i,0],samples[i,1],samples[i,2],ret_all=True)
			else:
				results = get_pars_dartmouth(samples[i,0],samples[i,1],ret_all=True)
		elif isochrones == 'Girardi':
			if feh_free:
				results = get_pars_girardi(samples[i,0],samples[i,1],samples[i,2],ret_all=True)
			else:
				results = get_pars_girardi(samples[i,0],samples[i,1],ret_all=True)

		ftefs.append(results[0])
		fars.append(results[1])
		floggs.append(results[2])
		frads.append(results[3])
		flums.append(results[4])
		fmvs.append(results[5])
		fmps.append(results[6])
		frps.append(results[7])
		aas.append(results[8])
		teqs.append(results[9])
		rhos.append(results[10])

	ftefs, fars, floggs, frads, flums, fmvs, fmps, fmrs, aas, teqs, rhos = np.array(ftefs), np.array(fars), np.array(floggs),\
				np.array(frads), np.array(flums), np.array(fmvs), np.array(fmps), np.array(frps), np.array(aas), np.array(teqs), np.array(rhos) 

	fage,age1,age2 = get_vals(fages)
	fmass,mass1,mass2 = get_vals(fmasses)
	ftef,tef1,tef2 = get_vals(ftefs)
	far,ar1,ar2  = get_vals(fars)
	flogg, logg1, logg2 = get_vals(floggs)
	frad,rad1,rad2 = get_vals(frads)
	flum,lum1,lum2 = get_vals(flums)
	fmv,mv1,mv2 = get_vals(fmvs)
	fmp,mp1,mp2 = get_vals(fmps)
	frp,rp1,rp2 = get_vals(frps)
	aav,aa1,aa2 = get_vals(aas)
	tqs,tq1,tq2 = get_vals(teqs)
	tqs,tq1,tq2 = get_vals(teqs)
	rho,rho1,rho2 = get_vals(rhos)


	print '\n'
	#print 'AGE =', fage, '(',age1, age2,')'
	#print 'MASS =', fmass, '(',mass1,mass2,')'
	print 'Teff =', ftef, '(',tef1, tef2,')'
	print 'a/Rs =', far, '(',ar1,ar2,')'
	print 'log(g) =', flogg, '(',logg1, logg2,')'
	print 'Rs =', frad, '(',rad1,rad2,')'
	print 'L =', flum, '(',lum1, lum2,')'
	print 'Mv =', fmv, '(',mv1,mv2,')'
	print 'Mp =', fmp, '(',mp1,mp2,')'
	print 'RP =', frp, '(',rp1,rp2,')'
	print 'a =', aav, '(',aa1,aa2,')'
	print 'Teq =', tqs, '(',tq1,tq2,')'
	print 'Rho =', rho, '(',rho1,rho2,')'


	dout = {        'AGE':fage, 'lage':age1, 'uage':age2, \
			'MASS':fmass, 'lmass':mass1, 'umass':mass2, \
			'Teff':ftef, 'lTeff':tef1, 'uTeff':tef2, \
			'aR':far, 'laR':ar1, 'uaR':ar2, \
			'logg':flogg, 'llogg':logg1, 'ulogg':logg2, \
			'Rs':frad, 'lRs':rad1, 'uRs':rad2, \
			'Ls':flum, 'lLs':lum1, 'uLs':lum2, \
			'Mv':fmv, 'lMv':mv1, 'uMv':mv2, \
			'Mp':fmp, 'lMp':mp1, 'uMp':mp2, \
			'Rp':frp, 'lRp':rp1, 'uRp':rp2, \
			'a':aav, 'la':aa1, 'ua':aa2, \
			'tq':tqs, 'ltq':tq1, 'utq':tq2, \
			'rho':rho, 'rhol':rho1, 'rhou':rho2, \
			}
	if feh_free:
		dout['feh']=ffeh
		dout['lfeh'] = feh1
		dout['ufeh'] = feh2

	pickle.dump(dout,open(infile+'_out.pkl','w'))
	return dout
	#print get_pars(2.3,-0.13,1.29,)

def inspect(tmasses):
	global K, e, inc, P, FEH, FEH_val, FEH_err, feh_free,isochrones, fehs, ages
	f = open('input.dat','r')
	lines = f.readlines()
	for line in lines:
		cos = line.split()
		if cos[0] == 'K':
			K = float(cos[1])
		elif cos[0] == 'e':
			e = float(cos[1])
		elif cos[0] == 'inc':
			inc = float(cos[1])*np.pi/180.
		elif cos[0] == 'P':
			P = float(cos[1]) * 24. * 3600.
		elif  cos[0] == 'feh':
			FEH_val, FEH_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'aR':
			AR_val, AR_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'teff':
			TEFF_val, TEFF_err = float(cos[1]), float(cos[2])
		elif cos[0] == 'isoc':
			isochrones = cos[1]
		elif cos[0] == 'feh_free':
			if cos[1] == 'True':
				feh_free = True
			elif cos[1] == 'False':
				feh_free = False
	if isochrones == 'YY':
		vec_ages = np.arange(0.0011,11,0.1)
		vec_mass = np.arange(0.41,4.5,0.1)
		fehs = np.array([-1.288247, -0.681060, -0.432835, -0.272683, 0.046320, 0.385695, 0.603848, 0.775363])
		ages = np.array([0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, \
		        		 0.8, 0.9, 1., 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, \
		        		 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0])
	elif isochrones == 'Dartmouth':
		vec_ages = np.arange(1.001,15,0.1)
		vec_mass = np.arange(0.21,2.2,0.1)
		fehs = np.array([-1.0, -0.5, 0.0, 0.2, 0.3, 0.5])
		ages = np.array([1.,1.25,1.5,1.75,2.,2.25,2.5,2.75,3.,3.25,3.5,3.75,4.,4.25,4.5,4.75,5.,5.5,6.,6.5,\
			7.,7.5,8.,8.5,9.,9.5,10.,10.5,11.,11.5,12.,12.5,13.,13.5,14.,14.5,15.])
	elif isochrones == 'Girardi':
		vec_ages = np.arange(0.07,17,0.1)
		vec_mass = np.arange(0.15,6.5,0.1)
		vec_ages = np.arange(0.07,17,0.5)
		vec_mass = np.arange(0.15,6.5,0.2)
		fehs = np.array([-1.54, -1.14, -0.54, -0.23, 0.17, 0.4])
		ages = (10**np.arange(7.8,10.26,0.05))/(1e9)
	"""
	for ag in tages:
		veca, vect = [],[]
		for ma in tmasses:
			t1,a1 = get_pars_girardi(ag,ma)
			veca.append(a1)
			vect.append(t1)
		plot(vect,veca)
	"""
	for ma in tmasses:
		vec = np.linspace(1,10**(end(ma)/10.0),1000)
		tages = 10*np.log10(vec)[300:]
		I = np.where(tages>ages[0])[0]
		tages = tages[I]
		veca, vect = [],[]
		for ag in tages:
			if isochrones == 'YY':
				t1,a1 = get_pars(ag,ma,ret='logg')
			elif isochrones == 'Dartmouth':
				t1,a1 = get_pars_dartmouth(ag,ma)
			elif isochrones == 'Girardi':
				t1,a1 = get_pars_girardi(ag,ma)
			veca.append(a1)
			vect.append(t1)
		veca, vect = np.array(veca), np.array(vect)
		I = np.where( np.isfinite(veca)==True )[0]
		vect,veca = vect[I],veca[I]
		I = np.where( (veca>0) & (veca<100) )[0]
		vect,veca = vect[I],veca[I]
		I = np.where( (vect>1000) & (vect<30000) )[0]
		vect,veca = vect[I],veca[I]
		print ma
		print veca
		print vect
		#if len(veca)>0:
		#	plot(vect,veca)
	ylim([0,40])
	xlim([1000,10000])
	errorbar(TEFF_val,3.52,xerr=TEFF_err,yerr=0.216,fmt='ko')
	show()

def test():
	global K, e, inc, P, FEH, FEH_val, FEH_err, feh_free,isochrones, fehs, ages
	f = open('input.dat','r')
	lines = f.readlines()
	for line in lines:
		cos = line.split()
		if cos[0] == 'K':
			K = float(cos[1])*100
		elif cos[0] == 'e':
			e = float(cos[1])
		elif cos[0] == 'inc':
			inc = float(cos[1])*np.pi/180.
		elif cos[0] == 'P':
			P = float(cos[1]) * 24. * 3600.
		elif cos[0] == 'logg':
			LOGG_val,LOGG_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'feh':
			FEH_val, FEH_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'aR':
			AR_val, AR_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'teff':
			TEFF_val, TEFF_err = float(cos[1]), float(cos[2])
		elif cos[0] == 'isoc':
			isochrones = cos[1]
		elif cos[0] == 'feh_free':
			if cos[1] == 'True':
				feh_free = True
			elif cos[1] == 'False':
				feh_free = False
	#"""


	it = 0
	agss,msss,loggs,rads = [],[],[],[]
	print TEFF_val, TEFF_err
	while it<100:
		FEH = FEH_val  + (np.random.random()*2 - 1)*FEH_err
		stt = TEFF_val + (np.random.random()*2 - 1)*TEFF_err
		sta = LOGG_val + (np.random.random()*2 - 1)*LOGG_err
		#sta = AR_val   + (np.random.random()*2 - 1)*AR_err

		print stt,sta,FEH
		min_diff = 9999999999
		for mm in np.arange(1.0,1.4,0.01):
			vec1,vec2 = [],[]
			for tt in np.linspace(0.05,end(mm),200):
				t,a = get_pars_YY_isoc_fortran(tt,mm,ret='TA')
				#t,a = get_pars(tt,mm,ret='TA')

				#vec1.append(t)
				#vec2.append(a)
				diff = ((t-stt)/TEFF_err)**2 + ((a-sta)/LOGG_err)**2
				
				#llk = lnlike([tt,mm], y, yerr)
				if diff < min_diff:
					best_vec = [tt,mm]
					min_diff = diff
				#print mm,tt,t,a,diff,best_vec
			#plot(vec1,vec2)
		#errorbar(stt,sta,xerr=100,yerr=0.2,fmt='ko')
		#show()
		agss.append(best_vec[0])
		msss.append(best_vec[1])
		out = get_pars_YY_tracks(best_vec[0],best_vec[1],ret='TGAR')
		loggs.append(out[1])
		rads.append(out[3])
		print it, best_vec, out

		it+=1
	agss,msss,loggs,rads = np.array(agss),np.array(msss),np.array(loggs),np.array(rads)
	print np.median(agss),np.sqrt(np.var(agss))
	print np.median(msss),np.sqrt(np.var(msss))
	print np.median(loggs),np.sqrt(np.var(loggs))
	print np.median(rads),np.sqrt(np.var(rads))

def make_plot(infile,ags,lumi='aR',mass0 =1.,ylims=[0.6,2.0],xlims=[4000,7800],dens=1000):

	G = 6.67259e-8

	# Set seaborn contexts:
	sns.set_context("talk")
	sns.set_style("ticks")

	# Fonts:
	# Arial font (pretty, not latex-like)
	#rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
	# Latex fonts, quick:
	#matplotlib.rcParams['mathtext.fontset'] = 'stix'
	#matplotlib.rcParams['font.family'] = 'STIXGeneral'
	# Latex fonts, slow (but accurate):
	
	rc('font', **{'family': 'Helvetica'})
	#rc('text', usetex=True)
	matplotlib.rcParams.update({'font.size':20})
	plt.rc('legend', **{'fontsize':7})

	# Ticks to the outside:
	rcParams['axes.linewidth'] = 3.0
	rcParams['xtick.direction'] = 'out'
	rcParams['ytick.direction'] = 'out'


	fig = figure()	
	ax = fig.add_subplot(111)
	#ax.set_yscale('log')

	global K,eK, e,ee, inc,einc, P, FEH, FEH_val, FEH_err, feh_free,isochrones, fehs, ages, rat, rat_err
	f = open(infile,'r')
	lines = f.readlines()
	for line in lines:
		cos = line.split()
		if cos[0] == 'K':
			K = float(cos[1])*100
			eK = float(cos[2])*100
		elif cos[0] == 'e':
			e = float(cos[1])
			ee = float(cos[2])
		elif cos[0] == 'inc':
			inc = float(cos[1])*np.pi/180.
			einc = float(cos[2])*np.pi/180.
		elif cos[0] == 'P':
			P = float(cos[1]) * 24. * 3600.
		elif cos[0] == 'logg':
			LOGG_val,LOGG_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'feh':
			FEH_val, FEH_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'aR':
			AR_val, AR_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'rat':
			rat, rat_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'teff':
			TEFF_val, TEFF_err = float(cos[1]), float(cos[2])
		elif  cos[0] == 'rho':
			RHO_val, RHO_err = float(cos[1]), float(cos[2])
		elif cos[0] == 'isoc':
			isochrones = cos[1]
		elif  cos[0] == 'rstar':
			Rs_val, Rs_err = float(cos[1]), float(cos[2])
		elif cos[0] == 'feh_free':
			if cos[1] == 'True':
				feh_free = True
			elif cos[1] == 'False':
				feh_free = False
	FEH = FEH_val
	rhosun = constants.M_sun.cgs.value / ((4.*np.pi*constants.R_sun.cgs.value**3)/3.)
	#"""
	for age in ags:
		ts,gs = [],[]
		mmax = end(age,get='mass')
		allmasses = np.linspace(0.4,mmax,dens)
		for mass in allmasses:
			t,g = get_pars_YY_isoc_fortran(age,mass,ret='TR')
			#t,g = get_pars(age,mass,ret='TG')
			print age, mass,t,g
			ts.append(t)
			gs.append(g)


		ts,gs = np.array(ts),np.array(gs)

		#gg = 10**gs
		#RADIUS   = np.sqrt(G*allmasses*1.98855e33/gg)
		#AR2   = (G/(4*np.pi**2))**(1./3.) * P**(2./3.) * (mp(K,allmasses*1.98855e33,P,inc,e) + allmasses*1.98855e33)**(1./3.) / RADIUS


		if lumi == 'rho':
			plot(ts,gs*rhosun,'k')
		elif lumi == 'logg':
			plot(ts,gs,'k')
		elif lumi == 'aR':
			plot(ts,AR2,'k')
		else:
			plot(ts,gs,'k')

	#for mass in [0.8,1.0,1.2,1.4,1.6]:
	#	ts,gs = [],[]
	#	mmax = end(mass,get='age')
	#	print mass, mmax
	#	for age in np.linspace(0.1,mmax,100):
	#		t,g = get_pars_YY_isoc_fortran(age,mass,ret='TD')
	#		#t,g = get_pars(age,mass,ret='TG')
	#		ts.append(t)
	#		gs.append(g)
	#	ts,gs = np.array(ts),np.array(gs)
	#	plot(ts,gs,'r')
	#t,g = get_pars_YY_isoc_fortran(3.53,1.4003,ret='TD')
	#print t,g

	if lumi == 'logg':
		errorbar(TEFF_val,LOGG_val,xerr=TEFF_err,yerr=LOGG_err,fmt='b.')
		ylim([3e-3,5])
	elif lumi == 'aR':
		errorbar(TEFF_val,AR_val,xerr=TEFF_err,yerr=AR_err,fmt='b.')
	elif lumi == 'rho':
		errorbar(TEFF_val,RHO_val,xerr=TEFF_err,yerr=RHO_err,fmt='b.')
	else:
		#errorbar(TEFF_val,Rs_val,xerr=TEFF_err,yerr=Rs_err,fmt='b.')
		ax.add_artist(Ellipse(xy=[TEFF_val,Rs_val], width=3*TEFF_err, height=3*Rs_err,alpha=0.2))
		ax.add_artist(Ellipse(xy=[TEFF_val,Rs_val], width=2*TEFF_err, height=2*Rs_err,alpha=0.2))
		ax.add_artist(Ellipse(xy=[TEFF_val,Rs_val], width=1*TEFF_err, height=1*Rs_err,alpha=0.2))
	#plot(t,g,'ro')
	xlim(xlims)
	ylim(ylims)

	gca().invert_xaxis()
	#gca().invert_yaxis()
	xlabel(r'T$_{eff}$ [K]')
	if lumi == 'logg':
		ylabel(r'$log(g)$ ')
	elif lumi == 'rho':
		ylabel(r'$\rho_{\star}$ [$\rho_{\odot}$]')
	elif lumi == 'aR':
		ylabel(r'$a/R_{\star}$ ')
	elif lumi == 'rstar':
		ylabel(r'$R_{\star}$ [$R_{\odot}$]')
	#
	#show()
	savefig(infile+'_iso.pdf',format='pdf', bbox_inches='tight')
	#show()
	"""
	amax = end(mass0,get='age')
	print amax
	allages = np.linspace(0.1,amax,100)
	rough = np.arange(1,amax,1)
	#allages = allages.max() - allages[::-1]
	ts,gs = [],[]
	for age in allages:
		print age
		t,g = get_pars_YY_isoc_fortran(age,mass0,ret='TR')
		ts.append(t)
		gs.append(g)
	ts,gs = np.array(ts),np.array(gs)
	tsr,gsr = [],[]
	for age in rough:
		print age
		t,g = get_pars_YY_isoc_fortran(age,mass0,ret='TR')
		tsr.append(t)
		gsr.append(g)
	tsr,gsr = np.array(tsr),np.array(gsr)

	plot(ts,gs)
	plot(tsr,gsr,'ro')
	errorbar(TEFF_val,Rs_val,xerr=TEFF_err,yerr=Rs_err,fmt='bo')
	gca().invert_xaxis()
	#gca().invert_yaxis()
	show()
	"""