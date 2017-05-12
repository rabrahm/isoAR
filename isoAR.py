from pylab import *
import scipy
from scipy import interpolate
import emcee
import scipy.optimize as op
import corner
import pickle
import os

def downloadYY():
	os.system('wget http://csaweb.yonsei.ac.kr/~kim/YYiso_v2.tar.gz')
	os.system('tar -xf YYiso_v2.tar.gz')
	os.system('mkdir YY')
	os.system('mv V2 YY/')
	os.system('rm YYiso_v2.tar.gz')

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
	if feh_free:
		AGE, MASS, FEH = theta
		if isochrones == 'YY':
			model = get_pars_fehfree(AGE,MASS,FEH)
		elif isochrones == 'Dartmouth':
			model = get_pars_dartmouth_fehfree(AGE,MASS,FEH)
		
	else:
		AGE, MASS = theta
		if isochrones == 'YY':
			model = get_pars(AGE,MASS)
		elif isochrones == 'Dartmouth':
			model = get_pars_dartmouth(AGE,MASS)

	model = np.array([model[0],model[1]])
	inv_sigma2 = 1.0/(yerr**2)
	return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(theta):
	if feh_free:
		AGE, MASS, FEH = theta
		if isochrones == 'YY':
			if 0.01 < AGE < 20.0 and 0.4 < MASS < 2.5 and -1.0 < FEH < 0.5:
				return 0.0
		elif isochrones == 'Dartmouth':
			if 1. < AGE < 15. and 0.2 < MASS < 2.1 and -1.0 < FEH < 0.5:
				return 0.0
	else:
		AGE, MASS = theta
		if isochrones == 'YY':
			if 0.01 < AGE < 20.0 and 0.4 < MASS < 2.5:
				return 0.0
		elif isochrones == 'Dartmouth':
			if 1. < AGE < 15. and 0.2 < MASS < 2.1:
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

def get_pars(AGE,MASS, ret_all=False):
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

def get_pars_dartmouth(AGE,MASS, ret_all=False):
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

def comp():

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
	y=np.array([TEFF_val,AR_val])
	yerr = np.array([TEFF_err,AR_err])
	nwalkers =100

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

	it=0
	for vec_a in vec_ages:
		for vec_m in vec_mass:
			if feh_free:
				theta = vec_a,vec_m,FEH_val
			else:
				theta = vec_a,vec_m
			try:
				like = lnlike(theta, y, yerr)
				if np.isinf(like) == False:
					#print vec_a,vec_m, like
					if it == 0 or like > like_max:
						like_max = like
						best_a,best_m = vec_a,vec_m
			except:
				'pass'
			it+=1
	print 'Initial Guess:', best_a, best_m, like_max

	if feh_free:
		guess = np.array([best_a,best_m,FEH_val])
	else:
		guess = np.array([best_a,best_m])

	pos = [guess + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(y, yerr))
	sampler.run_mcmc(pos, 500)
	samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

	if feh_free:
		fig = corner.corner(samples, labels=["$AGE$", "$MASS$", "$[Fe/H]$"])
	else:
		fig = corner.corner(samples, labels=["$AGE$", "$MASS$"])

	fig.savefig("triangle.png")
	dicti = {'samples':samples}
	pickle.dump( dicti, open( "samples.pkl", 'w' ) )

	fages   = np.sort(samples[:,0])
	fmasses = np.sort(samples[:,1])

	fage,age1,age2 = get_vals(fages)
	fmass,mass1,mass2 = get_vals(fmasses)

	print 'AGE =', fage, '(',age1, age2,')'
	print 'MASS =', fmass, '(',mass1,mass2,')'
	if feh_free:
		ffehs = np.sort(samples[:,2])
		ffeh,feh1,feh2 = get_vals(ffehs)
		print 'FEH =', ffeh, '(',feh1, feh2,')'

	ftefs, fars, floggs, frads, flums, fmvs = [],[],[],[],[],[]
	for i in range(len(samples)):
		if isochrones == 'YY':
			if feh_free:
				results = get_pars_fehfree(samples[i,0],samples[i,1],samples[i,2],ret_all=True)
			else:
				results = get_pars(samples[i,0],samples[i,1],ret_all=True)
		elif isochrones == 'Dartmouth':
			if feh_free:
				results = get_pars_dartmouth(samples[i,0],samples[i,1],samples[i,2],ret_all=True)
			else:
				results = get_pars_dartmouth(samples[i,0],samples[i,1],ret_all=True)

		ftefs.append(results[0])
		fars.append(results[1])
		floggs.append(results[2])
		frads.append(results[3])
		flums.append(results[4])
		fmvs.append(results[5])
	ftefs, fars, floggs, frads, flums, fmvs = np.array(ftefs), np.array(fars), np.array(floggs), np.array(frads), np.array(flums), np.array(fmvs) 

	fage,age1,age2 = get_vals(fages)
	fmass,mass1,mass2 = get_vals(fmasses)
	ftef,tef1,tef2 = get_vals(ftefs)
	far,ar1,ar2  = get_vals(fars)
	flogg, logg1, logg2 = get_vals(floggs)
	frad,rad1,rad2 = get_vals(frads)
	flum,lum1,lum2 = get_vals(flums)
	fmv,mv1,mv2 = get_vals(fmvs)
	print '\n'
	#print 'AGE =', fage, '(',age1, age2,')'
	#print 'MASS =', fmass, '(',mass1,mass2,')'
	print 'Teff =', ftef, '(',tef1, tef2,')'
	print 'a/Rs =', far, '(',ar1,ar2,')'
	print 'log(g) =', flogg, '(',logg1, logg2,')'
	print 'Rs =', frad, '(',rad1,rad2,')'
	print 'L =', flum, '(',lum1, lum2,')'
	print 'Mv =', fmv, '(',mv1,mv2,')'

	dout = {'Teff':ftef, 'lTeff':tef1, 'uTeff':tef2, \
			'aR':far, 'laR':ar1, 'uaR':ar2, \
			'logg':flogg, 'llogg':logg1, 'ulogg':logg2, \
			'Rs':frad, 'lRs':rad1, 'uRs':rad2, \
			'Ls':flum, 'lLs':lum1, 'uLs':lum2, \
			'Mv':fmv, 'lMv':mv1, 'uMv':mv2, \
			}
	if feh_free:
		dout['feh']=ffeh
		dout['lfeh'] = feh1
		dout['ufeh'] = feh2


	return dout
	#print get_pars(2.3,-0.13,1.29,)