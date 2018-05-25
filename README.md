# isoAR
Stellar physical parameters from Teff, [Fe/H] and a luminosity indicator using YY or Dartmouth isochrones. The luminosity indicator can be: log(g), a/R_star, rho_star, R_star.

For downloading the YY isochrones and compiling the fortran interpolator:
```
import isoAR
isoAR.downloadYY()
```

For computing the parameters:
```
import isoAR
isoAR.comp('input.dat',lumi='rstar',imass=1.,iage=0.5)
```

where 'lumi' corresponds to the luminosity indicator (rstar, a/R, rho, logg), and 'imass' and 'iage' are initial guesses for the mass and age of the star in solar masses and Gyr.