"General-Purpose Differentially-Private Confidence Intervals"


#############################################################

Scripts:

	- "DPCIs-expfam.py": main Python functions and driver to run confidence interval experiments
	- "DPCIs-expfam-mv.py": multivariate version of Gaussian of known variance (analysis on dim 0)
	- "DPCIs-OLS.py": main Python functions and driver to run confidence interval experiments for OLS
	- "functions.py": helper functions, including sensitivity and privatization mechanisms
	- "plot.py": visualization functions to plot the CI results of a specific distribution
			(designed to compare results for N = [50, 100, 500, 1000, 5000, 10000])
	- "barplot.py": plot function to compare private CI width across methods
		
Requirements:
	
	NumPy, SciPy, argparse, matplotlib.pyplot, pandas, seaborn

#############################################################

To reproduce the experiments:

	1) choose experiment parameters:
	
		--N	[int] data size
		
		--theta	[float] first parameter (manually set to parameter vector if using multivariate version 'CImv.py')
		
		--theta2 [float] second parameter, if any (manually set to parameter vector if using multivariate version 'CImv.py')
		
		--d	[str] distribution:
				'poisson' -> Poisson 
				'gaussian' -> Gaussian of known variance
				'gaussian2' -> Gaussian of unknown variance
				'gamma' -> Gamma of known shape
				'gaussianMV' -> multivariate Gaussian of known variance (use 'CImv.py')
		
		--mode	[str] 
			'empirical' for bootstrap quantile CIs
			'analytic' for standard normal CIs
		
		--e	[float] differential privacy parameter epsilon (for example 0.1 or 0.5)
		
		--clip	[bool]
			'True' -> clamp data to the sensitivity bounds found on a n=1000 dataset
			'False' -> conservative sensitivity, no clamping to the sensitivity bounds
		
		--rng	[float]
			data range radius for gaussian experiment comparison
			if rng=0., range is computed out of a surrogate dataset of size 1000
		
		--rho	[float]
			for two-parameter privatization (like Gaussian with unknown variance)
			share of privacy budget to allocate to privatization of the first parameter
			(1-rho will be allocated to privatize the other parameter)
	
	
	2) run driver with choice of experiment parameters
	
	3) results will be saved in .npy

#############################################################

To visualize the results:

	1) Run "plot.py" with the following parameters
			path -> folder containing the .npy files from the experiment
			d -> string representing the distribution of interest
			eps -> string representing the value of privacy parameter epsilon of interest
		(multiple plot types available, uncomment plot of interest in main call)
	
	2) the code will output a) a coverage analysis plot for different CI methods, and b) a
		visualization of the CIs for different CI methods

##############################################################
