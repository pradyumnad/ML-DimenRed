#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x, mu, sigma):
	"""
	Returns the density of the Gaussian distribution at the point x, 
	given that the Gaussian has a mean mu and density sigma.
	"""
	normalization = 1.0 / (np.sqrt(2 * np.pi * sigma**2))
	exponent = (-1) * (x - mu)**2 / (2 * sigma**2)
	return normalization * np.e**exponent


def create_dataset(sigmas, means, ns):
	datasets = [np.random.normal(s, m, n) for s, m, n in zip(sigmas, means, ns)]
	return np.hstack(datasets)

def weighted_mean(values, weights):
	"""https://en.wikipedia.org/wiki/Weighted_arithmetic_mean"""

	total = 0
	for v, w in zip(values, weights):
		total += v*w
	return total / sum(weights)

def weighted_variance(values, weights, mean):
	"""https://en.wikipedia.org/wiki/Weighted_arithmetic_mean"""

	biased_variance = 0
	for v, w in zip(values, weights):
		biased_variance += w * (v - mean)**2
	biased_variance /= sum(weights)

	squared_weights = [w**2 for w in weights]

	corrected_variance = biased_variance / (1.0 - sum(squared_weights)/(sum(weights))**2)
	return corrected_variance

def check_if_done(means, sigmas, new_means, new_sigmas):
	""" When the algorithm has converged, changes in the mean and standard
	deviation will be very small."""

	for mean, new_mean in zip(means, new_means):
		if abs(mean - new_mean) > 0.001:
			return False
	for sigma, new_sigma in zip(sigmas, new_sigmas):
		if abs(sigma - new_sigma) > 0.001:
			return False
	return True

def gaussian_mixture(dataset, total_gaussians):
	""" This is the function that does all the heavy lifiting, and uses the
	functions defined above. NOTE: only does 1D Gaussian mixtures. """

    # get random initial guesses. We'll do this so that
    # the initial guesses are somewhere in the range of
    # the dataset.
	means = np.random.uniform(min(dataset), max(dataset), total_gaussians)
	sigmas = [1]*total_gaussians


    # plots used in debugging, not useful anymore
#     fig, ax = plt.subplots(figsize=[10, 2])
#     ax.scatter(dataset, [0]*len(dataset), linewidths=0.1, alpha=0.4, s=20)
#     ax.errorbar(means, [1]*len(means), xerr=sigmas, c="k", fmt=".", markersize=40)

	num_iterations = 1
	while True:
		credits = []
		for idx_2, point in enumerate(dataset):
			this_credit = []
			for mean, sigma in zip(means, sigmas):
				this_credit.append(gaussian(point, mean, sigma))
			this_credit = [c / sum(this_credit) for c in this_credit]

			credits.append(this_credit)

		# transpose the list, so that it is a list of the weights for each data point
		# in terms of a given model
		weights = np.array(credits).T
        
        # calculate the new means and sigmas    
		new_means = [weighted_mean(dataset, weight) for weight in weights]
		new_sigmas = [np.sqrt(weighted_variance(dataset, weight, mean)) for weight, mean in zip(weights, new_means)]

        # plots used in debugging, not useful anymore
#         fig, ax = plt.subplots(figsize=[10, 2])
#         ax.scatter(dataset, [0]*len(dataset), c=[str(w) for w in weights[0]], cmap="seismic", linewidths=0.1, alpha=0.4, s=20)
#         ax.errorbar(new_means, [1]*len(new_means), xerr=new_sigmas, c="k", fmt=".", markersize=40)
        
		if check_if_done(means, sigmas, new_means, new_sigmas):
			return means, sigmas, num_iterations
		else:
			means = new_means
			sigmas = new_sigmas
			num_iterations += 1

def error(means, sigmas, actual_means, actual_sigmas):
    """
    define error as the sum of the square differences for both mean and sigma.
    """
    error = 0
    for m, real_m in zip(means, actual_means):
        error += (m - real_m)**2
    for s, real_s in zip(sigmas, actual_sigmas):
        error += (s - real_s)**2
    return error / (len(means) + len(sigmas))


def main():
	""" This function just makes the plots. If you just want to try the
	Gaussian mixture simply, use the code below, addind the values for the 
	parameters that you want into the first two functions. The code in the 
	main function below is fancy applications of this basic workflow, used
	to test the performance of the code under various conditions.

	# make a datset:
	# means is a list of the means of the Gaussian distributions, sigmas is 
	# a list of the standard deviations, and numbers is a list of the number
	# of data points in each Gaussian. They will all be combined into one
	# list, which is returned.
	dataset = create_dataset(means, sigmas, numbers)

	# then we can use the Gaussian mixture code. Pass in the dataset we just
	# created and the number of Gaussians present in it. The function returns
	# a list of means, a list of standard deviations, and the number of
	# iterations taken to converge
	calculated_means, calculate_sigmas, num_iters = gaussian_mixture(dataset, num_gaussians)

	# We can then calculate the error
	err = error(calculated_means, calculated_means, means, sigmas)

	"""
    # explore how size of dataset impacts error and iterations
	real_means = [0, 2, 4]
	real_sigmas = [1, 1, 1]
	errors = []
	iters = []
	ns = range(10, 100, 10)
	for n in ns:
		this_error = []
		this_iter = []
		for i in range(20):
			dataset = create_dataset(real_means, real_sigmas, [n, n, n])
			means, sigmas, iterations = gaussian_mixture(dataset, 3)
			means.sort()
        
			err = error(means, sigmas, real_means, real_sigmas)
			this_error.append(err)
			this_iter.append(iterations)
            
		errors.append(np.median(this_error))
		iters.append(np.median(this_iter))

	fig, ax = plt.subplots(figsize=[7, 4])    
	ax.scatter(ns, errors, linewidth=0.1, s=70)
	ax.set_xlabel("Size of dataset")
	ax.set_ylabel("Error")
	ax.set_ylim(bottom=0)
	ax.set_xlim(right=105)
	fig.savefig("error_vs_size.png", format="png", dpi=300)

	fig, ax = plt.subplots(figsize=[7, 4])
	ax.scatter(ns, iters, linewidth=0.1, s=70)
	ax.set_xlim(right=105)
	ax.set_xlabel("Size of dataset")
	ax.set_ylabel("Iterations to converge")
	ax.set_ylim(bottom=0)
	ax.set_xlim(right=105)
	fig.savefig("iterations_vs_size.png", format="png", dpi=300)
    
	# explore how separation matters
    
	errors = []
	iters = []
	ns = np.arange(0.25, 5, 0.25)
	for n in ns:
		real_means = [0, 1, 2]
		real_sigmas = [1, 1, 1]
        
		real_means = [mean * n for mean in real_means]
        
		this_error = []
		this_iter = []
		for i in range(20):
			dataset = create_dataset(real_means, real_sigmas, [50, 50, 50])
			means, sigmas, iterations = gaussian_mixture(dataset, 3)
			means.sort()
        
			err = error(means, sigmas, real_means, real_sigmas)
			this_error.append(err)
			this_iter.append(iterations)
            
		errors.append(np.median(this_error))
		iters.append(np.median(this_iter))

	fig, ax = plt.subplots(figsize=[7, 4])    
	ax.scatter(ns, errors, linewidth=0.1, s=70)
	ax.set_xlabel("Separation (distance / sigma)")
	ax.set_ylabel("Error")
	ax.set_ylim(bottom=0)
	ax.set_xlim(right=5.25)
	fig.savefig("error_vs_separation.png", format="png", dpi=300)

	fig, ax = plt.subplots(figsize=[7, 4])
	ax.scatter(ns, iters, linewidth=0.1, s=70)
	ax.set_xlabel("Separation (distance / sigma)")
	ax.set_ylabel("Iterations to converge")
	ax.set_ylim(bottom=0)
	ax.set_xlim(right=5.25)
	fig.savefig("iterations_vs_separation.png", format="png", dpi=300)
    
	# explore how number of gaussians affects performance
	ns = [int(n) for n in np.arange(1, 10.1, 1)]
	print ns
	errors = []
	iters = []
	for n in ns:
		real_means = []
		real_sigmas = []
		for i in range(n):
			real_means.append(i*2)
			real_sigmas.append(1)
        
		this_error = []
		this_iter = []
		for i in range(20):
			dataset = create_dataset(real_means, real_sigmas, [50, 50, 50])
			means, sigmas, iterations = gaussian_mixture(dataset, n)
			means.sort()

			err = error(means, sigmas, real_means, real_sigmas)
			this_error.append(err)
			this_iter.append(iterations)

		errors.append(np.median(this_error))
		iters.append(np.median(this_iter))

	fig, ax = plt.subplots(figsize=[7, 4])    
	ax.scatter(ns, errors, linewidth=0.1, s=70)
	ax.set_xlabel("Number of Gaussians")
	ax.set_ylabel("Error")
	ax.set_ylim(bottom=0)
	ax.set_xlim(right=11)
	fig.savefig("error_vs_gaussians.png", format="png", dpi=300)

	fig, ax = plt.subplots(figsize=[7, 4])
	ax.scatter(ns, iters, linewidth=0.1, s=70)
	ax.set_xlabel("Number of Gaussians")
	ax.set_ylabel("Iterations to converge")
	ax.set_ylim(bottom=0)
	ax.set_xlim(right=11)
	fig.savefig("iterations_vs_gaussians.png", format="png", dpi=300)
        
    
	# see if relative weights of Gaussians can matter
	real_means = [0, 2]
	real_sigmas = [1, 1]
	errors = []
	iters = []
	ns = range(1, 10)
	for n in ns:
		this_error = []
		this_iter = []
		for i in range(20):
			dataset = create_dataset(real_means, real_sigmas, [50, 150*n])
			means, sigmas, iterations = gaussian_mixture(dataset, 2)
			means.sort()
        
			err = error(means, sigmas, real_means, real_sigmas)
			this_error.append(err)
			this_iter.append(iterations)
            
		errors.append(np.median(this_error))
		iters.append(np.median(this_iter))

	fig, ax = plt.subplots(figsize=[7, 4])    
	ax.scatter(ns, errors, linewidth=0.1, s=70)
	ax.set_xlabel("Relative sizes of the two Gaussians")
	ax.set_ylabel("Error")
	ax.set_ylim(bottom=0)
	ax.set_xlim(right=11)
	fig.savefig("error_vs_relative_size.png", format="png", dpi=300)

	fig, ax = plt.subplots(figsize=[7, 4])
	ax.scatter(ns, iters, linewidth=0.1, s=70)
	ax.set_xlabel("Relative sizes of the two Gaussians")
	ax.set_ylabel("Iterations to converge")
	ax.set_ylim(bottom=0)
	ax.set_xlim(right=11)
	fig.savefig("iterations_vs_relative_size.png", format="png", dpi=300)



if __name__ == "__main__":
	main()