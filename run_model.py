import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az

# Read in the data
df = pd.read_csv('samples.csv') ## Path to the data file ## Your code here
##n =  len(df) # Number of samples    
n = df.shape[0] # Number of samples
## Your code here

# When is low tide?
starttime =  df['timestamp'].min() ## Your code here
## Your code here

# Make seconds from lowtide using timestamps
seconds = np.arange(0, n) * 60

##seconds =np.arange(0, n) ## Your code here
##seconds = np.array((df['timestamp'] - starttime).dt.total_seconds()) ## Your code here
##seconds = (df['timestamp'] - starttime).dt.total_seconds() ## Your code here 
## Your code here

# Get the fish counts as a numpy array
fish_counts =  df['jellyfish_entering'].values ## Your code here
## Your code here

# How many seconds between lowtides?
period = 12.0 * 60.0 * 60.0

# Create a model
basic_model = pm.Model()
## Your code here
with basic_model:

    # Give priors for unknown model parameters
    magnitude = pm.Uniform('magnitude', lower=0, upper=200) ## Your code here
  
    sigma =  pm.Uniform('sigma', lower=0, upper=100) ## Your code here
    

    # Create the model
    ## Your code here
   ## model = magnitude * pm.math.sin(2 * np.pi * seconds / period)
    expected_count = magnitude * pm.math.sin(2 * np.pi * seconds / period) ## Your code here   
    ## Your code here
    
   ## model = magnitude * pm.math.sin(2 * np.pi * seconds / period)
    ##expected_count = pm.Poisson('expected_count', mu=basic_model, observed=fish_counts)
    # Make chains
    trace = pm.sample(2000, tune=500, cores=1) ## Your code here
    ## Your code here

    # Find maximum a posteriori estimations
    map_magnitude = pm.find_MAP(model=basic_model)['magnitude'] ## Your code here 
    ## Your code here
    map_sigma =  pm.find_MAP(model=basic_model)['sigma'] ## Your code here
    ## Your code here

# Let the user know the MAP values
print(f"Based on these {n} measurements, the most likely explanation:")
print(
    f"\tWhen the current is moving fastest, {map_magnitude:.2f} jellyfish enter the bay in 15 min."
)
print(f"\tExpected residual? Normal with mean 0 and std of {map_sigma:.2f} jellyfish.")

# Do a contour/density plot
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
## Your code here


# Plot the data
ax.plot(seconds, fish_counts, '+', label='observed')
ax.plot(seconds, expected_count, label='expected')


##az.plot_kde(trace['magnitude'], ax=ax, label='magnitude')
##az.plot_kde(trace['sigma'], ax=ax, label='sigma')
ax.legend()

#az.plot_kde(trace['magnitude'], ax=ax, label='magnitude')
##az.plot_kde(trace['sigma'], ax=ax, label='sigma')
##ax.legend()
##ax.set_xlabel('Magnitude')
##ax.set_ylabel('Sigma')
plt.show()
fig.savefig("pdf.png")

# Plot your function and confidence against the observed data
fig, ax = plt.subplots(figsize=(8, 6))
## Your code here
ax.plot(seconds, fish_counts, 'o', label='observed')
ax.plot(seconds, map_magnitude * np.sin(2 * np.pi * seconds / period), label='model')
ax.set_xlabel('Seconds from low tide')
ax.set_ylabel('Jellyfish entering the bay')
ax.legend()
#plt.show()
#fig.savefig("pdf2.png")
    
fig.savefig("jellyfish.png")
