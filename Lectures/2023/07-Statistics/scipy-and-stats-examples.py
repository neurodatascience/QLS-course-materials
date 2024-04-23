#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np  # Importing numpy
import pandas as pd  # Importing the pandas module
import statsmodels as stm
from nilearn.datasets import fetch_abide_pcp

get_ipython().run_line_magic("matplotlib", "inline")


# # Part I : introduction to scipy and csv wrangling

# ## A couple of scipy examples

# ### scipy example 0 : linalg

# In[2]:


from scipy import linalg as lin

A = np.array([[1, 2], [3, 4], [5, 6]])

# svd decomposition
u, la, vt = lin.svd(A, 0)

print(vt.shape)
print(u.shape)

l1, l2 = la[0], la[1]
print(l1, l2)  # eigenvalues
# print('u[:,0] = ', u[:,0])
# print('vt[0,:] = ',vt[0,:])   # first eigenvector

print(np.sum(abs(vt**2), axis=0))  # eigenvectors are unitary

assert np.allclose(np.sum(abs(vt**2), axis=0), 1)

print(lin.norm(A - u @ np.diag(la) @ vt))  # check the computation


# ### scipy example 1 : optimize

# In[3]:


from scipy import optimize

# In[4]:


x_data = np.linspace(-5, 5, num=50)

y_data = 2.9 * np.sin(1.5 * x_data) + np.random.normal(size=50)

fig, ax = plt.subplots()
ax.plot(x_data, y_data)


def test_func(x, a, b):
    return a * np.sin(b * x)


# optimize
params, params_covariance = optimize.curve_fit(test_func, x_data, y_data, p0=[2, 2])
print(params)

y_fitted = test_func(x_data, params[0], params[1])
ax.plot(x_data, y_fitted, "-o")


# ### scipy example 2:  ndimage

# In[5]:


# img = plt.imread('./moonlanding.png').astype(float)
# plt.imshow(img)
# plt.title('Original image')

np.random.seed(0)
x, y = np.indices((100, 100))
sig = np.sin(2 * np.pi * x / 50.0) * np.sin(2 * np.pi * y / 50.0) * (1 + x * y / 50.0**2) ** 2
mask = sig > 1

plt.figure(figsize=(5, 2.5))
plt.subplot(1, 2, 1)
plt.imshow(sig)
plt.axis("off")
plt.title("sig")
plt.subplot(1, 2, 2)
plt.imshow(mask, cmap=plt.cm.gray)
plt.axis("off")
plt.title("mask")
plt.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.9)


# In[6]:


from scipy import ndimage

labels, nb = ndimage.label(mask)
plt.figure(figsize=(2.5, 2.5))
plt.imshow(labels)
plt.title("label")
plt.axis("off")
plt.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.9)


# In[7]:


sl = ndimage.find_objects(labels == 4)
plt.figure(figsize=(2.5, 2.5))
plt.imshow(sig[sl[0]])
plt.title("Cropped connected component")
plt.axis("off")
plt.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.9)


# In[ ]:


# * **Signal processing:** scipy.signal
# * **Integration:** scipy.integrate
# * **Image processing:**" scikit image
# * **Stats:** scipy.stats
# * **optimization:** scipy.optimize
# * **linear algebra:** scipy.linalg
# * **differential equations** scipy....
# * **etc etc etc etc...**

# ## CSV wrangling

# In[8]:


# data_dir="" arg if you wish to store this file elsewhere, default is $HOME/nilearn_data
abide = fetch_abide_pcp(legacy_format=False, derivatives=[])


# In[9]:


data = abide.phenotypic
print(type(data), len(data))
colnames = list(abide.phenotypic.columns)


# In[10]:


print(colnames)


# In[11]:


to_keep = [
    "i",
    "SUB_ID",
    "SITE_ID",
    "DX_GROUP",
    "AGE_AT_SCAN",
    "SEX",
    "FIQ",
    "VIQ",
    "PIQ",
    "func_mean_fd",
]


# In[12]:


kdata = data[to_keep]


# In[13]:


print(kdata.columns)


# In[14]:


kdata.describe()


# In[15]:


plt.hist(kdata["func_mean_fd"], bins=50)


# In[16]:


plt.hist(kdata["SEX"], bins=50)
#
print(type(kdata["SEX"]))
print(np.unique(kdata["SEX"]))
print(pd.unique(kdata["SEX"]))


# In[17]:


my_var = "VIQ"
plt.hist(kdata[my_var], bins=50)
#
print(type(kdata[my_var]))
print(np.unique(kdata[my_var]), len(np.unique(kdata[my_var])))


# OK, -9999 seems like encoding a missing value. Let's make that assumption

# In[18]:


type(kdata[kdata["VIQ"] < 50].index)


# In[19]:


# ckdata = kdata.drop
kdata = kdata.drop(kdata[kdata["VIQ"] < 50].index)
kdata.describe()


# What is wrong with this method ? Note that we haven't yet defined how we will analyze these data.

# In[20]:


list(kdata.columns)[1:]


# In[21]:


kdata = data[to_keep]
for mycol in ["VIQ", "FIQ", "PIQ", "func_mean_fd"]:  # ,'func_mean_fd','SEX','AGE_AT_SCAN']:
    try:
        kdata = kdata.drop(kdata[kdata[mycol] < -9998].index)
    except:
        print(mycol)


# In[22]:


kdata = kdata.dropna()


# In[23]:


kdata.describe()


# What if the NaN values or missing values are not randomly distributed ?

# In[24]:


# Let's assume 1 is male and 2 is female
kdata = kdata.replace({"SEX": {1: "male", 2: "female"}})


# In[25]:


print(kdata.head())
kdata.describe()


# Ok, we are all set, right ? dataset is clean. Let's do a last check, surely for nothing.

# In[26]:


plt.hist(kdata["FIQ"], bins=50)


# In[ ]:


# In[27]:


print("kdata.shape: ", kdata.shape)
print("kdata.columns: ", kdata.columns)  # It has columns
print("\nFemale VIQ mean: ", kdata[kdata["SEX"] == "female"]["VIQ"].mean())
print("\nMale VIQ mean: ", kdata[kdata["SEX"] == "male"]["VIQ"].mean())


# In[28]:


# "Group by"
groupby_sex = kdata.groupby("SEX")
for sex, value in groupby_sex["PIQ"]:
    print((sex, value.mean()))


# In[29]:


# how would you investigate what is a groupby_sex object ?


# In[ ]:


# # Part II : simple stats with scipy stats

# In[30]:


from scipy import stats, stats as sst

# ## A quick detour by an effect size question

# Cohen's d effect size :
#
# $\hspace{3cm} d = \frac{\mu}{\sigma}$
#
# $\mu$ the non normalized effect size, $\sigma$ the standard deviation of the **data**
#
# Author report: APOE effect on hippocampal volume has a p value of 6.6311e-10, n=733
# What is the effect size of APOE on the hippocampal volume ?
#

# In[31]:


# create a normal(0,1) variable
p_val = 6.6311e-10
sample_size = 733

n01 = sst.norm(0, 1.0)
t733 = sst.t(df=sample_size)

z = n01.isf(p_val)
t = t733.isf(p_val)

check_p = 1 - n01.cdf(z)
# print(check_p)
assert np.isclose(check_p, p_val)

d = n01.isf(p_val) / np.sqrt(sample_size)
d_with_t_distrib = t733.isf(p_val) / np.sqrt(sample_size)

print("z = {:4.3f}; d_with_normal    = {:4.3f};".format(z, d))
print("t = {:4.3f}; d_with_t_distrib = {:4.3f};".format(t, d_with_t_distrib))


# In[32]:


# scipy stats has a great number of distribution, all with pdf cdf, sf, isf, etc ...
# but you can also sample from these:
n, start, width = 1000, 10, 20
unif10_20 = sst.uniform(loc=start, scale=width)
data_uniform = unif10_20.rvs(size=(n,))

fig, ax = plt.subplots(1, 1)
_ = ax.hist(data_uniform, bins=50, color="skyblue", linewidth=15, alpha=1)

ax.set_xlabel("Uniform Distribution")
ax.set_ylabel("count")


# ## Probability distributions : discrete, continuous

# ![title](stats-distrib.png)

# In[ ]:


#
# ## Simple statistical tests:
#
# https://en.wikipedia.org/wiki/Statistical_hypothesis_testing
#
# we will use the `scipy.stats` sub-module of [scipy](http://docs.scipy.org/doc/)

# ## Student's t-test: one of the simplest statistical test

# `scipy.stats.ttest_1samp`
#
# tests if the population mean of data is likely to be equal to a given value (technically if observations are
# drawn from a Gaussian distributions of given population mean). It returns the [T statistic](https://en.wikipedia.org/wiki/Student%27s_t-test), and the [p-value](https://en.wikipedia.org/wiki/P-value) (see the
# function's help)

# In[33]:


stats.ttest_1samp(kdata["VIQ"], 110)


# ## Wait, what's a p-value again?
#
# Probability of observing a statistic equal to the one seen in the data,
# or one that is more extreme, when the null hypothesis is true

# Requires:
# * Knowledge of the null hypothesis
# * Choice of a statistic
# * Concept of repeating the whole study in the same way
#     - Same study design
#     - Same sampling scheme
#     - Same definition of the statistic

# In[ ]:


# ### 2-sample t-test: testing for difference across populations
#
# We have seen above that the mean VIQ in the  male and
# female populations were different. To test if this is
# significant, we do a 2-sample t-test with
#
# `scipy.stats.ttest_ind`
#

# In[34]:


np.set_printoptions(precision=4)  # he non !

my_col = "PIQ"

female_viq = kdata[kdata["SEX"] == "female"][my_col]
male_viq = kdata[kdata["SEX"] == "male"][my_col]
stats.ttest_ind(female_viq, male_viq, alternative="less")


# In[35]:


# sst.ttest_ind?


# Even with the most simple test, many hypotheses and many things to know. Investing in a stat course is a good idea.

# ### Paired tests: repeated measurements on the same individuals
#
# PIQ, VIQ, and FIQ give 3 measures of IQ.
# Let us test if FIQ and PIQ are significantly
# different. We can use a 2 sample test

# In[36]:


# Box plots of different columns for each gender
groupby_sex.boxplot(column=["FIQ", "VIQ", "PIQ"])
groupby_sex.groups


# In[37]:


female_viq = kdata[kdata["SEX"] == "female"]["VIQ"]
female_piq = kdata[kdata["SEX"] == "female"]["PIQ"]
stats.ttest_ind(female_viq, female_piq)


# ### Can you see a problem with this approach ?

# The problem with this approach is that it forgets that there are links between
# observations: FSIQ and PIQ are measured on the same individuals. Thus the
# variance due to inter-subject variability is confounding, and can be removed,
# using a "paired test", or ["repeated measures test"](https://en.wikipedia.org/wiki/Repeated_measures_design)

# In[38]:


sst.ttest_rel(kdata["FIQ"], kdata["PIQ"])


# In[39]:


# T-tests assume Gaussian errors.
# We can use a [Wilcoxon signed-rank test](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test) that relaxes this assumption:

stats.wilcoxon(kdata["FIQ"], kdata["PIQ"])


# ## Linear regression with statsmodels

# In[40]:


kdata.columns


# In[41]:


from statsmodels.formula.api import ols

model = ols("VIQ ~ SEX + DX_GROUP + AGE_AT_SCAN + func_mean_fd", kdata).fit()
print(model.summary())


# In[42]:


# model.f_test?


# In[43]:


# Here, we don't need to define a contrast, as we are testing a single
# coefficient of our model, and not a combination of coefficients.
# However, defining a contrast, which would then be a 'unit contrast',
# will give us the same results
A = np.identity(5)
model.f_test(A[4, :])


# In[44]:


model.model.endog_names


# In[45]:


model.model.exog_names


# In[ ]:
