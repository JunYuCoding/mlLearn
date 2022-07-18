import numpy as np
W = np.array([[1,111,3]
             ,[1,2,3]])

scores = np.array([1, 2, 3])
scores = np.array([123, 456, 789])
scores -= np.max(scores)
print(scores)
p = np.exp(scores) / np.sum(np.exp(scores))
print(p) # [5.75274406e-290 2.39848787e-145 1.00000000e+000]