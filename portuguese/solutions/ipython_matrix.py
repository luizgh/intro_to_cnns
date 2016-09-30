A = np.ones((3,4)) * 4
b = np.ones(3) * 2
c = np.array([10,20,30,40])
print A.T.dot(b) + c

def my_sigmoid(x):
    return 1./(1 + np.exp(-x))

print 'Resultados sigmoid:'
print my_sigmoid(0)
print my_sigmoid(np.array([-100,0,100]))
