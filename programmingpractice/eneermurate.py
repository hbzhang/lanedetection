
import numpy as np

l = range(5, 10)

print(l[0])
print(len(l))
for i, obj in enumerate(l):
    print(str(i) + " " + str(obj))

    if(i<len(l)-1):
        print(str(l[i]) + " " + str(l[i+1]))


x = np.array([[1,2,3], [4,5,6]], dtype =np.int32)


print(type(x))

print(x[1][0])

print(x.dtype)

print(x[-2,1])

print(x[:-1])


#y= x.reshape([2,2])

#print(y)
