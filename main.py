import numpy as np
import matplotlib.pyplot as plt
import module as m

# arr= np.array([1,2,3,4,5])
# print(arr)

# A= np.array([[1,2,3],[7,8,9]])
# # print(A)
# # A = np.array([[1,2,3],
# #              [7,8,9]])
# # print(A)
# A=np.array([[1,2,\
#              3],
#             7,8,9])
# # print(A)

# v=np.arange(1,7)
# print(v,"\n")
# v=np.arange(-2,7)
# print(v,"\n")
# v=np.arange(1,10,3)
# print(v,"\n")
# v=np.arange(1,10.1,3)
# print(v,"\n")
# v=np.arange(1,11,3)
# print(v,"\n")
# v=np.arange(1,2,0.1)
# print(v,"\n")


# v=np.linspace(1,3,4)
# print(v)
# v=np.linspace(1,10,4)
# print(v)

# X=np.ones((2,3))
# Y=np.zeros((2,3,4))
# Z=np.eye(3)
# Q=np.random.rand(2,5)
# # print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)
#
# U=np.block([[A],[X,Z]])
# print(U)


# V = np.block([[
# np.block([
# np.block([[np.linspace(1,3,3)],
# [np.zeros((2,3))]]) ,
# np.ones((3,1))])
# ],
# [np.array([100, 3, 1/2, 0.333])]] )
# print(V)

# P=np.block([[np.linspace(1,6,6)],[np.zeros((3,6))]])
# print(P)

#4.2
# print(V[0,2])
# print(V[3,0])
# print(V[3,3])
# print(V[-1,-1])
# print(V[-4,-3])
#
# print( V[3,:] )
# print( V[:,2] )
# print( V[3,0:3] )
# print( V[np.ix_([0,2,3],[0,-1])] )
# print("\n\n")
#
# print( V[3] )

#4.3
# Q = np.delete(V, 2, 0)

# print(Q)

# Q = np.delete(V, 3, 0)
# print(Q)
# v = np.arange(1,7)
# print("\n\n")
# print( np.delete(v, 3, 0) )

#4.4
# print(np.size(v),
# np.shape(v),
# np.size(V),
# np.shape(V))

#4.5
# A = np.array([[1, 0, 0],
# [2, 3, -1],
# [0, 7, 2]] )
# B = np.array([[1, 2, 3],
# [-1, 5, 2],
# [2, 2, 2]] )
# print( A+B )
# print( A-B )
# print( A+2 )
# print( 2*A )

# MM1 = A@B
# print(MM1)
# MM2 = B@A
# print(MM2)

# MT1 = A*B
# print(MT1)
# MT2 = B*A
# print(MT2)

# DT1=A/B
# print(DT1)


# C = np.linalg.solve(A,MM1)
# print(C)
# print(B)
# x = np.ones((3,1))
# b = A@x
# y = np.linalg.solve(A,b)
# print(y)

# PM = np.linalg.matrix_power(A,2)
# PT = A**2
# print(A)
# print(PT)
# print(PM)

# print(A.T)
# print(A.transpose())
# print(A.conj().T)
# print(A.conj().transpose())



# A == B
# A != B
# 2 < A
# A > B
# A < B
# A >= B
# A <= B

# np.logical_not(A)
# print(np.logical_and(A, B),
# np.logical_or(A, B),
# np.logical_xor(A, B))

# print( np.all(A) )
# print( np.any(A) )

# print( v > 4 )
# print( np.logical_or(v>4, v<2))
# print( np.nonzero(v>4) )
# print( v[np.nonzero(v>4) ] )
# print(A)
# print(np.max(A))
# print(np.min(A))
# print(np.max(A,0))
# print(np.max(A,1))
# print( A.flatten() )
# print( A.flatten('F') )

#Matlabplotlib
# x = [1,2,3]
# y = [4,6,5]
# plt.plot(x,y)
# plt.show()


# x = np.arange(0.0, 2.0, 0.01)
# y = np.sin(2.0*np.pi*x)
# plt.plot(x,y,'r:',linewidth=3)
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Nasz pierwszy wykres')
# plt.grid(True)
#
#
# plt.show()


# x = np.arange(0.0, 2.0, 0.01)
# y1 = np.sin(2.0*np.pi*x)
# y2 = np.cos(2.0*np.pi*x)
# plt.plot(x,y1,'r:',x,y2,'g')
# plt.legend(('dane y1','dane y2'))
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Wykres')
# plt.grid(True)
# plt.show()


# x = np.arange(0.0, 2.0, 0.01)
# y1 = np.sin(2.0*np.pi*x)
# y2 = np.cos(2.0*np.pi*x)
# y = y1*y2
# l1, = plt.plot(x,y,'b')
# l2,l3 = plt.plot(x,y1,'r:',x,y2,'g')
# plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
# plt.xlabel('Czas')
# plt.ylabel('Pozycja')
# plt.title('Wykres')
# plt.grid(True)
# plt.show()

#ZADANIA
#z3
A = np.block([np.block([
    [np.block([
        [np.arange(1,6)],
        [np.arange(5,0,-1)]
    ])],
    [np.block([
    np.zeros((3,2)),np.block([[np.full((2,3),2)],[np.arange(-90,-60,10)]])])]]),np.full((5,1),10)])
print(A)
#z4
B=A[1]+A[3]
print("\n\n",B)

#z5
C=np.max(A,axis=0)
print("\n\n",C)

#z6
D=np.delete(B,(0,-1))
print("\n\n",D)

#z7
D[D==4]=0
print("\n\n",D)

#z8

index=[max(C)]
E=np.delete(C,np.argwhere((C==np.max(C)) | (C==np.min(C))))
print("\n\n",E)

#z9
# row,col=A.shape
# for i in range (row):
#     if(np.max(A[i])==np.max(A) and np.min(A[i])==np.min(A)): print(A[i])

#z10
# MM1=D@E
# print("\n",MM1)
# MT1=D*E
# print("\n",MT1)

#z11

# def fun1(a):
#     A=np.random.randint(11, size=(a, a))
#     print("\n",A)
#     s=0
#     for i in range(a):
#         s+=A[(i,i)]
#     print(s)
#
# fun1(4)

#z12

# def fun2(a):
#     A=np.random.randint(11, size=(a, a))
#     print("\n",A)
#     s=0
#     for i in range(a):
#         A[(i,i)]=0
#         A[(a-i-1,i)]=0
#     print("\n",A)
#
# fun2(8)

#z13

def fun3(a):
    A=np.random.randint(11, size=(a, a))
    print("\n",A)
    s=0
    for i in range(a):
        for j in range(a):
            if((i%2)!=0): s+=A[i,j]


    print(s)

fun3(6)

#z14
x=np.arange(-10,10.1,0.1)
y=lambda x:plt.plot(x,np.cos(2*x),'--',color='red')
y(x)

#z15
m.mod(x)


#z17
p=3*np.cos(2*x)+m.mod(x)
plt.plot(x,p,'*',color='blue')
#plt.show()

#z18
AA=np.array([[10 , 5, 1, 7],
           [10, 9, 5, 5],
           [1,6 ,7 ,3],
            [10, 0 ,1 , 5]])
BB=np.array([[34],[44],[25],[27]])

AAi=np.linalg.inv(AA)
Xr=AAi@BB

print(Xr)

