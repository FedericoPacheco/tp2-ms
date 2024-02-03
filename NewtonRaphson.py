import numpy as np
import sympy as sp
from sympy import *
from timeit import default_timer as timer

def Newton(F,x0,maxiter=2000,tol=1e-5):

    Y = Matrix([x, y, z])
    J = F.jacobian(Y)
    
    solucion=np.asarray((0,0,0))
    i = 0
    J0=np.asarray(J.subs({x:x0[0],y:x0[1],z:x0[2]}),dtype=float)
    F0=np.asarray(F.subs({x:x0[0],y:x0[1],z:x0[2]}),dtype=float)

    while(not(np.allclose(F0,solucion,0,1e-5)) and i < maxiter):

        y0=np.linalg.solve(J0,-F0).reshape(-1)
        x0+=y0
        F0=np.asarray(F.subs({x:x0[0],y:x0[1],z:x0[2]}),dtype=float)
        
        i+=1
    
    return x0,i


if __name__ == "__main__":
    
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')

    print("------------------------------- Caso 1 -------------------------------")

    F = Matrix([[z-2],[x + y + z + 1],[x + 2]])
    x0=np.asarray((0.,0.,0.))
    start = timer()
    r,i = Newton(F,x0)
    caso1 = timer()
    print("{z - 2 = 0, x + y + z + 1 = 0, x + 2 = 0}: " + str(r))
    print("Tiempo Newton-Raphson: " + str(caso1-start) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 2 -------------------------------")

    F = Matrix([[z],[x**2 + y**2 + z],[x]])
    x0=np.asarray((2.,2.,2.))
    r,i = Newton(F,x0)
    caso2 = timer()
    print("{z = 0, x^2 + y^2 + z = 0, x = 0}: " + str(r))
    print("Tiempo Newton-Raphson: " + str(caso2-caso1) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 3 -------------------------------")

    F = Matrix([[sp.cos(x) - z],[x ** 2 + y ** 2 + 1 - z],[x + y + z - 1]])
    x0=np.asarray((1.,0.,0.))
    r,i = Newton(F,x0)
    caso3 = timer()
    print("{cos(x) - z = 0, x^2 + y^2 + 1 - z = 0, x + y + z - 1 = 0}: " + str(r))
    print("Tiempo Newton-Raphson: " + str(caso3-caso2) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    """""
    print("------------------------------- Caso 4 -------------------------------")
    F = Matrix([[z-1],[z-2],[z-3]])
    x0=np.asarray((1.,0.,0.))
    r,i = Newton(F,x0)
    caso4 = timer()
    print("{z - 1 = 0, z - 2 = 0, z - 3 = 0}: " + str(r))
    print("Tiempo Newton-Raphson: " + str(caso4-caso3) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")
    #Devuelve matriz singular
    """""
    print("------------------------------- Caso 5 -------------------------------")

    F = Matrix([[z - sp.exp(x * y)],[-x + y - z - 4],[(y - 3)**2 + x - z]])
    x0=np.asarray((-3.,4.,5.)) # 0.,0.,0.
    r,i = Newton(F,x0)
    caso5 = timer()
    print("{z - e^(xy) = 0, - x + y - z - 4 = 0, (y - 3)^2 + x - z = 0}: " + str(r))
    print("Tiempo Newton-Raphson: " + str(caso5-caso3) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    """""
    print("------------------------------- Caso 6 -------------------------------")

    F = Matrix([[x * sp.cos(y) - z],[x**2 + y**2 - 2 - z],[(1/x) + y**2 - 2 - z ]])
    x0=np.asarray((1.,0.,0.))
    r,i = Newton(F,x0)
    caso6 = timer()
    print("{cos(y) . x - z = 0, x^2 + y^2 - 2 - z = 0, (1/x) + y**2 - 2 - z = 0}: " + str(r))
    print("Tiempo Newton-Raphson: " + str(caso6-caso5) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")
    #Devuelve matriz singular
    """""

    print("------------------------------- Caso 7 -------------------------------")

    F = Matrix([[(sp.cos(x)/x) + y - z],[z**2 + x - 2 - z],[(1/y) + 2*x - z]])
    x0=np.asarray((1.,1.,0.))
    r,i = Newton(F,x0)
    caso7 = timer()
    print("{cos(x)/x - z = 0, z^2 + x - 2 - z = 0, (1/y) + 2*x - z = 0}: " + str(r))
    print("Tiempo Newton-Raphson: " + str(caso7-caso5) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 8 -------------------------------")

    F = Matrix([[(sp.cos(y)*x) + sp.sin(x) - z],[(1/sp.sin(x)) + 4 - z],[y + x - z]])
    x0=np.asarray((2.,3.,4.))
    r,i = Newton(F,x0)
    caso8 = timer()
    print("{cos(y)*x + sen(x) - z = 0, 1/sen(x) + 4 - z = 0, y + x - z = 0}: " + str(r))
    print("Tiempo Newton-Raphson: " + str(caso8-caso7) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 9 -------------------------------")

    F = Matrix([[x**2 + y - 1],[y],[sp.log(x) - z]])
    x0=np.asarray((1.,4.,1.))
    r,i = Newton(F,x0)
    caso9 = timer()
    print("{cos(y)*x + sen(x) - z = 0, 1/sen(x) + 4 - z = 0, y + x - z = 0}: " + str(r))
    print("Tiempo Newton-Raphson: " + str(caso9-caso8) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")
    