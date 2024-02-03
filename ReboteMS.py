import numpy as np
import sympy as sp
from scipy.misc import derivative
from scipy.optimize import newton
from timeit import default_timer as timer

#---------------------------------------------------------------------------------------------------------------------
# Metodos auxiliares
t0Newton = 0.
maxiterNewton = 250
tolNewton = 1e-5
# Interseca la superficie con la recta, retornando el punto de interseccion
# vd: vector director de la recta
# pp: punto de paso de la recta
def intersecarConRecta(fi, vd, pp):
    # Reemplaza los componentes de la recta en la superficie
    def fAux(t, fi, vd, pp):
        p = pp + vd * t
        return fi(*p)

    # Newton escalar
    try:
        tRaiz = newton(fAux, t0Newton, args = (fi, vd, pp), maxiter = maxiterNewton, tol = tolNewton)
        return pp + vd * tRaiz
    except:
        return pp

# Da el normal a la superficie (o el normal al plano tangente a la superficie) en el punto
def grad(fi, p):
    dx = derivadaParcial(fi, p, 0)
    dy = derivadaParcial(fi, p, 1)
    dz = derivadaParcial(fi, p, 2)

    return np.array([dx, dy, dz])

h = 1e-5
def derivadaParcial(fi, p, nroVar = 0):
    pAux = p.tolist()
    def fAux(xI):
        pAux[nroVar] = xI
        return fi(*pAux)

    return derivative(fAux, p[nroVar], dx = h) 

#---------------------------------------------------------------------------------------------------------------------
def reboteMS(f1, f2, f3, p0, maxiter = 50, tol = 1e-5):
    try: #Por si llegara a ser un punto que no este definido para esta superficie
        if np.all(np.isclose([f1(*p0), f2(*p0), f3(*p0)], np.zeros(3), tol)): # Si ya llegaste, no te muevas
            return 0, p0   
        else:
            return reboteMSAux(f1, f2, f3, p0, maxiter, tol)
    except Exception as e:
        return 0, "Alguna de las funciones no pudo ser evaluada en el punto inicial "+ str(p0) +": " + str(e)

# Metodo iterativo
def reboteMSAux(f1, f2, f3, p0, maxiter, tol):

    pn = p0
    i = 0
    while i < maxiter and tol < np.linalg.norm([f1(*pn), f2(*pn), f3(*pn)], np.inf):
            
        seguir, q = hallarRectaEInterseccion(f1, f2, f3, pn)
        if seguir:
            seguir, q = hallarRectaEInterseccion(f2, f3, f1, q)
            if seguir:
                seguir, q = hallarRectaEInterseccion(f1, f3, f2, q)
                if seguir:
                    pn = q
                else:
                    return i, q
            else:
                return i, q
        else:
            return i, q
           
        i += 1

    return i, pn

# Halla el producto cruz entre los vectores normales a las dos primeras superficies en un
# punto, luego construye una recta y la interseca con la tercera superficie
def hallarRectaEInterseccion(fi, fj, fk, p):
    
    n_fi = grad(fi, p)
    n_fj = grad(fj, p)
    vd = np.cross(n_fi, n_fj)

    if np.any(np.isnan(vd)):
        print("El método no puede continuar (gradiente no definido)")
        return False, p
    else:
        if np.all(np.isclose(vd, np.zeros(3))):
            print("El método no puede continuar (vector director de la recta nulo).")
            return False, p
        else:
            return True, intersecarConRecta(fk, vd, p)


#---------------------------------------------------------------------------------------------------------------------
# Casos de prueba:
def tests():

    print("------------------------------- Caso 1 -------------------------------")

    f1 = lambda x, y, z: z - 2
    f2 = lambda x, y, z: x + y + z + 1
    f3 = lambda x, y, z: x + 2
    start = timer()
    i,r = reboteMS(f1, f2, f3, np.array([0, 0, 0])) 
    caso1 = timer()
    print("{z - 2 = 0, x + y + z + 1 = 0, x + 2 = 0} - Punto inicial: [0,0,0] ")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso1-start) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 2 -------------------------------")

    f1 = lambda x, y, z: z
    f2 = lambda x, y, z: x ** 2 + y ** 2 + z
    f3 = lambda x, y, z: x
    i,r = reboteMS(f1, f2, f3, np.array([-40, 38, 22]))
    caso2 = timer()
    print("{z = 0, x^2 + y^2 + z = 0, x = 0}: - Punto inicial: [-40,39,22] ")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso2-caso1) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 3 -------------------------------")

    f1 = lambda x, y, z: np.cos(x) - z
    f2 = lambda x, y, z: x ** 2 + y ** 2 + 1 - z
    f3 = lambda x, y, z: x + y + z - 1
    i,r = reboteMS(f1, f2, f3, np.array([1, 0, 0]))
    caso3 = timer()
    print("{cos(x) - z = 0, x^2 + y^2 + 1 - z = 0, x + y + z - 1 = 0}: - Punto inicial: [1,0,0] ")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso3-caso2) + " s.")
    print("Iteraciones realizadas: " + str(i))
    i,r = reboteMS(f1, f2, f3, np.array([1, 1, 1]))
    caso3b = timer()
    print("Nuevo punto inicial: [1,1,1]")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso3b-caso3) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 4 -------------------------------")

    f1 = lambda x, y, z: z - 1
    f2 = lambda x, y, z: z - 2
    f3 = lambda x, y, z: z - 3
    i,r = reboteMS(f1, f2, f3, np.array([1, 1, 1]))
    caso4 = timer()
    print("{z - 1 = 0, z - 2 = 0, z - 3 = 0}: - Punto inicial: [1,1,1] ")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso4-caso3b) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 5 -------------------------------")

    f1 = lambda x, y, z: z - np.exp(x * y)
    f2 = lambda x, y, z: -x + y - z - 4
    f3 = lambda x, y, z: (y - 3)**2 + x - z
    i,r = reboteMS(f1, f2, f3, np.array([-3, 4, 5])) 
    caso5 = timer()
    print("{z - e^(xy) = 0, - x + y - z - 4 = 0, (y - 3)^2 + x - z = 0}: - Punto inicial: [-3,4,5] ")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso5-caso4) + " s.")
    print("Iteraciones realizadas: " + str(i))
    i,r = reboteMS(f1, f2, f3, np.array([0,0,0]))
    caso5b = timer()
    print("Nuevo punto inicial: [0,0,0]")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso5b-caso5) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 6 -------------------------------")

    f1 = lambda x, y, z: x * np.cos(y) - z
    f2 = lambda x, y, z: x**2 + y**2 - 2 - z
    f3 = lambda x, y, z: (1/x) + y**2 - 2 - z 
    i,r = reboteMS(f1, f2, f3, np.array([1,0,0])) 
    caso6 = timer()
    print("{cos(y) . x - z = 0, x^2 + y^2 - 2 - z = 0, (1/x) + y**2 - 2 - z = 0}: - Punto inicial: [1,0,0] ")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso6-caso5b) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 7 -------------------------------")

    f1 = lambda x, y, z: (np.cos(x)/x) + y - z
    f2 = lambda x, y, z: z**2 + x - 2 - z
    f3 = lambda x, y, z: (1/y) + 2*x - z
    i,r = reboteMS(f1, f2, f3, np.array([1,1,0])) 
    caso7 = timer()
    print("{cos(x)/x - z = 0, z^2 + x - 2 - z = 0, (1/y) + 2*x - z = 0}: - Punto inicial: [1,1,0] ")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso7-caso6) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 8 -------------------------------")

    f1 = lambda x, y, z: (np.cos(y)*x) + np.sin(x) - z
    f2 = lambda x, y, z: (1/np.sin(x)) + 4 - z
    f3 = lambda x, y, z: y + x - z
    i,r = reboteMS(f1, f2, f3, np.array([2,3,4])) 
    caso8 = timer()
    print("{cos(y)*x + sen(x) - z = 0, 1/sen(x) + 4 - z = 0, y + x - z = 0}: - Punto inicial: [2,3,4] ")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso8-caso7) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")

    print("------------------------------- Caso 9 -------------------------------")

    f1 = lambda x, y, z: x**2 + y - 1
    f2 = lambda x, y, z: y
    f3 = lambda x, y, z: np.log(x) - z
    i,r = reboteMS(f1, f2, f3, np.array([0,4,1]))
    caso9 = timer()
    print("{x^2 + y - 1 = 0, y = 0, ln(x) - z = 0}: - Punto inicial: [0,4,1] ")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso9-caso8) + " s.")
    print("Iteraciones realizadas: " + str(i))
    i,r = reboteMS(f1, f2, f3, np.array([1,4,1]))
    caso9b = timer()
    print("Nuevo punto inicial: [1,4,1]")
    print("Solución: " + str(r))
    print("Tiempo ReboteMS: " + str(caso9b-caso9) + " s.")
    print("Iteraciones realizadas: " + str(i))
    print("")


#---------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    tests()

