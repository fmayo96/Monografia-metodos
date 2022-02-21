import numpy as np
import matplotlib.pyplot as plt

def integrar_difusion_2d(f0, ccx, ccy, dx, dy, dt, tf, nu, orden_t=1):
    """
        Integra la ecuación de difusión 2D con condiciones de contorno de tipo
    Dirichlet estacionarias. Utiliza diferencias finitas de segundo orden para
    la parte espacial y un método de Runge-Kutta explícito de orden variable
    para la parte temporal.

    Entradas:
        `f0`:  Arreglo bidimensional (NX,NY), con las condiciones iniciales para
                todos los puntos de la grilla (incluídos los contornos).
        `ccx`: Arreglo bidimensional (2,NY) con los valores con la condición de
                Dirichlet para x=0 (ccx[0]) y x=Lx (ccx[1]).
        `ccy`: Arreglo bidimensional (2,NX) con los valores con la condición de
                Dirichlet para y=0 (ccy[0]) y y=Ly (ccy[1]).                
        `dx`: Espaciamiento entre puntos en la dirección x.
        `dy`: Espaciamiento entre puntos en la dirección y.
        `dt`: Espaciamiento entre puntos en tiempo.
        `tf`: Tiempo final de integración.
        `nu`: Difusividad (constante).

        `orden_t`: Orden de la integración temporal (OPCIONAL)
    
    Salida:
        `f`:  Arreglo tridimensional f(t,x,y) con la solución para cada paso
                temporal y para cada punto de la grilla (incluyendo los
                contornos).
    """
    from scipy.sparse import diags
    import scipy
    # Cantidad de pasos temporales a realizar
    pasos = int(round(tf/dt))

    # Cantidad de puntos interiores en cada dirección
    nx, ny = f0.shape[0]-2, f0.shape[1]-2

    # Arreglo que va a contener a la solución
    f    = np.zeros( (pasos+1, nx, ny) )
    f[0] = f0[1:-1,1:-1]

    # -------------- NO MODIFICAR ARRIBA DE ESTA LÍNEA -------------------------

    # COMPLETAR: Obtener operadores que permitan estimar la derivada segunda en
    # cada dirección para cada punto interior del dominio.
    # Recuerde que para la primer y la última tira de puntos en cada dirección 
    # deberá utilizar una matrices que contemple condiciones de contorno.
    # --------------------------------------------------------------------------
    #Definimos el operador diferencial derivada segunda centrada de segundo orden.
    Dx = diags([-0.5*np.ones(nx-1), 0.5*np.ones(nx-1)], offsets=[-1,1])
    Dy = diags([-0.5*np.ones(ny-1), 0.5*np.ones(ny-1)], offsets=[-1,1])
    Dx /= dx
    Dy /= dy 
    Dy = Dy.T
    Dxx = diags([-2*np.ones(nx), np.ones(nx-1), np.ones(nx-1)], offsets = [0,1,-1])
    Dxx /= dx**2
    Dyy = diags([-2*np.ones(ny), np.ones(ny-1), np.ones(ny-1)], offsets = [0,1,-1])
    Dyy /= dy**2
    Dyy = Dyy.T 
    bx = np.zeros([nx,ny])
    by = np.zeros([nx,ny])
    bx[0,:], bx[-1,:] = ccx[:,1:-1]/dx
    by[:,0], by[:,-1] = ccy[:,1:-1]/dy
    b = bx - by
    bxx = np.zeros([nx, ny])
    byy = np.zeros([nx, ny])
    bxx[0,:], bxx[-1,:] = ccx[:,1:-1]/dx**2
    byy[:,0], byy[:,-1] = ccy[:,1:-1]/dy**2
    b2 = bxx - byy
    #Completamos con el integrador de Runge Kutta orden_t
    
    for n in range(pasos):
        K1 = 1j/(2*m)*(Dxx*f[n] - f[n]*Dyy) - gamma*(X[1:-1,1:-1] - Y[1:-1,1:-1])*(Dx*f[n] - f[n]*Dy) - 2*m*gamma*T*(X[1:-1,1:-1] - Y[1:-1,1:-1])**2*f[n]
        K2 = 1j/(2*m)*(Dxx*(f[n] + K1*dt/2) - (f[n] + K1*dt/2)*Dyy) - gamma*(X[1:-1,1:-1] - Y[1:-1,1:-1])*(Dx*(f[n] + K1*dt/2) - (f[n] + K1*dt/2)*Dy) - 2*m*gamma*T*(X[1:-1,1:-1] - Y[1:-1,1:-1])**2*(f[n] + K1*dt/2)
        K3 = 1j/(2*m)*(Dxx*(f[n] + K2*dt/2) - (f[n] + K2*dt/2)*Dyy) - gamma*(X[1:-1,1:-1] - Y[1:-1,1:-1])*(Dx*(f[n] + K2*dt/2) - (f[n] + K2*dt/2))*Dy - 2*m*gamma*T*(X[1:-1,1:-1] - Y[1:-1,1:-1])**2*(f[n] + K2*dt/2)
        K4 = 1j/(2*m)*(Dxx*(f[n] + K3*dt) - (f[n] + K3*dt)*Dyy)- gamma*(X[1:-1,1:-1] - Y[1:-1,1:-1])*(Dx*(f[n] + K3*dt) - (f[n] + K3*dt)*Dy) - 2*m*gamma*T*(X[1:-1,1:-1] - Y[1:-1,1:-1])**2*(f[n] + K3*dt)
        f[n + 1] = f[n] + 1/6 * dt * (K1 + K2 + K3 + K4)
        
    
    # ----------------- NO MODIFICAR DEBAJO DE ESTA LÍNEA ----------------------
    # Agrego los bordes
    f = np.hstack((ccx[0,None,1:-1]*np.ones( (pasos+1,1,ny) ), f,
                   ccx[1,None,1:-1]*np.ones( (pasos+1,1,ny) ) ))
    f = np.dstack((ccy[0,:,None]*np.ones( (pasos+1,nx+2,1) ), f,
                   ccy[1,:,None]*np.ones( (pasos+1,nx+2,1) ) ))

    # Para t=0 considero los bordes de la condición inicial
    f[0,1:-1,1:-1]               = f0[1:-1,1:-1]
    f[:,[0,0,-1,-1],[0,-1,0,-1]] = np.nan         # Remuevo las esquinas

    return f


def h(y):
  return np.zeros(len(y))

def g(y):
  return np.zeros(len(y))

def u(x):
  return np.zeros(len(x))

def v(x):
  return np.zeros(len(x))


m = 1
gamma = 1e-2
T = 300
Nx = 128
Ny = 128
tf = 10
dt = 1e-3
Nt = int(tf/dt)
L = 1
d = 0.05
D = 0.25 
Lx, Ly = 1 , 1
Nx, Ny = 64, 64
nu = 0.1
dt = 1e-3
tf = 15
x, dx = np.linspace(0,Lx, Nx ,endpoint=True, retstep=True)
y, dy = np.linspace(0,Ly, Ny ,endpoint=True, retstep=True)

X,Y = np.meshgrid(x,y)
f0 = 0.5/(2*np.pi*d**2)*(np.exp(-(X - 0.5 + D/2)**2/(2*d**2)) + np.exp(-(X - 0.5 - D/2)**2/(2*d**2))) * (np.exp(-(Y - 0.5 + D/2)**2/(2*d**2)) + np.exp(-(Y - 0.5 - D/2)**2/(2*d**2)))

f = integrar_difusion_2d(f0, np.array([g(y), h(y)]), np.array([u(x), v(x)]), dx, dy, dt, tf, nu, orden_t=1)

fig, ax = plt.subplots(1,1, figsize = (7,7))
col = ax.imshow(f[-1].T, origin="lower", extent=(x[0], x[-1], y[0], y[-1]) )
fig.colorbar(col)
plt.xlabel('x')
plt.ylabel('y')
plt.show()