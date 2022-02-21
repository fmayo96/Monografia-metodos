import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from scipy.sparse import diags
import scipy
#-------------------Parametros------------------
m = 1
gamma = 1e-2
T = 300
Nx = 128
Ny = 128
tf = 20
dts = np.array([1e-1, 1e-2, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
rho_prev = np.zeros([Nx,Ny])
err = []
for dt in dts:
    Nt = int(tf/dt)
    L = 1
    d = 0.05
    D = 0.25 
    #-------------------Operadores-------------------
    x, dx = np.linspace(0,L,Nx, retstep=True)
    y, dy = np.linspace(0,L,Ny, retstep=True)
    X,Y = np.meshgrid(x,y)
    Dx = diags([0.5*np.ones(Nx-1), -0.5*np.ones(Nx-1), 0.5, -0.5], offsets=[-1,1, Nx-1, -Nx+1])
    Dy = diags([0.5*np.ones(Ny-1), -0.5*np.ones(Ny-1), 0.5, -0.5], offsets=[-1,1, Ny-1, -Ny+1])
    Dx /= dx
    Dy /= dy 
    Dy = Dy.T
    Dxx = diags([-2*np.ones(Nx), np.ones(Nx-1), np.ones(Nx-1), 1, 1], offsets = [0,1,-1, Nx-1,-Nx+1 ])
    Dxx /= dx**2
    Dyy = diags([-2*np.ones(Ny), np.ones(Ny-1), np.ones(Ny-1), 1, 1], offsets = [0,1,-1, Ny-1, -Ny+1])
    Dyy /= dy**2
    Dyy = Dyy.T                                                                               
    
    #-------------------Condiciones Iniciales----------
    rho = np.zeros([Nx, Ny])
    rho = 0.5/(2*np.pi*d**2)*(np.exp(-(X - 0.5 + D/2)**2/(2*d**2)) + np.exp(-(X - 0.5 - D/2)**2/(2*d**2))) * (np.exp(-(Y - 0.5 + D/2)**2/(2*d**2)) + np.exp(-(Y - 0.5 - D/2)**2/(2*d**2)))
    rho[:,0] = np.zeros(Nx)
    rho[0,:] = np.zeros(Ny)
   # rho2 = np.zeros([Nt, Nx, Ny])
   # rho2[0] = 0.5/(2*np.pi*d**2)*(np.exp(-(X - 0.5 + D/2)**2/(2*d**2)) + np.exp(-(X - 0.5 - D/2)**2/(2*d**2))) * (np.exp(-(Y - 0.5 + D/2)**2/(2*d**2)) + np.exp(-(Y - 0.5 - D/2)**2/(2*d**2)))
    #rho2[0,:,0] = np.zeros(Nx)
   # rho2[0,0,:] = np.zeros(Ny)
    #-------------------Evolucion Temporal-------------
    for n in range(Nt-1):
        K1 = 1j/(2*m)*(Dxx*rho - Dyy*rho) - gamma*(X - Y)*(Dx*rho - Dy*rho) - 2*m*gamma*T*(X - Y)**2*rho
        K2 = 1j/(2*m)*(Dxx*(rho + K1*dt/2) - Dyy*(rho + K1*dt/2)) - gamma*(X - Y)*(Dx*(rho + K1*dt/2) - Dy*(rho + K1*dt/2)) - 2*m*gamma*T*(X - Y)**2*(rho + K1*dt/2)
        K3 = 1j/(2*m)*(Dxx*(rho + K2*dt/2) - Dyy*(rho + K2*dt/2)) - gamma*(X - Y)*(Dx*(rho + K2*dt/2) - Dy*(rho + K2*dt/2)) - 2*m*gamma*T*(X - Y)**2*(rho + K2*dt/2)
        K4 = 1j/(2*m)*(Dxx*(rho + K3*dt) - Dyy*(rho + K3*dt))- gamma*(X - Y)*(Dx*(rho + K3*dt) - Dy*(rho + K3*dt)) - 2*m*gamma*T*(X - Y)**2*(rho + K3*dt)
        rho = rho + 1/6 * dt * (K1 + 2*K2 + 2*K3 + K4)
       # K12 = 1j/(2*m)*(Dxx*rho2[n] - Dyy*rho2[n]) - gamma*(X - Y)*(Dx*rho2[n] - Dy*rho2[n]) - 2*m*gamma*T*(X - Y)**2*rho2[n]
       # rho2[n+1] = rho2[n] + dt*K12
    if dt < 5e-2:
        err.append(np.max(np.abs(rho - rho_prev)))
    
    rho_prev  = rho  

plt.figure()
plt.plot(dts[1:], err, '.')
plt.xlabel("dt", fontsize = 12)
plt.ylabel("err", fontsize = 12)
plt.xticks(fontsize = 11)
plt.yticks(fontsize = 11)
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()    
""""  
    #-------------------Plot---------------------------


err = np.zeros(Nt)
for i in range(Nt):
    err[i] = np.max(np.abs((rho[i]-rho2[i])))/np.max(rho[n])
t = np.linspace(0,tf, Nt)
plt.figure()
plt.plot(t, err, linewidth = 2)
plt.grid()
plt.xlabel('Tiempo', fontsize = 14)
plt.ylabel('Error', fontsize = 14)
plt.show()
plt.figure()
ax = plt.axes(projection ='3d')
 
# syntax for plotting
ax.plot_surface(X, Y, (rho[0]), cmap ='viridis', edgecolor ='green')
ax.set_title(r'$\rho(x,y,t=0)$')
plt.show()

plt.figure()
ax = plt.axes(projection ='3d')
 
# syntax for plotting
ax.plot_surface(X, Y, (rho[int(Nt/4)]), cmap ='viridis', edgecolor ='green')
ax.set_title(r'$\rho(x,y,t=5)$')
plt.show()

plt.figure()
ax = plt.axes(projection ='3d')

# syntax for plotting
ax.plot_surface(X, Y, (rho[int(Nt/2)]), cmap ='viridis', edgecolor ='green')
ax.set_title(r'$\rho(x,y,t=10)$')
plt.show()


plt.figure()
ax = plt.axes(projection ='3d')
 
# syntax for plotting
ax.plot_surface(X, Y, (rho[Nt-1]), cmap ='viridis', edgecolor ='green')
ax.set_title(r'$\rho(x,y,t=20)$')
plt.show()
"""