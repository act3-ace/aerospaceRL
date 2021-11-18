import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,'font.size': 18, 'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']})
plt.rcParams.update({'figure.autolayout': True})
from scipy import integrate

def x_dot(t,x):
	n = 0.001027
	m = 12
	A = np.array([
		[0, 0, 1, 0],
		[0, 0, 0, 1],
		[3 * n ** 2, 0, 0, 2 * n],
		[0, 0, -2 * n, 0],
	], dtype=np.float64)

	return np.matmul(A,x)


class Parameters_2D():
	def __init__(self):
		self.n = 0.001027
		self.m = 12
		self.dt = 1
		self.max_t = 5000
		self.u_max = 1.
		self.nu0 = 0
		self.nu1 = 2*self.n
		self.max_vel = 10

		self.A = np.array([
			[0, 0, 1, 0],
			[0, 0, 0, 1],
			[3 * self.n ** 2, 0, 0, 2 * self.n],
			[0, 0, -2 * self.n, 0],
		], dtype=np.float64)

		self.B = np.array([
			[0, 0],
			[0, 0],
			[1 / self.m, 0],
			[0, 1 / self.m],
		], dtype=np.float64)

env = Parameters_2D()
for i in range(10):
	x = np.array([(i+1)/2,0,0,-2*env.n*(i+1)/2])
	xs = [x[0]]
	ys = [x[1]]
	rH = [np.linalg.norm(x[0:2])]
	vH = [np.linalg.norm(x[2:4])*1000]
	for t in range(7500):
		xd = integrate.solve_ivp(x_dot,[0,10],x)
		x = [xd.y[0][-1],xd.y[1][-1],xd.y[2][-1],xd.y[3][-1]]
		xs.append(x[0])
		ys.append(x[1])
		rH.append(np.linalg.norm(x[0:2]))
		vH.append(np.linalg.norm(x[2:4])*1000)

	plt.figure(1)
	plt.plot(rH,vH)
	plt.figure(2)
	plt.plot(xs,ys)


plt.figure(1)
plt.fill_between([0,15],0,[env.nu0,env.nu0+env.nu1*15000],color=(244/255,249/255,241/255))
plt.fill_between([0,15],[env.nu0,env.nu0+env.nu1*15000],[1000,1000],color=(255/255,239/255,239/255))
plt.plot([0,15],[env.nu0,env.nu0+env.nu1*15000],'k--',linewidth=2)
plt.xlim([0, 10])
plt.ylim([0, 12])
plt.xticks([])
plt.yticks([])
plt.xlabel('Relative Position ($\Vert \pmb{r}_{\mathrm{H}} \Vert$)')
plt.ylabel('Relative Velocity ($\Vert \pmb{v}_{\mathrm{H}} \Vert$)')
plt.title('Distance Dependent Speed Limit')
plt.grid(True)

plt.figure(2)
plt.title('Closed Elliptical NMTs')
plt.xlim([-11, 11])
plt.ylim([-11, 11])
plt.xticks([])
plt.yticks([])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.grid(True)


fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(0,0,'k*', markersize=10)
c1 = plt.Circle((0, 0), 1000, fill=False, color='r')
ax1.add_patch(c1)
ax1.set_xlabel('$x$ [m]')
ax1.set_ylabel('$y$ [m]')
ax1.grid(True)
ax1.set_aspect('equal')
ax1.set_title('Training Initial Conditions')
ax1.set_xticks([-1000,-500,0,500,1000])
ax1.set_yticks([-1000,-500,0,500,1000])

fig = plt.figure()
ax1 = fig.add_subplot(111)
plt.plot(0,0,'k*', markersize=10)

Distance = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
Angle = [1.57, 5.5, 4.71, 2.36, 3.93, 0.79, 1.18, 3.14, 4.32, 0]
for i in range(len(Distance)):
	x = Distance[i]*np.cos(Angle[i])
	y = Distance[i]*np.sin(Angle[i])
	ax1.plot(x,y,'r*')

ax1.set_xlabel('$x$ [m]')
ax1.set_ylabel('$y$ [m]')
ax1.grid(True)
ax1.set_aspect('equal')
ax1.set_title('10 Test Points')
ax1.set_xticks([-1000,-500,0,500,1000])
ax1.set_yticks([-1000,-500,0,500,1000])

plt.show()
