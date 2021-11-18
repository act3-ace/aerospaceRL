import numpy as np
import quadprog


class RTA():
	def __init__(self, env):
		self.m = env.mass_deputy
		self.n = env.n
		self.u_max = env.force_magnitude
		self.dt = env.tau
		self.nu1 = 2*self.n
		self.nu0 = 0.2
		self.T_backup = 5
		self.kappa = 0.6
		self.Nsteps = int(self.T_backup/self.dt) + 1
		self.Nskip = 1
		self.N_checkall = 5

		self.A = np.array([
			[0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 1],
			[3 * self.n ** 2, 0, 0, 0, 2 * self.n, 0],
			[0, 0, 0, -2 * self.n, 0, 0],
			[0, 0, -self.n ** 2, 0, 0, 0],
		], dtype=np.float64)

		self.B = np.array([
			[0, 0, 0],
			[0, 0, 0],
			[0, 0, 0],
			[1 / self.m, 0, 0],
			[0, 1 / self.m, 0],
			[0, 0, 1 / self.m],
		], dtype=np.float64)

	def main(self, x0, u_des):
		phi, S = self.integrate(x0)

		M = np.eye(3)
		q = np.reshape(u_des,3)
		# Constraints:
		self.G = [[1., 0., 0.],[-1., 0., 0.],[0., 1., 0.],[0., -1., 0.],[0., 0., 1.],[0., 0., -1.]]
		self.h_a = [-self.u_max, -self.u_max, -self.u_max, -self.u_max, -self.u_max, -self.u_max]

		self.Ax0 = self.A @ x0

		# Set invariance constraints
		for i in range(0, self.N_checkall):
			self.invariance_constraints(i, phi, S)

		for i in range(self.N_checkall+self.Nskip, self.Nsteps, self.Nskip):
			self.invariance_constraints(i, phi, S)

		# Solve optimization program!
		u_act = quadprog.solve_qp(M, q, np.array(self.G).T, np.array(self.h_a), 0)[0]
		u = np.reshape(u_act,(3,1))

		return np.reshape(u_act,(3,1))

	def h(self, x0):
		x = x0[0]
		y = x0[1]
		xd = x0[3]
		yd = x0[4]
		h = self.nu0 + self.nu1*np.linalg.norm([x,y]) - np.linalg.norm([xd,yd])
		return h

	def alpha(self, x):
		return 0.1*x + 0.1*x**3

	def grad(self,x):
		Hs = np.array([[2*self.nu1**2, 0, 0, 0, 0, 0],
					   [0, 2*self.nu1**2, 0, 0, 0, 0],
					   [0, 0, 2*self.nu1**2, 0, 0, 0],
					   [0, 0, 0, -2, 0, 0],
					   [0, 0, 0, 0, -2, 0],
					   [0, 0, 0, 0, 0, -2]])

		ghs = Hs @ x
		ghs[0] = ghs[0] + 2*self.nu1*self.nu0*x[0]/np.linalg.norm([x[0],x[1],x[2]])
		ghs[1] = ghs[1] + 2*self.nu1*self.nu0*x[1]/np.linalg.norm([x[0],x[1],x[2]])
		ghs[2] = ghs[2] + 2*self.nu1*self.nu0*x[2]/np.linalg.norm([x[0],x[1],x[2]])
		return ghs

	def integrate(self, x):
		phi = np.zeros([6, self.Nsteps])
		S = np.zeros([6, 6, self.Nsteps])
		phi[:, 0] = np.reshape(x,6)
		S[:, :, 0] = np.eye(6)

		for i in range(1, self.Nsteps):
			# Dynamics
			x = np.reshape(phi[:, i-1],(6,1))
			u_des = self.u_b(x)
			x1 = x + (self.A @ x + self.B @ u_des)*self.dt
			phi[:, i] = np.reshape(x1,6)
			# # Sensitivity
			Dphi = self.get_Jacobian(phi[:, i])
			S[:, :, i] = S[:, :, i-1] + (Dphi @ S[:, :, i-1])*self.dt

		return phi, S

	def invariance_constraints(self, i, phi, S):
		d = self.grad(phi[:, i]) @ (S[:, :, i] @ self.Ax0) + self.alpha(self.h(phi[:, i]))
		c =  self.grad(phi[:, i]) @ (S[:, :, i] @ self.B)
		self.G.append([c[0], c[1], c[2]])
		self.h_a.append(-d)

	def u_b(self, x0):
		x = x0[0,0]
		y = x0[1,0]
		xdot = x0[3,0]
		ydot = x0[4,0]

		e1 = xdot-0.5*self.n*y
		e2 = ydot+2*self.n*x

		u = np.array([[self.u_max*np.tanh((-1.5*self.n*e2 - self.kappa*e1)*(self.m/self.u_max))],
					  [self.u_max*np.tanh((-self.kappa*e2)*(self.m/self.u_max))],
					  [0]])

		return u

	def get_Jacobian(self, phi):
		e1 = phi[3]-(self.n/2)*phi[1]
		e2 = phi[4]+(2*self.n)*phi[0]

		# Initialize control contribution to Jacobian
		D_control = np.zeros([6, 6])

		# Control terms without "tanh()" component
		ubx = (-1.5*self.n*e2 - self.kappa*e1)*(self.m/self.u_max)
		uby = (-self.kappa*e2)*(self.m/self.u_max)

		D_control[3, 0] = -3*self.n**2*(1-(np.tanh(ubx*(self.m/self.u_max)))**2)
		D_control[3, 1] = 0.5*self.n*(1-(np.tanh(ubx*(self.m/self.u_max)))**2)
		D_control[3, 3] = -self.kappa*(1-(np.tanh(ubx*(self.m/self.u_max)))**2)
		D_control[3, 4] = -1.5*self.n*(1-(np.tanh(ubx*(self.m/self.u_max)))**2)

		D_control[4, 0] = -2*self.n*self.kappa*(1-(np.tanh(uby*(self.m/self.u_max)))**2)
		D_control[4, 4] = -self.kappa*(1-(np.tanh(uby*(self.m/self.u_max)))**2)

		return (self.A + D_control)
