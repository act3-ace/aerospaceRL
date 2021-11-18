import numpy as np


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
		x0 = np.vstack(x0)
		x1 = x0 + (self.A @ x0 + self.B @ u_des) * self.dt

		phi = self.integrate(x1)

		for i in range(0, self.Nsteps):
			if self.h(x1) < 0:
				return self.u_b(x0)

		return u_des


	def h(self, x0):
		x = x0[0]
		y = x0[1]
		xd = x0[3]
		yd = x0[4]
		h = self.nu0 + self.nu1*np.linalg.norm([x,y]) - np.linalg.norm([xd,yd])
		return h

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

	def integrate(self, x):
		phi = np.zeros([6, self.Nsteps])

		phi[:, 0] = np.reshape(x,6)
		for i in range(1, self.Nsteps):
			# Dynamics
			x = np.reshape(phi[:, i-1],(6,1))
			u_des = self.u_b(x)
			x1 = x + (self.A @ x + self.B @ u_des)*self.dt
			phi[:, i] = np.reshape(x1,6)

		return phi
