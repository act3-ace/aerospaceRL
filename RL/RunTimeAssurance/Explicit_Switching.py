import numpy as np


class RTA():
	def __init__(self, env):
		self.m = env.mass_deputy
		self.n = env.n
		self.u_max = env.force_magnitude
		self.dt = env.tau
		self.nu1 = 2*self.n
		self.nu0 = 0.2

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

		if self.h(x1) >= 0:
			return u_des
		else:
			return self.u_b(x0, x1)


	def h(self, x0):
		x = x0[0]
		y = x0[1]
		xd = x0[3]
		yd = x0[4]
		h = self.nu0 + self.nu1*np.linalg.norm([x,y]) - np.linalg.norm([xd,yd])
		return h

	def u_b(self, x0, x1):
		x0 = np.vstack(x0)
		rH = np.linalg.norm(x1[0:3])
		vH = np.linalg.norm(x1[3:6])
		vH_max = self.nu1 * rH + self.nu0
		x1[3:6] = x1[3:6]/vH*vH_max

		acc = (x1[3:6] - x0[3:6]) / self.dt
		u = (acc[0:3]-self.A[3:6]@x1)*self.m

		return u
