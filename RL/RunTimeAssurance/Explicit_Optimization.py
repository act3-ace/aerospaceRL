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
		M = np.eye(3)
		q = np.reshape(u_des,3)
		G = [[1., 0., 0.],[-1., 0., 0.],[0., 1., 0.],[0., -1., 0.],[0., 0., 1.],[0., 0., -1.]]
		h = [-self.u_max, -self.u_max, -self.u_max, -self.u_max, -self.u_max, -self.u_max]

		g_temp = np.reshape(self.grad(x0),6) @ self.B
		h_temp = -np.reshape(self.grad(x0),6) @ self.A @ x0 - self.alpha(self.h(x0))

		G.append([g_temp[0],g_temp[1],g_temp[2]])
		h.append(h_temp)

		# Solve optimization program!
		u_act = quadprog.solve_qp(M, q, np.array(G).T, np.array(h), 0)[0]
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
