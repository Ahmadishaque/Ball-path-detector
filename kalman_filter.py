import numpy as np

class KalmanFilter:
    def __init__(self, dt=1, u_noise=1, m_noise=10):
        self.dt = dt
        
        self.x = np.zeros((4, 1))  # [x, y, vx, vy]
        
        self.P = np.eye(4) * 1000
        
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        
        self.R = np.eye(2) * m_noise
        
        self.Q = np.array([[1/4*dt**4, 0, 1/2*dt**3, 0],
                           [0, 1/4*dt**4, 0, 1/2*dt**3],
                           [1/2*dt**3, 0, dt**2, 0],
                           [0, 1/2*dt**3, 0, dt**2]]) * u_noise

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0, 0], self.x[1, 0]

    def update(self, z):
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
