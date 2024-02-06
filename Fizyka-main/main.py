import numpy as np
from scipy.integrate import odeint
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Define the derivative function
def calculate_derivative(state_vector, time, spring_constant, mass, left_boundary_force=None, right_boundary_force=None):
    total_points = state_vector.size // 2
    deriv = np.zeros_like(state_vector)

    # Apply boundary conditions if specified
    deriv[0] = left_boundary_force(state_vector, time) if left_boundary_force else 0
    deriv[total_points - 1] = right_boundary_force(state_vector, time) if right_boundary_force else 0

    # Calculate derivatives for internal points
    deriv[1:total_points - 1] = (state_vector[total_points + 2:2 * total_points] + state_vector[total_points:2 * total_points - 2] - 2 * state_vector[total_points + 1:2 * total_points - 1]) * spring_constant / mass
    deriv[total_points:2 * total_points] = state_vector[:total_points]

    return deriv

# Parameters
spring_constant = 25
mass = 0.25
total_points = 100

# Initial conditions
initial_condition = np.zeros(2 * total_points)

# Simulation parameters
dt = 0.001
max_time = 20
fps = 30
speedup = 1
frame_select = int(speedup / (fps * dt))

# Time array
times = np.arange(0, max_time, dt)

# Driving force parameters
frequency = 0.2
omega = 2 * np.pi * frequency
Amplitude = 0.5

def left_boundary_force(y, t):
    half_points = y.size // 2
    return -omega**2 * y[half_points]

initial_condition[0] = omega * Amplitude

# Integrate the system
result = odeint(calculate_derivative, initial_condition, times, args=(spring_constant, mass, left_boundary_force, None))
result = result[::frame_select, :]

# Setup for Plotly animation
fig = make_subplots(rows=1, cols=1)

# Initial plot setup
fig.add_trace(go.Scatter(x=np.linspace(0, 1, total_points), y=initial_condition[total_points:], mode='lines', line=dict(color='red')))
fig.add_trace(go.Scatter(x=[0], y=[initial_condition[total_points]], mode='markers', marker=dict(color='green', size=10)))

fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False,
                  xaxis_zeroline=False, yaxis_zeroline=False,
                  xaxis_visible=False, yaxis_visible=False,
                  margin=dict(l=0, r=0, t=0, b=0))

# Adding frames for animation
frames = [go.Frame(data=[go.Scatter(x=np.linspace(0, 1, total_points), y=result[i, total_points:]),
                         go.Scatter(x=[0], y=[result[i, total_points]])])
          for i in range(len(result))]

fig.frames = frames

# Adding Play Button only
fig.update_layout(updatemenus=[dict(type="buttons",
                                    buttons=[dict(label="Play",
                                                  method="animate",
                                                  args=[None, dict(frame=dict(duration=1000/fps, redraw=True),
                                                                   fromcurrent=True, mode='immediate')])],
                                    direction="left",
                                    pad={"r": 10, "t": 87},
                                    showactive=False,
                                    x=0.1,
                                    xanchor="right",
                                    y=0,
                                    yanchor="top")])

# Display the animation
fig.show()