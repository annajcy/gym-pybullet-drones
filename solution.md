# Q1 

See the video attachment.

# Q2 

### File Locations

For observation state refer to: 
`def _observationSpace(self)` in `gym-pybullet-drones/gym_pybullet_drones/envs/BaseRLAviary.py`

For action state refer to:
`def _actionSpace(self)` in `gym-pybullet-drones/gym_pybullet_drones/envs/BaseRLAviary.py`

For reward function refer to:
`def _computeReward(self)` in `gym-pybullet-drones/gym_pybullet_drones/envs/HoverAviary.py`

### Observation State 

- **Position** (3D): `obs[0:3]` → (x, y, z) coordinates in world frame
- **Euler Angles** (3D): `obs[7:10]` → (roll, pitch, yaw) orientation
- **Linear Velocity** (3D): `obs[10:13]` → (vx, vy, vz) velocity
- **Angular Velocity** (3D): `obs[13:16]` → (wx, wy, wz) angular velocity

### Action State

- Dimensions: 4D action vector $\mathbf{a} = [a_1, a_2, a_3, a_4]^T$
- Range: Each $a_i \in [-1, +1]$
- Physical meaning: RPM adjustment coefficients for four propellers
- Action Processing
  $$\text{RPM}_i = \text{HOVER\_RPM} \times (1 + 0.05 \times a_i)$$

  When $a_i = -1$: $\text{RPM}_i = 0.95 \times \text{HOVER\_RPM}$ (95% of hover RPM)

  When $a_i = 0$: $\text{RPM}_i = \text{HOVER\_RPM}$ (baseline hover RPM)  

  When $a_i = +1$: $\text{RPM}_i = 1.05 \times \text{HOVER\_RPM}$ (105% of hover RPM)

  This allows fine-grained control with $\pm 5\%$ adjustment around the stable hovering RPM.

### Reward Function

$$r_t = \max(0, 2 - \|\mathbf{p}_{target} - \mathbf{p}_{current}\|^4)$$

Where:
- $\mathbf{p}_{target} = [0, 0, 1]^T$ (hovering target)
- $\mathbf{p}_{current} = \text{state}[0:3]$ (drone's current position)

#### Reward Characteristics

1. **Maximum reward**: $2.0$ (when drone is exactly at target position)
2. **Distance penalty**: Uses fourth power ($\cdot^4$) for aggressive penalization
3. **Zero reward threshold**: When distance $> 2^{1/4} \approx 1.19$ meters
4. **Reward range**: $[0, 2]$

#### Reward Behavior
- **At target** (distance $= 0$): $r_t = 2.0$
- **Close to target** (distance $= 0.5$): $r_t = 2 - (0.5)^4 = 2 - 0.0625 = 1.9375$
- **Moderate distance** (distance $= 1.0$): $r_t = 2 - (1.0)^4 = 2 - 1 = 1.0$  
- **Far from target** (distance $= 1.2$): $r_t = \max(0, 2 - (1.2)^4) = \max(0, 2 - 2.07) = 0$

# Q3

