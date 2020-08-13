from gym.envs.registration import register

register(
    id='dubins-aircraft-v0',
    entry_point='aero_gym.envs:DubinsAircraft',
)

register(
    id='dubins-aircraft-continuous-v0',
    entry_point='aero_gym.envs:DubinsAircraftContinuous',
)

register(
    id='spacecraft-docking-v0',
    entry_point='aero_gym.envs:SpacecraftDocking',
)

register(
    id='spacecraft-docking-continuous-v0',
    entry_point='aero_gym.envs:SpacecraftDockingContinuous',
)
