import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
robot = p.loadURDF("robot.urdf", useFixedBase=True)

num_joints = p.getNumJoints(robot)
print(f"Loaded robot with {num_joints} joints:")
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    print(f"  Joint {i}: {info[1].decode()} ({['fixed','revolute','prismatic'][info[2]]})")

# Add sliders for each non-fixed joint
sliders = {}
for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    if info[2] in [0, 1]:  # revolute or prismatic
        name = info[1].decode()
        sliders[i] = p.addUserDebugParameter(name, -3.14, 3.14, 0)

while True:
    for joint_id, slider_id in sliders.items():
        val = p.readUserDebugParameter(slider_id)
        p.setJointMotorControl2(robot, joint_id,
                                p.POSITION_CONTROL,
                                targetPosition=val)
    p.stepSimulation()
    time.sleep(1/240)