import numpy as np
import math

V0 = 1.3
TAU = 0.5
BETA = 8.

def time_to_collision_with_segment(position_robot, velocity_robot, radius,p1, p2):
    pr = np.array(position_robot)
    vr = np.array(velocity_robot)
    
    # Define the line segment points
    p1 = np.array(p1)
    p2 = np.array(p2)
    
    # Direction of the line segment
    segment_direction = p2 - p1
    
    # Find the normal vector to the segment
    normal_vector = np.array([segment_direction[1], -segment_direction[0]])
    normal_vector = normal_vector/ np.linalg.norm(normal_vector)
    
    # Project the robot's velocity onto the normal of the wall (dot product)
    velocity_normal = np.dot(vr, normal_vector) 
    
    
    # Check if the robot is moving towards the wall (velocity_normal < 0)
    if velocity_normal >= 0:
        return float('inf')  # No collision
    
    # Distance from the human to the infinite line of the wall
    distance_to_line = np.abs(np.cross(p2 - p1, pr - p1)) / np.linalg.norm(segment_direction)
    
    # Time to collision with the infinite line
    t_collision = (distance_to_line - radius) / -velocity_normal
    return t_collision
    # # Now check if the intersection point lies within the segment bounds
    # intersection_point = pr + t_collision * vr
    # print(intersection_point)
    
    # # Check if the intersection point is within the line segment bounds
    # if (min(p1[0], p2[0]) <= intersection_point[0] <= max(p1[0], p2[0]) and
    #     min(p1[1], p2[1]) <= intersection_point[1] <= max(p1[1], p2[1])):
    #     return t_collision
    # else:
    #     return float('inf')  # No collision with the finite segment

def solve_quadratic(A, B, C):
    # Calculate the discriminant
    discriminant = B**2 - 4*A*C
    
    # Check if the discriminant is negative (complex roots)
    if discriminant < 0:
        return None, None  # No real roots
    
    # Compute the two roots using the quadratic formula
    root1 = (-B + math.sqrt(discriminant)) / (2*A)
    root2 = (-B - math.sqrt(discriminant)) / (2*A)
    
    # Return the minimum root
    return root1, root2

def time_to_collision_with_human(position_robot, 
                                 velocity_robot,
                                 r_robot,
                                 position_other,
                                 pred_velo_other,
                                 r_other):

    vxi, vyi = velocity_robot[0],velocity_robot[1]
    vxj, vyj = pred_velo_other
    pxi, pyi = position_robot
    pxj, pyj = position_other

    A = (vxj - vxi) ** 2 + (vyj - vyi) ** 2
    B = 2 * (vxj - vxi) * (pxj - pxi) + 2 * (vyj - vyi) * (pyj - pyi)
    C = (pxj - pxi) ** 2 + (pyj - pyi) ** 2 - (r_robot + r_other) ** 2

    if A == 0:
        if B == 0:
            return float('inf')  # No movement between the two objects
        else:
            # Linear case: solve Bx + C = 0 -> x = -C / B
            root = -C / B
            return root if root > 0 else float('inf')

    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        return float('inf')  # No real roots, so no collision

    root1 = (-B + np.sqrt(discriminant)) / (2 * A)
    root2 = (-B - np.sqrt(discriminant)) / (2 * A)

    if root1 > 0 and root2 > 0:
        return min(root1, root2)
    elif root1 > 0:
        return root1
    elif root2 > 0:
        return root2
    else:
        return float('inf')

def pred_other_steering(position_robot,position_other):

    GAMMA = 3.5
    IOTA = 37.49
    EPSILON = np.random.normal(0,3)
    robot_to_other = np.array(position_robot)-np.array(position_other)
    distance = np.linalg.norm(robot_to_other)
    direction_robot_to_other = robot_to_other/distance
    angle = np.arctan2(direction_robot_to_other[1],direction_robot_to_other[0])
    steering = -distance*GAMMA+IOTA+EPSILON
    steering = steering*np.pi/180.
    result_angle = angle-steering
    pred_other_speed_direction = np.array([np.cos(result_angle),np.sin(result_angle)])

    return pred_other_speed_direction

def cal_steering(position_robot,velo_robot,goal_robot,r_robot,others,walls,beta=BETA):
    
    da = np.array(goal_robot)-np.array(position_robot)
    da = np.arctan2(da[1],da[0])
    min_d_alpha = np.inf
    Eh = np.inf
    alpha_des = 0
    for alpha in np.arange(-90,90.1,3):
        self_steering_angle = alpha*np.pi/180.
        robot_theta = np.arctan2(velo_robot[1],velo_robot[0])+self_steering_angle
        velo_robot_steered = V0*np.array([np.cos(robot_theta),
                                            np.sin(robot_theta)])
        min_dist_to_other = np.inf
        min_dist_to_wall = np.inf
        # cal time to collide wall
        for wall in walls:
            p1, p2 = wall
            ttc_wall = time_to_collision_with_segment(position_robot, 
                                                      velo_robot_steered, 
                                                      r_robot,p1, p2)
            # print(p1,p2,ttc_wall)
            dist_to_wall = ttc_wall*V0
            if dist_to_wall<min_dist_to_wall:
                min_dist_to_wall = dist_to_wall

        # print(position_robot, velo_robot_steered)
        # print("d wall",min_dist_to_wall)
        # input()

        # cal time to collide other
        for position_other,velo_other,r_other in others:
            pred_other_speed_direction = pred_other_steering(position_robot,
                                                             position_other)
            # TODO check article
            pred_velo_other = velo_other#pred_other_speed_direction*np.linalg.norm(velo_other)
            ttc_human = time_to_collision_with_human(position_robot,velo_robot_steered,r_robot,
                                                     position_other,pred_velo_other,r_other)
            dist_to_human = ttc_human*V0
            # print("d human",dist_to_human)
            if dist_to_human<min_dist_to_other:
                min_dist_to_other = dist_to_human
        
        min_dist = min(min_dist_to_other,min_dist_to_wall)
        # input()
        d_alpha = beta**2+min_dist**2-2*beta*min_dist*np.cos(da-robot_theta)
        if d_alpha<min_d_alpha:
            min_d_alpha=d_alpha
            alpha_des = robot_theta
            Eh = min_dist

    return alpha_des,Eh

    
if __name__ == "__main__":
    from harl.envs.robot_crowd_sim.crowd_env import RobotCrowdSim
    import os
    import json
    from harl.common.video import VideoRecorder
    video_recorder = VideoRecorder(".")
    human_beta = [0.5,0.5,8,8,8]
    model_dir="/home/dl/wu_ws/HARL/results/crowd_env/crowd_navi/robot_crowd_happo/robust_navi/seed-00001-2024-09-17-21-16-39"
    config_file = os.path.join(model_dir,"config.json")
    with open(config_file, encoding="utf-8") as file:
        all_config = json.load(file)
    algo_args = all_config["algo_args"]
    env_args = all_config["env_args"]
    env_args["human_num"] = 2
    env = RobotCrowdSim(env_args,"test",1,0,"Att")
    x,y = env._map.getBoundary()
    walls = []
    for i in range(len(x)-1):
        walls.append(([x[i],y[i]],[x[i+1],y[i+1]]))

    env.reset(seed=-1)
    video_recorder.init(env)
    for i in range(50):
        actions = [np.array([-1,0])]
        for human in env.crowd:
            position_robot = human.get_position()
            velo_robot = human.get_velocity()
            goal_robot = human.get_goal_position()
            r_robot = human.radius
            others = [(env.robot.get_position(),
                    env.robot.get_velocity(),
                    env.robot.radius)]
            for other in env.crowd:
                if other.id == human.id:continue
                others.append((
                    other.get_position(),
                    other.get_velocity(),
                    other.radius
                ))
            alpha_des,Eh = cal_steering(position_robot,
                                        velo_robot,
                                        goal_robot,
                                        r_robot,
                                        others,walls,human_beta[human.id-1])  
            speed_des = min(V0,Eh/TAU) 
            action = speed_des*np.array([np.cos(alpha_des),
                                        np.sin(alpha_des)])
            action = velo_robot + (action-velo_robot)/TAU*0.25
            actions.append(action)

        action = np.array(actions)
        # print(action)
        # input()
        env.step(action)
        video_recorder.record(env)

    video_recorder.save("test.mp4")
        
