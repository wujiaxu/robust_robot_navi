import matplotlib.pyplot as plt
from matplotlib import collections as mc
import matplotlib.lines as mlines
from matplotlib import patches
from matplotlib import colors as mcolors

from typing import List
import numpy as np
from harl.envs.robot_crowd_sim.utils.agent import Agent
from harl.envs.robot_crowd_sim.utils.map import Map
class Render:
    def __init__(self,map,agents):

        self._map:Map = map
        self._agents:List[Agent] = agents

        fig,self.render_axis = plt.subplots(figsize=(8,8)) 
        size = max(self._map._map_width,self._map._map_hight)
        # fig.patch.set_alpha(0.0)  # Figure background transparency
        # self.render_axis.set_facecolor((0, 0, 0, 0))  # Axes background transparency
        #(0, 0) is the bottom-left corner and (1, 1) is the top-right corner of the axis.
        # self.text = self.render_axis.text(-3.9, -3.9, 'v:{}[m/s]'.format(0.), fontsize=14, color='black')
        self.crowd_pref_text = self.render_axis.text(0,
                                                     size/2.+0.3, 
                                                     'crowd_pref:{}'.format(-1), fontsize=14, color='black',
                                                     verticalalignment="center", horizontalalignment="center",)
        # self.crowd_pref_text.set_visible(False)
        self.robot_log_prob_text = self.render_axis.text(-size/2.+0.1,
                                                     size/2.-0.6, 
                                                     'robot input prob:{}'.format(-1), fontsize=14, color='black')   
        self.robot_log_prob_text.set_visible(False)
        plt.xlim(-size/2.-1,size/2+1)
        plt.ylim(-size/2.-1,size/2+1)

        self.current_scans = {}
        self.attention_values = {}
        self.robot_rec = False
        # Get all colors
        # colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

        # Sort colors by value (V), then by saturation (S), and finally by hue (H)
        # by_hsv = sorted(colors.items(), key=lambda item: (
        #     mcolors.rgb_to_hsv(mcolors.to_rgba(item[1])[:3])[2],  # Sort by value (V) first
        #     mcolors.rgb_to_hsv(mcolors.to_rgba(item[1])[:3])[1],  # Then by saturation (S)
        #     mcolors.rgb_to_hsv(mcolors.to_rgba(item[1])[:3])[0]   # Finally by hue (H)
        # ))
        # self.human_colors = [name for hsv, name in by_hsv] #yellow is robot color
        # self.robot_color = 'yellow'
        self.agent_colors = [
            'blue', 'green', 'cyan', 'magenta', 'yellow', 
            'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'lime', 'aquamarine',
            'beige', 'bisque', 'coral', 'crimson', 'gold', 'indigo', 'khaki', 'lavender',
            'navy', 'salmon', 'tan', 'teal', 'aqua', 'maroon', 'chartreuse', 'turquoise',
            # 'black'
        ]
        self.agent_colors_alpha = [1,0.8,0.6,0.4]

        self.trajs = {}
        self.traj_data = {}
        self.agent_id_text = {}
        self.agent_circle = {}
        self.agent_goal = {}
        self.agent_goal_text = {}
        for agent in self._agents:
            self.trajs[agent.id] =  mlines.Line2D([], [], color="black")
            self.render_axis.add_line(self.trajs[agent.id])
            self.agent_id_text[agent.id] = self.render_axis.text(0,0,str(agent.id), fontsize=10, color='black', verticalalignment="center", horizontalalignment="center",visible=False)
            self.agent_circle[agent.id] = patches.Circle((0,0), 0.3, fill=True, color="black",alpha=1,visible=False)
            self.render_axis.add_artist(self.agent_circle[agent.id])
            self.agent_goal[agent.id] = mlines.Line2D(xdata=[0], ydata=[0], color="black", marker='*', linestyle='None', markersize=15,alpha=1,visible=False)
            self.render_axis.add_line(self.agent_goal[agent.id])
            self.agent_goal_text[agent.id] = self.render_axis.text(0,0,str(agent.id), fontsize=10, color='black', verticalalignment="center", horizontalalignment="center",visible=False)
        self._distracted_human = []

    def reset(self,distracted_humans=[]):
        self.traj_data = {}
        for agent in self._agents:
            self.trajs[agent.id].set_data([],[])
            self.trajs[agent.id].set_color("black")
        self._distracted_humans = distracted_humans
        return
    
    def add_scan_data(self,agent_id, scan_data, attention_data):
        self.current_scans[agent_id] = scan_data
        self.attention_values[agent_id] = attention_data
        return 

    def rend(self,mode,time=None,preference=None,log_probs=None): #TODO: add timer
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        artists = []
        velo_text = ''

        # map
        x_b,y_b = self._map.getBoundary()
        x_b,y_b = list(x_b),list(y_b)
        boundary = [(x_b[i],y_b[i]) for i in range(len(x_b))]
        polygon = patches.Polygon(boundary[:-1], closed=True, edgecolor='b', facecolor='none')
        self.render_axis.add_patch(polygon)
        artists.append(polygon)

        # agent_disks = []
        agent_labels = ["agent","goal"]
        agent_legends = [mlines.Line2D([1], [1], marker='o', color='black', fillstyle="none", markersize=15,linestyle='None'),
                         mlines.Line2D(xdata=[0], ydata=[0], color="black", marker='*', linestyle='None',markersize=15)]
        preference_legends = []
        preference_label = []
        if preference is not None:
            pref_type = preference["meta"]
        else:
            pref_type = None
        if self.robot_rec == True:
            agent_labels.append("using ad-hoc policy")
            agent_legends.append(mlines.Line2D([1], [1], marker='o', color='red',alpha=0.3, fillstyle="full",linestyle='None',markersize=15))

        # add crowd
        for i,agent in enumerate(self._agents):
            if preference is not None and agent.agent_type=="human":
                if pref_type == "category":
                    control_code = int(np.argmax(preference[agent.id]))
                    pref_text = "z={},".format(control_code)
                    if pref_text not in preference_label:
                        preference_label.append(pref_text)
                        preference_legends.append(mlines.Line2D([1], [1], marker='o', color=self.agent_colors[control_code],fillstyle="none", markersize=15,linestyle='None'))
                    agent_color = self.agent_colors[control_code]
                elif pref_type == "ccp":
                    pref_text = "{},".format(
                        str(round(preference[agent.id][0],1))+"-"+str(round(preference[agent.id][1],1)))
                    preference_label.append(pref_text)
                    preference_legends.append(mlines.Line2D([1], [1], marker='o', color=self.agent_colors[agent.id],fillstyle="none", markersize=15,linestyle='None'))
                    agent_color = self.agent_colors[agent.id]
                    # pref_text = "{},".format(
                    #     str(round(preference[key][0],1))+"-"+str(round(preference[key][1],1))
                    #     )
                    # if pref_text not in preference_label:
                    #     preference_label.append(pref_text)
            else:
                agent_color = "black"#self.agent_colors[agent.id%len(self.agent_colors)]
            # alpha=self.agent_colors_alpha[agent.id//len(self.agent_colors)]
            # goal=mlines.Line2D([agent.gx], [agent.gy], color=agent_color, marker='*', linestyle='None', markersize=15,alpha=alpha)
            # agent_goals.append(goal)
            # agent_goal_labels.append("{}_goal".format(agent.id))
            # self.render_axis.add_artist(goal)
            # artists.append(goal)    
            # add agent
            if agent.agent_type == "robot" and self.robot_rec == True:
                self.agent_circle[agent.id].set(center=agent.get_position(), radius=agent.radius, fill=True, color="red",alpha=0.3,visible=True)
            # elif agent.agent_type=="robot" and self.robot_rec == False:
            #     self.agent_circle[agent.id].set(center=agent.get_position(), radius=agent.radius, fill=False, color=agent_color, visible=True)
            else:
                self.agent_circle[agent.id].set(center=agent.get_position(), radius=agent.radius, fill=False, color=agent_color, visible=True)
            self.agent_goal[agent.id].set(xdata=[agent.gx], ydata=[agent.gy], color=agent_color, visible=True)
            
            if agent.agent_type == "robot":
                self.agent_id_text[agent.id].set(position=agent.get_position(),text="R"+str(agent.id),visible=True)
                self.agent_goal_text[agent.id].set(position=(agent.gx+0.1,agent.gy+0.1),text="G"+str(agent.id) ,visible=True)
            # elif agent.agent_type == "robot" and self.robot_rec == False:
            #     self.agent_id_text[agent.id].set(position=agent.get_position(),text="R"+str(agent.id),visible=True)
            #     self.agent_goal_text[agent.id].set(position=agent.get_goal_position(),text="R"+str(agent.id) ,visible=True)
            elif agent.id in self._distracted_humans:
                # agent_labels.append("{}_{}".format(i,"distracted_human"))
                self.agent_id_text[agent.id].set(position=agent.get_position(),text="DH"+str(agent.id),visible=True)
                self.agent_goal_text[agent.id].set(position=(agent.gx+0.1,agent.gy+0.1),text="G"+str(agent.id) ,visible=True)
            else:
                # agent_labels.append("{}_{}".format(i,agent.agent_type))
                self.agent_id_text[agent.id].set(position=agent.get_position(),text="H"+str(agent.id),visible=True)
                self.agent_goal_text[agent.id].set(position=(agent.gx+0.1,agent.gy+0.1),text="G"+str(agent.id),visible=True)

            if agent.id not in self.traj_data.keys():
                self.traj_data[agent.id] = []
            self.traj_data[agent.id].append(np.array(agent.get_position()))

            # render traj
            current_traj = np.array(self.traj_data[agent.id])
            self.trajs[agent.id].set_data(current_traj[:,0].tolist(),
                    current_traj[:,1].tolist())
            self.trajs[agent.id].set_color(agent_color)
            if agent.px == 999 and agent.py==999:
                self.trajs[agent.id].set_visible(False)
            else:
                self.trajs[agent.id].set_visible(True)
            
            # render scan
            if agent.id in self.current_scans.keys():
                ii = 0
                lines = []
                colors = []
                while ii < len(self.current_scans[agent.id]):
                    # if self.attention_values[agent.id][ii] == 0 and agent.agent_type == "human":
                    #     ii = ii + 18
                    #     continue
                    if self.attention_values[agent.id][ii] == 1:
                        colors.append("r")
                    else:
                        colors.append(agent_color)
                    lines.append(self.current_scans[agent.id][ii])
                    ii = ii + 18
                    
                lc = mc.LineCollection(lines,linewidths=1,linestyles='--',alpha=0.7,color=colors)
                self.render_axis.add_artist(lc)
                artists.append(lc)


        plt.legend(agent_legends+preference_legends, 
                   agent_labels+preference_label, 
                   bbox_to_anchor=(-0.1, 1.02, 1., .102), loc=3, borderaxespad=0.,
                   ncol=5,
                   fontsize=10)
        
        if mode == "human":
            plt.pause(0.1)
            
            for item in artists:
                item.remove()
        elif mode == "rgb_array":
            if time is not None:
                time = round(time,2)
        
            self.crowd_pref_text.set_text('time:{:.2f},'.format(time))
                
            if log_probs is not None:
                if log_probs[0] is not None:
                    self.robot_log_prob_text.set_text('robot input prob:{:.2f},'.format(log_probs[0]))
                    self.robot_log_prob_text.set_visible(True)
                else:
                    self.robot_log_prob_text.set_visible(False)
            # self.text.set_text(velo_text)
            self.render_axis.figure.canvas.draw()
            data = np.frombuffer(self.render_axis.figure.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = self.render_axis.figure.canvas.get_width_height()
            data = data.reshape((h, w, 3))
            for item in artists:
                item.remove()
            return data
        return