import random
import numpy as np
import matplotlib.pyplot as plt

from DQN import DQNAgent
from env import customeEnvironment
from help_functions import create_frequncies, get_T_local,\
    uploading_time, upload_rate, tcomp, response_time,calculate_distance

number_of_servers = 3
number_of_users = 3
x_range = (0, 1000)
y_range = (0, 1000)

power = 500 * 1e-3
gains = 10
N0 = 1e-6
bandwidth = 20 * 1e6
number_cycle = 2


transistion_time = create_frequncies(300, 500, 1e-6)
twait = create_frequncies(300, 500, 1e-3 )
cpu_power = create_frequncies(800, 1200, 1e3)
local_computing_power = create_frequncies(300, 500 , 1e6)
edge_server_computing = create_frequncies(12, 20 , 1e9)
total_edge_server_computing = edge_server_computing * number_of_servers
length_of_task = create_frequncies(300,500, 1e3)
upload_rate = upload_rate(bandwidth,number_of_users, power, gains, N0)
upload_time =  uploading_time(length_of_task, upload_rate)
tLocal =  get_T_local(number_cycle, local_computing_power) / 1e-6

tcomp = tcomp(cpu_power, total_edge_server_computing)
response_time = response_time(twait, tcomp)[-1]

all_users_cords = []
all_servers_cords = []
def generate_users_task(number_of_users):
    min_task = 10
    max_task = 80

    users_details = []
    for urs in range(1, number_of_users + 1):
        task_value = random.randint(min_task, max_task)
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        lists = [urs, x, y, task_value]
        users_details.append(lists)
    return users_details


def generate_server_locations(number_of_servers):
    servers = []
    for _ in range(1, number_of_servers + 1):
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        computing_power = create_frequncies(12,20, 1e6)
        lists = [_, x, y, computing_power]
        servers.append(lists)
    return servers




class Ops:

    def calculate_delay(self):
        self.users = generate_users_task(number_of_users)
        self.servers = generate_server_locations(number_of_servers)
        all_users_cords.append(self.users)
        all_servers_cords.append(self.servers)
        self.shortest_distance_users = []
        for usr in self.users:
            distance = []
            for server in self.servers:
                tup = {}
                tup["server"] = server[0]
                tup["distance"] = round(calculate_distance(usr[1:-1], server[1:-1]),2)
                distance.append(tup)
                # print(distance)
            short_distance_server = sorted(distance, key=lambda x: x['distance'], reverse= True)[-1]
            print(f"Distance from User id {usr[0]} to server id {short_distance_server['server']} is {short_distance_server['distance']}")
            self.shortest_distance_users.append([usr[0],short_distance_server['server']])

        for sdu in self.shortest_distance_users:
            print(f"User id - {sdu[0]}, Server id - {sdu[1]}")
        return self.shortest_distance_users


server_map = {
    "m1" : 1,
    "m2" : 2,
    "m3" : 3,
}

def plot_users_servers(users, servers):

    x_u = [x[1] for x in users]
    y_u = [x[2] for x in users]

    x2 = [x[1] for x in servers]
    y2 = [x[2] for x in servers]
    user_id = [str(x) for x in range(1,number_of_users)]

    #plt.figure(figsize=(10, 6))
    plt.scatter(x_u, y_u, marker='o', color='blue')

    # Add labels for each point
    for i, name in enumerate(user_id):
        plt.annotate(name, (x_u[i], y_u[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.show()
def time_calculations():
    op = Ops()
    filtered_user = op.calculate_delay()
    t_off = 0
    user_tast = 0
    for fuser in filtered_user:
        t_task = [x for x, y in server_map.items() if y == fuser[-1]][-1]
        if t_task == 'm2':
            t_off = upload_time + response_time
        elif t_task == 'm3':
            t_off = upload_time + response_time + transistion_time
        else:
            t_off = upload_time + response_time + transistion_time
        try:
            user_tast = op.users[fuser[0]-1][-1]
        except:
            continue
    avg = (t_off * user_tast) / len(filtered_user)
    for f in filtered_user:
        f.append(avg)

    print(f"Average Offloadind time {avg})")
    # print(filtered_user)
    return filtered_user


def check_if_done(current_step, max_steps):
    if current_step >= max_steps:
        return True
    else:
        return False



batch_size = 1
def main(data, user_check):

    episodes = 100
    state_space = [y[-1] for x in data for y in x]
    agent = DQNAgent(len(state_space))
    env = customeEnvironment(state_space)
    done = False
    for episode in range(1,episodes-2):
        print(episode)
        state = np.array([state_space])
        total_rewords = 0


        reward, done = env.step(data[episode])
        agent.remember(state[0],0,reward,state, done)
        total_rewords += reward
        agent.replay(batch_size)
        # done  = check_if_done(episode, episodes)
    agent.model.save(rf"E:\projects\New folder\model_1.keras")
    if user_check > number_of_users:
        return "Invalid user"
    else:
        try:
            return data[agent.act([user_check])]
        except:
            return data[agent.act([user_check])]

if __name__ == '__main__':
    data = []
    user_check = 2
    for _ in range(1, 100):
        data.append(time_calculations())
    print(data)
    time = [y[-1] for x in data for  y in x ]
    plt.figure()
    plt.plot(time)
    result = main(data, user_check)
    result
