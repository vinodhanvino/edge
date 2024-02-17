import random
import numpy as np
import matplotlib.pyplot as plt

number_of_servers = 3
number_of_users = 3


x_range = (0, 1000)
y_range = (0, 1000)

rate_of_upload = np.random.randint(1,1000)


min_power = 101
max_power = 500
min_task = 10
max_task = 15
servers = []


local_server_values = [number_of_servers + 1, 350, 125, 100]


def generate_server_locations(number_of_servers):
    for _ in range(1, number_of_servers + 1):
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        computing_power = random.randint(min_power, max_power)
        lists = [_, x, y, computing_power]
        servers.append(lists)
    servers.append(local_server_values)
    return servers


users_details = []

def calculate_distance(loc1, loc2):
    return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def generate_action_space(users, servers):
    action_space = []
    for _ in range(len(users)):
        user = users[_]
        distances = []
        for server in servers:
            distances.append(calculate_distance(user, server))
        sorted_servers = [server for _, server in sorted(zip(distances, servers), key=lambda pair: pair[0])]
        action_space.append(sorted_servers)
    return action_space

def generate_users_task(number_of_users):
    for urs in range(1, number_of_users + 1):
        task_value = random.randint(min_task, max_task)
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        lists = [urs, x, y, task_value]
        users_details.append(lists)
    return users_details




if __name__ == '__main__':
    server_details = generate_server_locations(number_of_servers)
    user_details = generate_users_task(number_of_users)
    print(f"server array - [s, x,y, pw] {server_details}")
    print(f"users array - [u, x,y, pw] {user_details}")
    space = generate_action_space(user_details, server_details)
    print(f"transpos array  - [t, x,y, pw] {space}")






    # plt.scatter(x2[0],y2[0],marker='o', label='users')
    # plt.show()