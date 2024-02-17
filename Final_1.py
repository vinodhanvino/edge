
import math
import random
import torch

import matplotlib.pyplot as plt
import numpy as np

nu = 3  # the number of CUs
nd = 2  # the number of D2D pairs
nc = nu  # the number of channels
darea = 1000  # the length of the square area to generate users.
d2dmin = 15  # min d2d distance
d2dmax = 50  # max d2d distance
dmin = 20  # minimum distance between BS and CUs
dbmin = 20  # min distance between d2d tx and base station
multiplier_generating_comp = 5

pumax = torch.ones(1)*100  # the max power of each CU
pdmax = torch.ones(1)*100  # the max power of each D2D pair
rcmin = 10  # the min data rate of CUs
sinrcmin_l = 2**rcmin - 1  # min linear SINR for CUs
rdmin = 0  # the min data rate of CUs
sinrdmin_l = 2**rdmin - 1  # min linear SINR for CUs
pre = 5
noise = 1e-14  # noise power

# generate locations of BS and CUs
bs_loc = [0,0]

u_loc_pre = torch.FloatTensor(nu*pre, 2).uniform_(-darea/2, darea/2)  # generate more CUs locations

dist_pre = torch.zeros(nu*pre)
u_loc = torch.zeros(nu,2)
dist_bu = torch.zeros(nu)
dist_pre = torch.sqrt((bs_loc[0] - u_loc_pre[:,0]) ** 2 + (bs_loc[1] - u_loc_pre[:,1]) ** 2)
#print('dist_pre',dist_pre)

users=3
Edge_servers=3
Link_bandwidth = 200*1E9
user_upload_PWR = 500*1E-3
gain = 10

#########################Edge_setup######################

# Define the ""length of task ""
start_frequency_KHz = (300)
end_frequency_KHz = (500)

# Convert GHz values to Hz
start_frequency_Hz = start_frequency_KHz * 1e3
end_frequency_Hz = end_frequency_KHz * 1e3

# Generate a random integer within the specified range
random_frequency_Hz = random.randint(int(start_frequency_Hz), int(end_frequency_Hz))

# Print the generated random frequency in Hz
#print (random_frequency_Hz/1e9)

# Input float number
float_number = random_frequency_Hz

float = random_frequency_Hz

# Extract the whole number part using int()
Task_TX_len = int(float_number)

# Print the extracted whole number
print("task transfer length")
print(int(Task_TX_len/1e3))


##########################################################




##################################################
import random

Edge_servers=3

# Define the frequency range in""" GHz for EDGE SERVER COMPUTING POWER"""
start_frequency_GHz = 10
end_frequency_GHz = 18

# Convert GHz values to Hz
start_frequency_Hz = start_frequency_GHz * 1e6
end_frequency_Hz = end_frequency_GHz * 1e6

for i in range(0,3):
    random_frequency_Hz = random.randint(int(start_frequency_Hz), int(end_frequency_Hz))


# Input float number
    float_number = random_frequency_Hz/Task_TX_len


# Extract the whole number part using int()
    EdgeServerPWR = int(float_number)

    ed1=ed2=ed3=0
    if i == 0:
        ed1 = EdgeServerPWR
    elif i == 1:
        ed2 = EdgeServerPWR
    elif i == 2:
        ed3 = EdgeServerPWR

# Print the extracted whole number
    print(f"task capability EdgeServerPWR {i + 1}: {EdgeServerPWR}")
    #print(f"EdgeServerPWR in FZ {i + 1}: {int(random_frequency_Hz/1e9)} ",f"MHz") 
   
    #print(f"EdgeServerPWR in FZ in GHz {i + 1}: {int(random_frequency_Hz)} " ,f"MHz")


    print(ed1,ed2,ed3)

####################################


import random

# Define the ''local comp PWR ""''
start_frequency_MHz = 300
end_frequency_MHz = 500

# Convert GHz values to Hz
start_frequency_Hz = start_frequency_MHz * 1e3
end_frequency_Hz = end_frequency_MHz * 1e3

# Generate a random integer within the specified range
random_frequency_Hz = random.randint(int(start_frequency_Hz), int(end_frequency_Hz))

# Input float number
float_number = random_frequency_Hz/Task_TX_len

# Extract the whole number part using int()
Local_Comp_PWR = int(float_number)

# Print the extracted whole number
print("Local_Comp_PWR", Local_Comp_PWR/1e3)
#print("\n Local comp pwr in Fz " ,random_frequency_Hz /1e6 )
#print("\n Local comp pwr in Fz in MHZ" ,random_frequency_Hz , "MHz" )



##########################################################

'''


import random

# Define the frequency range in GHz
start_frequency_GHz = 12
end_frequency_GHz = 20

# Convert GHz values to Hz
start_frequency_Hz = start_frequency_GHz * 1e9
end_frequency_Hz = end_frequency_GHz * 1e9

# Generate a random integer within the specified range
random_frequency_Hz = random.randint(int(start_frequency_Hz), int(end_frequency_Hz))

# Print the generated random frequency in Hz
#print (random_frequency_Hz/1e9)

# Input float number
float_number = random_frequency_Hz/1e9

# Extract the whole number part using int()
whole_number = int(float_number)

# Print the extracted whole number
#print(whole_number)
'''

######



upload_rate = (Link_bandwidth / users) * math.log2(1 + (user_upload_PWR * gain) / ((Link_bandwidth / users) * 10 ** (-174 / 100)))   # Rn = upload rate

print("upload rate" , int(upload_rate))



## T local 

No_of_cycle_u1 = 500            #no of task by users
no_of_cycle_u2 = 500
no_of_cycle_u3 = 500

#t_local  = No_of_cycle_u1/Local_Comp_PWR



t_upload_u1 = No_of_cycle_u1/ upload_rate
t_upload_u2 = no_of_cycle_u2/ upload_rate
t_upload_u3 = no_of_cycle_u3/ upload_rate

t_wait = 2


t_comp = ((No_of_cycle_u1 + no_of_cycle_u2 + no_of_cycle_u3) / (ed1+ed2+ed3))+t_wait
print("edge_computing time")
print(int(t_comp))

#################################channel_Gen################################################

count = 0
for i_u in range(nu*pre):  # loop over each user/column, save the users with both distance > dmin
    if (dist_pre[i_u] > dmin).all(): 
        u_loc[count,:] = u_loc_pre[i_u,:]
        dist_bu[count] = dist_pre[i_u]
        count = count + 1
    else:
        i_u = i_u + 1  
    if count == nu:
        break
#print('u_loc', u_loc)
#print('dist',dist_pre)

# generate locations of D2D pairs 
dt_loc_pre = torch.FloatTensor(nd*pre, 2).uniform_(-darea/2, darea/2)  # generate more D2D tx loctions
dt_loc = torch.zeros(nd,2)
dist_bd_pre = torch.zeros(nd*pre)
dist_bd_pre = torch.sqrt((bs_loc[0] - dt_loc_pre[:,0]) ** 2 + (bs_loc[1] - dt_loc_pre[:,1]) ** 2)
dist_bd = torch.zeros(nd)
#print('dist_bd_pre',dist_bd_pre)

count_d = 0
for i_d in range(nd*pre):  # loop over each user/column, save the users with both distance > dmin
    if (dist_bd_pre[i_d] > dbmin).all(): 
        dt_loc[count_d,:] = dt_loc_pre[i_d,:]
        dist_bd[count_d] = dist_bd_pre[i_d]
        count_d = count_d + 1
    else:
        i_d = i_d + 1  
    if count_d == nd:
        break
#print('dt_loc',dt_loc)

dr_loc = torch.zeros(nd,2)
d2d_dist = torch.FloatTensor(nd, 1).uniform_(d2dmin, d2dmax)
torch.pi = torch.acos(torch.zeros(1)).item() * 2
angle = torch.FloatTensor(nd, 1).uniform_(0, 2*torch.pi)
dr_loc[:,0] = (d2d_dist * torch.cos(angle) + dt_loc[:,0].reshape(nd,1)).reshape(1,nd)
dr_loc[:,1] = (d2d_dist * torch.sin(angle) + dt_loc[:,1].reshape(nd,1)).reshape(1,nd)

# calculate distance between CU and rx of D2D pairs
dist_udr = torch.zeros(nu,nd)
for i_u in range(nu):
    for i_dr in range(nd):
        dist_udr[i_u,i_dr] = torch.sqrt((u_loc[i_u,0] - dr_loc[i_dr,0])**2 + (u_loc[i_u,1] - dr_loc[i_dr,1])**2)
# print('dist_udr', dist_udr)

# calculate distance between D2D pairs
dist_dd = torch.zeros(nd,nd)
for i_dt in range(nd):
    dist_dd[i_dt,:] = torch.sqrt((dt_loc[i_dt,0] - dr_loc[:,0]) ** 2 + (dt_loc[i_dt,1] - dr_loc[:,1]) ** 2)
# print('dist_dd',dist_dd)

# calculate distance between CU and rx of D2D pairs
dist_udr = torch.zeros(nu,nd)
for i_u in range(nu):
    for i_dr in range(nd):
        dist_udr[i_u,i_dr] = torch.sqrt((u_loc[i_u,0] - dr_loc[i_dr,0])**2 + (u_loc[i_u,1] - dr_loc[i_dr,1])**2)
#print('dist_udr', dist_udr)

# calculate distance between tx of D2D pairs and BS
dist_dtb = torch.zeros(nd)
for i_dt in range(nd):
    dist_dtb[i_dt] = torch.sqrt((dt_loc[i_dt,0] - bs_loc[0])**2 + (dt_loc[i_dt,1] - bs_loc[1])**2)
# print('dist_dtb', dist_dtb)

# path loss models, dB
plbu = 128.1 + 37.6 * torch.log10(dist_bu/1e3)        # path loss from CUs to BS
pludr = 128.1 + 37.6 * torch.log10(dist_udr/1e3)  # path loss from CU to rx of D2D
pldd = 148 + 40 * torch.log10(dist_dd/1e3)  # path loss of D2D pairs
pldtb = 148 + 40 * torch.log10(dist_dtb/1e3)  # path loss from tx of D2D to BS

# convert path loss(dB) to linear
hbu = 10**(-plbu/10)
# print('hbu',hbu)# path loss from CUs to BS
hudr = 10**(-pludr/10)
# print('hudr',hudr)
hdd = 10**(-pldd/10)
#print('hdd',hdd)
hdtb = 10**(-pldtb/10)
#print('hdtb',hdtb)

# generate small scale fading with same distribution
csir_bu = torch.normal(mean=0,std=0.5**0.5,size=(1,nu))  # (1*nu)
csii_bu = torch.normal(mean=0,std=0.5**0.5,size=(1,nu))
csi_bu = csir_bu**2 + csii_bu**2
#print('csi_bu',csi_bu,"path loss from CUs to BS")    

import torch

# Extracting values and storing them in individual variables
path_loss_S1 = csi_bu[0, 0].item()
path_loss_S2 = csi_bu[0, 1].item()
path_loss_S3 = csi_bu[0, 2].item()

# Printing the extracted values
print(" path loss from CUs to BS to server 1 =", path_loss_S1)
print(" path loss from CUs to BS to server 2=", path_loss_S2)
print(" path loss from CUs to BS to server 3=", path_loss_S3)
print("\n")
                               # path loss from CUs to BS

csir_udr = torch.normal(mean=0,std=0.5**0.5,size=(nu,nd))
csii_udr = torch.normal(mean=0,std=0.5**0.5,size=(nu,nd))
csi_udr = csir_udr**2 + csii_udr**2
# print('csi_udr',csi_udr)

csir_dd = torch.normal(mean=0,std=0.5**0.5,size=(1,nd))
csii_dd = torch.normal(mean=0,std=0.5**0.5,size=(1,nd))
csi_dd = csir_dd**2 + csii_dd**2
# print('csi_dd',csi_dd)

csir_dtb = torch.normal(mean=0,std=0.5**0.5,size=(nd,nu))
csii_dtb = torch.normal(mean=0,std=0.5**0.5,size=(nd,nu))
csi_dtb = csir_dtb**2 + csii_dtb**2
# print('csi_dtb',csi_dtb)

# channel includes small and large scale fading
ch_bu = torch.sqrt(csi_bu * hbu)  # (1,nu)               # path loss from CUs to BS
#print('ch_bu',ch_bu,"small and large scale fadingfrom CUs to BS")

x = csi_bu[0, 0].item()
y = csi_bu[0, 1].item()
z = csi_bu[0, 2].item()

# Printing the extracted values
#print(" small and large scale fadingfrom CUs to BS to server 1 =", x)
#print(" small and large scale fadingfrom CUs to BS to server 2 =", y)
#print(" small and large scale fadingfrom CUs to BS to server 3 =", z)

ch_udr = torch.sqrt(csi_udr * hudr)
# print('ch_udr',ch_udr)
ch_dd = torch.sqrt(csi_dd * torch.diag(hdd))
# print('ch_dd',ch_dd)

ch_dtb = torch.zeros(nd,nu)
for i_u in range(nu):
    ch_dtb[:,i_u] = torch.sqrt(csi_dtb[:,i_u] * hdtb)
#print('ch_dtb',ch_dtb)




# Given location pairs
location_pairs = u_loc 

# Create the plot
plt.figure(figsize=(10, 6))
plt.axhline(0, color='black', linewidth=0.8)  # Add x-axis
plt.axvline(0, color='black', linewidth=0.8)  # Add y-axis
poker =1
# Plot lines from the origin to the points and show distance
for x, y in location_pairs:
    plt.scatter(x, y, color='green', marker='s', label='Server')  # Server symbol
    plt.plot([0, x], [0, y], color='blue', linestyle='dashed')
    distance = np.sqrt(x**2 + y**2)
    
    #print("distance from basestation to servers: ", poker, distance)

    if poker == 1:
        dist_S1  = distance.clone().detach()
    elif poker == 2:
        dist_S2  = distance.clone().detach()
    elif poker == 3: 
        dist_S3  = distance.clone().detach()
    poker += 1
    plt.text(x, y, f'{distance:.2f}', color='red', fontsize=10)


dist_S1  = dist_S1.item()
dist_S2  = dist_S2.item()
dist_S3  = dist_S3.item()
print("distance from basestation to server 1 " , dist_S1 , "\n distance from basestation to server 2",dist_S2," \n distance from basestation to server 3", dist_S3)
# Plot the base station at the origin (center)
plt.scatter(0, 0, color='blue', marker='o', label='Base Station')  # Base station symbol

plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Base Station and Servers with Distances from Origin')
plt.legend()
plt.grid(True)
plt.show()
