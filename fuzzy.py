

import math
import pandas as pd
import random
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


# seed to make it reproducible
random.seed(10)

# define convergence criteria, 0.001
epsilon = 0.001

# add global varaible t
t = 100



# Define manual function
def manual(data, m, c, distance_function, save_results = False, num = None):
    
    chosen_function = distance_function

    # create clusters_list
    clusters_list = []
    # iterate range of c
    for i in range(c):
        # Add i to clusters list
        clusters_list.append(i)


    # make dataframe with same shape as data but empty
    membership_matrix = pd.DataFrame(np.nan, index = [i for i in range(data.shape[0])], columns = [i for i in range(c)])


    # Initial cluster classification
    
    for j in range(membership_matrix.shape[0]):
        # Random int value between 0 and c 
        index = random.randint(0, c-1)

        # Loop through all index in membership_matrix
        for i in range(membership_matrix.shape[1]):

            # if i is equal to index, set value to 1
            if i == index:
                membership_matrix.iloc[j, i] = 1

            # else set value to 0
            else:
                membership_matrix.iloc[j, i] = 0
    previous_cluster_centers = updateClusterCenter(data, membership_matrix, clusters_list, m)
    cluster_centers = previous_cluster_centers.copy()


    for i in range(t):
        

        # convert 2D list cluster_centers to data frame
        cluster_centers_df = pd.DataFrame(cluster_centers, columns= ["X", "Y"], index = clusters_list) 


        # calculate distance matrix between the data and the cluster centers
        distance_matrix = distanceDF(data, chosen_function,cluster_centers_df)

        # calculate the membership matrix
        membership_matrix = calculateMembershipMatrix(distance_matrix, m)

        # Update cluster centers
        cluster_centers = updateClusterCenter(data, membership_matrix, clusters_list, m)


         # calculate distance between previous and current cluster centers
        if t != 0:
           
            # thid is going to iterate through cluster centers
            for i in range(len(cluster_centers)):
                # decide distance function
                if chosen_function == 1:
                # use EuclideanDistancePoint
                    distance = EuclideanDistancePoint(cluster_centers[i], previous_cluster_centers[i])
                elif chosen_function == 2:
                    
                    distance = ManhattanDistancePoint(cluster_centers[i], previous_cluster_centers[i])
                elif chosen_function == 3:
                    
                    distance = ChebyshevDistancePoint(cluster_centers[i], previous_cluster_centers[i])

         
            if distance < epsilon:
                break
        
        previous_cluster_centers = cluster_centers.copy()

    # this is going to convert 2d list cluster_centers to dataframe to save the most updated version of cluster centers
    cluster_centers_df = pd.DataFrame(cluster_centers, columns= ["X", "Y"], index = clusters_list) 

    
    plt.figure(1)


    # variable for scatter plot data 
    color = cm.rainbow(np.linspace(0, 1, c))
    # plt.scatter(data.iloc[:, 0], data.iloc[:, 1]) check if data is correct
    decision_df = DecisionMatrix(membership_matrix)

    
    for i in range(data.shape[0]):
        # plot scatter plot with color same as decision_df
        plt.scatter(data.iloc[i, 0], data.iloc[i, 1], color = color[decision_df.iloc[i, 0]])


    plt.scatter(cluster_centers_df.iloc[:, 0], cluster_centers_df.iloc[:, 1], color="black", marker="D")
    # for center in range(c):
    #     plt.scatter(cluster_centers_df.iloc[center, 0], cluster_centers_df.iloc[center, 1], color=color[center], marker="D")
    
    ######################### SAVE RESULTS OF MANUAL VALUES TO TXT FILES FOR TESTING AND REPORT #########################
   
    if save_results  == True:
        plt.savefig("Datos_"+ str(num) + "/plot_1_c" + str(c) + ".png")

   
    plt.figure(2)

    # iterate through membership_matrix
    for i in range(membership_matrix.shape[0]):

        # iterate through membership_matrix columns
        for j in range(membership_matrix.shape[1]):

            # plot membership_matrix with color same as color variable
            plt.scatter(i, membership_matrix.iloc[i][j], color = color[j])
    
    if save_results  == True:
        plt.savefig("Datos_"+ str(num) + "/plot_2_c" + str(c) + ".png")
    
    plt.show()
    ######################### SAVE RESULTS OF MANUAL VALUES TO TXT FILES FOR TESTING AND REPORT #########################
    
    return membership_matrix, decision_df, cluster_centers_df


# define automatic function
def automatic(run_c_list, m, distance_function, save_results = False):


######################### SAVE RESULTS OF AUTOMATIC VALUES TO TXT FILES FOR TESTING AND REPORT #########################
    # Make save_results dataframe if save_results is True
    if save_results == True:
        save_results_df = pd.DataFrame(columns = ["c", "PC", "FS", "Ball"])

    # Iterate through run_c_list
    for i in range(len(run_c_list)):
        # read data from Datos(i+1).csv
        data = pd.read_csv('Datos_' + str(i+1) + '.csv', header=None)
        # define c
        c = run_c_list[i]

        #print current input data and c 
        print("Datos_" + str(i+1) + ".csv" + " " + "c="+ str(c))
######################### SAVE RESULTS OF AUTOMATIC VALUES TO TXT FILES FOR TESTING AND REPORT #########################
        


        membership_matrix, decision_df, cluster_centers_df = manual(data, m, c, distance_function, save_results, i+1)

        #Call calcpc function and saves it to pc_result_automatic
        pc_result_automatic = calcPC(membership_matrix)
        #call calcFS function and saves it to fs_result_automatic
        fs_result_automatic = calcFS(data, membership_matrix, cluster_centers_df, distance_function)
        #call calcBall function and saves it to ball_result_automatic
        ball_result_automatic = calcBall(data, membership_matrix, decision_df, cluster_centers_df, distance_function)



        print("\n#################### Automatic Index Results ####################\n")
        # print pc_result_automatic
        print("PC: " + str(pc_result_automatic))
        # print fs_result_automatic
        print("FS: " + str(fs_result_automatic))
        # print ball_result_automatic
        print("Ball: " + str(ball_result_automatic))
        print("\n################################################################\n")

        if save_results == True:
            # save pc_result_automatic to save_results_df
            save_results_df.loc[0] = [c, pc_result_automatic, fs_result_automatic, ball_result_automatic]
            # save save_results_df to csv in Datos_i+1
            save_results_df.to_csv('Datos_' + str(i+1) + '/results_automatic_c' + str(c) + '.csv', index=False)



        

    










def updateClusterCenter(data, membership_matrix, clusters_list, m):
    # Update the cluster center

    clusters_classification = membership_matrix.T

    
    # Calculate position of each cluster center
    # dataframe that stores the centers of each cluster 
    cluster_centers = pd.DataFrame(np.nan, columns= ["X", "Y"], index = clusters_list)




    #make lists of numerator_sum and denominator_sum same size of the cluster lits
    numerator_sum = []
    denominator_sum = []
    for i in range(len(clusters_list)):
        temp = []
        for j in range(cluster_centers.shape[1]):
            temp.append(0)
        numerator_sum.append(temp.copy())
        denominator_sum.append(temp.copy())


    # iterate through each row of cluster_centers
    for i in range(cluster_centers.shape[0]):

        # iterate through each column of cluster_centers and i is the cluster number
        
        for j in range(cluster_centers.shape[1]):
            # iterate through each column in clusters_classification j is the dimension (X or Y)
           
            temp_numerator = 0
            for k in range(clusters_classification.shape[1]):
                # for each iteration its going to calculate numerator and denominator
                # k is the mew (u) column, also data index

                # Sum by mew^(m)
                denominator_sum[i][j] += clusters_classification.iloc[i, k] ** m
                # breakpoint()

                # mew^m
                temp = clusters_classification.iloc[i, k] ** m

                # multiplied by value of of data in cluster number i and dimension j
                temp *= data.iloc[k, j]

               
               
                temp_numerator += temp

            # add to numerator sum
            numerator_sum[i][j] = temp_numerator

    
    cluster_centers = []
    for i in range(len(numerator_sum)):
        temp = []
        for j in range(len(numerator_sum[i])):
            temp.append(numerator_sum[i][j] / denominator_sum[i][j])
        cluster_centers.append(temp.copy())
    return cluster_centers



# eulcidian diststance
def EuclideanDistancePoint(point1: list, point2:list):


    # sqrt((x2-x1)^2 + (y2-y1)^2)
    return math.sqrt(((point2[0] - point1[0])**2) + ((point2[1] - point1[1])**2))


#manhattan distance
def ManhattanDistancePoint(point1: list, point2:list):

    # abs(x2-x1) + abs(y2-y1)
    return abs(point2[0] - point1[0]) + abs(point2[1] - point1[1])

#chebyshev distance
def ChebyshevDistancePoint(point1: list, point2:list):

    # max(x2-x1, y2-y1)
    return max(abs(point2[0] - point1[0]), abs(point2[1] - point1[1]))



def distanceDF(data, distance_function, cluster_centers_df):

    # distance_function int between 1 and 3
        # 1 = Euclidean
        # 2 = Manhattan
        # 3 = Chebyshev
   

    #this is to make distance matrix
    #this is the dataframe with same shape as data but empty
    distance_matrix = pd.DataFrame(np.nan, index = [i for i in range(data.shape[0])], columns = [i for i in range(cluster_centers_df.shape[0])])

    # iterate through each row of data
    for i in range(distance_matrix.shape[0]):

        # iterate through each row of data
        for j in range(distance_matrix.shape[1]):
            if distance_function == 1:
                # Euclidean distance
                distance_matrix.iloc[i, j] = EuclideanDistancePoint(data.iloc[i, :].to_list(), cluster_centers_df.iloc[j, :].to_list())
           
            elif distance_function == 2:
                # Manhattan distance
                distance_matrix.iloc[i, j] = ManhattanDistancePoint(data.iloc[i, :].to_list(), cluster_centers_df.iloc[j, :].to_list())
            
            elif distance_function == 3:
                # Chebyshev distance
                distance_matrix.iloc[i, j] = ChebyshevDistancePoint(data.iloc[i, :].to_list(), cluster_centers_df.iloc[j, :].to_list())            
    return distance_matrix


# function to calculate member value
def calcMemberValue(distance_matrix, m, i, j):

 
    membership_value = 0
    temp = 0
    for center_number in range(distance_matrix.shape[1]):
        # divide dij by d_i_center_number

        temp = distance_matrix.iloc[i, j] / distance_matrix.iloc[i, center_number]

        # elvate temp to 2/(m-1)
        temp = temp ** (2/(m-1))

        membership_value += temp
        
    
    membership_value = 1 / membership_value
    # check if membership_value is nan
    if math.isnan(membership_value):
        membership_value = 1

    return membership_value



#  calculate the MembershipMatrix function
def calculateMembershipMatrix(distance_matrix, m):


    #this is to make membership matrix and  for dataframe with same shape as distance_matrix but empty
    membership_matrix = pd.DataFrame(np.nan, index = [i for i in range(distance_matrix.shape[0])], columns = [i for i in range(distance_matrix.shape[1])])

    # iterate through each row of distance_matrix
    for i in range(distance_matrix.shape[0]):

        # iterate through each column of distance_matrix
        for j in range(distance_matrix.shape[1]):

            # calculates membership using calcMemberValue
            membership_matrix.iloc[i, j] = calcMemberValue(distance_matrix, m, i, j)
            

    return membership_matrix



# desition matrix using membership matrix
def DecisionMatrix(membership_matrix):


    # this is to make decision matrix and also to make dataframe with same shape as membership_matrix but empty
    decision_matrix = pd.DataFrame(np.nan, index = [i for i in range(membership_matrix.shape[0])], columns = ["Decision"])

    # iterate through each row of membership_matrix
    for i in range(membership_matrix.shape[0]):

        # Find max number and column in current row
        max_number = max(membership_matrix.iloc[i, :])

        # Find column number of max number and Convert current row into list
        row_list = membership_matrix.iloc[i, :].to_list()

        # Find index of max number
        max_number_index = row_list.index(max_number)

        # Set index as decision in decision_matrix
        decision_matrix.iloc[i, 0] = max_number_index
        

    
    # Change all rows of decision_df to int type CASTING
    decision_matrix = decision_matrix.astype(int)

    return decision_matrix

#function calculate Partition Coefficient
def calcPC(membership_matrix):

    sum = 0
    # this is going to iterate through row and column of membership_matrix
    for i in range(membership_matrix.shape[0]):
        for j in range(membership_matrix.shape[1]):
            sum += (membership_matrix.iloc[i, j]**2)
    
    partition_coeff = 1/sum
    return partition_coeff

# function to calculate average of all data points in dataframe
def calcAverage(data):


    #average list that contains X and Y values
    average_list = []

    # this is going to iterate through each column of data
    for i in range(data.shape[1]):

        # calculate average of column
        average_list.append(data.iloc[:, i].mean())

    return average_list

#function to calculate  FS 
def calcFS(data, membership_matrix, cluster_centers_df, distance_function):
    # Calculate fukuyama-Sugeno Index

    # Calculate average of all data points
    average_data_vector = calcAverage(data)

    FS_sum = 0

    # loop through each row of data
    for i in range(data.shape[0]):
        cluster_sum = 0

        # loop throuhh all membership matrix columns
        for j in range(membership_matrix.shape[1]):
            temp = 0
            temp2 = 0
            temp3 = 0
            temp4 = 0

            # take current mamber and elevate it to 2
            temp = membership_matrix.iloc[i, j] ** 2

            # this is going to decide the distance function
            if distance_function == 1:
                # Euclidean distance
                temp2 = EuclideanDistancePoint(data.iloc[i, :].to_list(), cluster_centers_df.iloc[j, :].to_list())
                temp3 = EuclideanDistancePoint(cluster_centers_df.iloc[j, :].to_list(), average_data_vector)

            elif distance_function == 2:
                # Manhattan distance
                temp2 = ManhattanDistancePoint(data.iloc[i, :].to_list(), cluster_centers_df.iloc[j, :].to_list())
                temp3 = ManhattanDistancePoint(cluster_centers_df.iloc[j, :].to_list(), average_data_vector)

            elif distance_function == 3:
                # Chebyshev distance
                temp2 = ChebyshevDistancePoint(data.iloc[i, :].to_list(), cluster_centers_df.iloc[j, :].to_list())
                temp3 = ChebyshevDistancePoint(cluster_centers_df.iloc[j, :].to_list(), average_data_vector)
            
            # Elevate temp2 to 2
            temp2 = temp2 ** 2

            # Elevate temp3 to 2
            temp3 = temp3 ** 2

            # Subtract temp2 and temp3
            temp4 = temp2 - temp3

            # Multiply current member ** 2 and temp4
            temp = temp * temp4

            # Add temp to cluster_sum
            cluster_sum += temp
        # Add cluster_sum to FS_sum
        FS_sum += cluster_sum

    # Return FS_sum
    return FS_sum




            

            
        



# function to calculate the SSW
def calcBall(data, membership_matrix, decision_df, cluster_centers_df, distance_function):
    
    

    

    final_sum = 0
    # Loop 0 to c-1 using membership_matrix columns
    for i in range(membership_matrix.shape[1]):

        # Filter data based on decision_df column equal to i
        data_i = data[decision_df.iloc[:, 0] == i]

        # loop through rows of data_i
        distance_sum = 0
        for j in range(data_i.shape[0]):
            # calculate distance between data_i row and cluster_centers_df row
            if distance_function == 1:
                # Euclidean distance
                distance = EuclideanDistancePoint(data_i.iloc[j, :].to_list(), cluster_centers_df.iloc[i, :].to_list())

            elif distance_function == 2:
                # Manhattan distance
                distance = ManhattanDistancePoint(data_i.iloc[j, :].to_list(), cluster_centers_df.iloc[i, :].to_list())
            
            elif distance_function == 3:
                # Chebyshev distance
                distance = ChebyshevDistancePoint(data_i.iloc[j, :].to_list(), cluster_centers_df.iloc[i, :].to_list())
           
            # add distance to distance_sum
            distance_sum += distance
            
        # Add final_sum to distance_sum
        final_sum += distance_sum
    
    # divide final_sum by c
    ball_index = final_sum / membership_matrix.shape[1]

    return ball_index



    
            
    

#plot all data from csv
def option1_plot_data():

        df = glob.glob('*.csv')
        df.sort(key=len)

        for df in df :

                df1 = pd.read_csv(df, header = None, sep = ',')
                fig = plt.figure()

                plt.scatter(df1.iloc[:,0], df1.iloc[:,1])
                plt.show()

