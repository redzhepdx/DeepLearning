import csv as csv
import numpy as np

def readFile(filename):
    csv_file_object = csv.reader(open(filename, 'rU'))
    header = next(csv_file_object) #first line is header. next() function skips the next line
    data = []
    #Getting all personal infos from file
    for row in csv_file_object:
        data.append(row)
    data = np.array(data)
    return data

def reorganizeData(data):
    #Life situations (alive = 1 , dead = 0)
    old_results = np.array(data[:,1])
    new_res = []
    for result in old_results:
        new_res.append([float(result) , 1-float(result)])
    results = np.array(new_res)
    #Getting needed information for network from all personal infos
    #Class , sex , age , sipsp , parch
    passenger_infos_for_network = data[:,2:8]
    # deleting name field
    passenger_infos_for_network = np.delete(passenger_infos_for_network,1,axis=1)

    alive_total = 0
    dead_total = 0
    alive_count = 0
    dead_count = 0
    itemindex = 0
    #Changing sex feature type string to integer(male = 1 and female = 0)
    #Filling empty age values
    for passenger in passenger_infos_for_network:
        if (passenger[1] == 'male'):
            passenger[1] = 1
        else:
            passenger[1] = 0
        if(passenger[2] == ''):
            passenger[2] = 0
        passenger = [float(passenger[0]),float(passenger[1]),float(passenger[2]),float(passenger[3]),float(passenger[4])]
        if(data[itemindex,1] == '1'):
            alive_total += float(passenger[2])
            alive_count += 1
        else:
            dead_total += float(passenger[2])
            dead_count += 1
        itemindex += 1
    #Computing avarage age for deads and alives
    alive_avarage = alive_total / alive_count
    dead_avarage = dead_total / dead_count

    #Filling 0 aged passengers with avarage ages
    itemindex = 0
    for passenger in passenger_infos_for_network:
        if(float(passenger[2]) == 0):
            if(data[itemindex,1] == '1'):
                passenger[2] = float(alive_avarage)
            else:
                passenger[2] = float(dead_avarage)
        itemindex += 1
    #Changing datatype matrix for input(string -> float)
    t_i = np.float64(passenger_infos_for_network)
    new_results = vectorize_array(results,2)
    new_passenger_infos = vectorize_array(t_i,5)

    return (new_passenger_infos , new_results)

def vectorize_array(array,size):
    new_array = [np.reshape(x,(size,1)) for x in array]
    return new_array

def load_data():
    #Training Datas
    train_d = readFile('./train.csv')
    training_inputs , training_results = reorganizeData(train_d)
    training_data = zip(training_inputs[:700],training_results[:700])
    #Test Datas
    test_data = zip(training_inputs[700:],training_results[700:])
    return (training_data,test_data)
