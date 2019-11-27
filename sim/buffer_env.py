import numpy as np

RANDOM_SEED = 42
ACTION_SPACE = ['a+b~c-','a-b~c+','a~b+c-','a~b-c+','a+b-c~','a-b+c~']
NUMBER_DATASET = 100000

class Environment:
    def __init__(self, buffer_combine, ARE, total_number, random_seed=RANDOM_SEED):
        assert len(buffer_combine) == total_number

        np.random.seed(random_seed)

        self.buffer_combine = buffer_combine
        self.total_number = total_number
        self.ARE = [np.float32(a) for a in ARE]
        self.count = 0 # summary the data has been feed to training network
        self.end_of_dataset = False

    def get_initial_size(self):
        index = np.random.randint(self.total_number)
        self.select = self.buffer_combine[index] # current buffer distribution for reinforcement adjust
        self.count = self.count + 1

        return self.select, self.ARE[index], self.end_of_dataset

    def get_next_size(self, action, pre_select = None, train_flag=True):
        refer = ACTION_SPACE.index(action)
        are = 0
        # during training, pre_select is None
        # during testing, pre_select identify previous test-data
        if pre_select is not None:
            self.select = pre_select
        if refer == 0:
            heavy = self.select[0]+1
            light = self.select[2]-1
            select_list = []
            are_list = []
            for ii in range(len(self.buffer_combine)):
                if self.buffer_combine[ii][0] >= heavy and self.buffer_combine[ii][2] <= light:
                    select_list.append(self.buffer_combine[ii])
                    are_list.append(self.ARE[ii])
            if len(select_list) > 0:
                fit_index = self.find_fit(select_list, 'small', 0)
                self.select = select_list[fit_index]
                are = are_list[fit_index]
        if refer == 1:
            heavy = self.select[0]-1
            light = self.select[2]+1
            select_list = []
            are_list = []
            for ii in range(len(self.buffer_combine)):
                if self.buffer_combine[ii][0] <= heavy and self.buffer_combine[ii][2] >= light:
                    select_list.append(self.buffer_combine[ii])
                    are_list.append(self.ARE[ii])
            if len(select_list) > 0:
                fit_index = self.find_fit(select_list, 'big', 0)
                self.select = select_list[fit_index]
                are = are_list[fit_index]
        if refer == 2:
            buffer = self.select[1]+1
            light = self.select[2]-1
            select_list = []
            are_list = []
            for ii in range(len(self.buffer_combine)):
                if self.buffer_combine[ii][1] >= buffer and self.buffer_combine[ii][2] <= light:
                    select_list.append(self.buffer_combine[ii])
                    are_list.append(self.ARE[ii])
            if len(select_list) > 0:
                fit_index = self.find_fit(select_list, 'small', 1)
                self.select = select_list[fit_index]
                are = are_list[fit_index]
        if refer == 3:
            buffer = self.select[1]-1
            light = self.select[2]+1
            select_list = []
            are_list = []
            for ii in range(len(self.buffer_combine)):
                if self.buffer_combine[ii][1] <= buffer and self.buffer_combine[ii][2] >= light:
                    select_list.append(self.buffer_combine[ii])
                    are_list.append(self.ARE[ii])
            if len(select_list) > 0:
                fit_index = self.find_fit(select_list, 'big', 1)
                self.select = select_list[fit_index]
                are = are_list[fit_index]
        if refer == 4:
            heavy = self.select[0]+1
            buffer = self.select[1]-1
            select_list = []
            are_list = []
            for ii in range(len(self.buffer_combine)):
                if self.buffer_combine[ii][0] >= heavy and self.buffer_combine[ii][1] <= buffer:
                    select_list.append(self.buffer_combine[ii])
                    are_list.append(self.ARE[ii])
            if len(select_list) > 0:
                fit_index = self.find_fit(select_list, 'small', 0)
                self.select = select_list[fit_index]
                are = are_list[fit_index]
        if refer == 5:
            heavy = self.select[0]-1
            buffer = self.select[1]+1
            select_list = []
            are_list = []
            for ii in range(len(self.buffer_combine)):
                if self.buffer_combine[ii][0] <= heavy and self.buffer_combine[ii][1] >= buffer:
                    select_list.append(self.buffer_combine[ii])
                    are_list.append(self.ARE[ii])
            if len(select_list) > 0:
                fit_index = self.find_fit(select_list, 'big', 0)
                self.select = select_list[fit_index]
                are = are_list[fit_index]
        if are == 0 and train_flag:
            index = np.random.randint(self.total_number)
            self.select = self.buffer_combine[index]
            are = self.ARE[index]
        self.count = self.count + 1
        if self.count == NUMBER_DATASET:
            self.end_of_dataset = True

        return self.select, are, self.end_of_dataset

    def find_fit(self, select_list, small_or_big='big', adjust=-1):
        if small_or_big == 'small':
            minimum = 1000000
            for ii in range(len(select_list)):
                if select_list[ii][adjust] < minimum:
                    minimum = select_list[ii][adjust]
                    fit_index = ii
        else:
            maximum = -1000000
            for ii in range(len(select_list)):
                if select_list[ii][adjust] > maximum:
                    maximum = select_list[ii][adjust]
                    fit_index = ii
        return fit_index


