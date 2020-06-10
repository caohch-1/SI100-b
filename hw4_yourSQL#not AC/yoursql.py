# =============================================================================#
#                            Homework 4: yourSQL                              #
#         SI 100B: Introduction to Information Science and Technology         #
#                     Spring 2020, ShanghaiTech University                    #
#                     Author: Diao Zihao <hi@ericdiao.com>                    #
#                         Last motified: 02/18/2020                           #
# =============================================================================#
# Implement your database here.
import csv
import re
import copy
from itertools import product


class Row():
    """
    The `Row` class.

    You are building the row class here.
    """

    def set_pk(self, pk):
        self.__pkey = pk

    def add_key(self, key, value):
        self.__m_data[key] = value
        if key not in self.__lex_keys:
            self.__lex_keys.append(key)

    def get_data(self):
        return self.__m_data

    def __init__(self, keys, data, primary_key=None):
        if (isinstance(keys, list)) and (isinstance(data, list)):
            if (len(keys) != len(data)):
                raise KeyError
            if len(keys) == 0 or len(data) == 0:
                raise ValueError
            self.__m_data = {}
            if primary_key == None:
                self.__pkey = keys[0]
            else:
                if primary_key not in keys:
                    raise KeyError
                self.__pkey = primary_key
            self.__m_data = dict(zip(keys, data))

            temp = list(self.__m_data.keys())
            temp.sort()
            self.__lex_keys = temp

            self.__start_num = -1
        else:
            raise TypeError

    ''''deep copy'''

    def __getitem__(self, key):
        if key not in self.__lex_keys:
            raise KeyError
        else:
            res = copy.deepcopy(self.__m_data[key])
            return res
        ## YOUR CODE HERE ##

    def __setitem__(self, key, value):
        if key not in self.__lex_keys:
            raise KeyError
        self.__m_data[key] = value
        ## YOUR CODE HERE ##

    def __iter__(self):
        return self
        ## YOUR CODE HERE ##

    ''''deep copy'''

    def __next__(self):
        self.__start_num += 1
        if self.__start_num < len(self.__lex_keys):
            res = copy.deepcopy(self.__lex_keys[self.__start_num])
            return res
        elif self.__start_num >= len(self.__lex_keys):
            self.__start_num = -1
            raise StopIteration()
        ## YOUR CODE HERE ##

    def __len__(self):
        return len(self.__lex_keys)
        ## YOUR CODE HERE ##

    def __lt__(self, other):
        if (not isinstance(other, Row)) or (not isinstance(self, Row)) or (self.keys() != other.keys()) or (
                self.get_primary_key() != other.get_primary_key()):
            raise TypeError
        if self[self.get_primary_key()] < other[other.get_primary_key()]:
            return True
        else:
            return False
        ## YOUR CODE HERE ##

    ''''deep copy'''

    def keys(self):
        self.__lex_keys.sort()
        res = copy.deepcopy(self.__lex_keys)
        return res
        ## YOUR CODE HERE ##

    ''''deep copy'''

    def get_primary_key(self):
        res = copy.deepcopy(self.__pkey)
        return res
        ## YOUR CODE HERE ##


class Table():
    """
    The `Table` class.

    This class represents a table in your database. The table consists
    of one or more lines of rows. Your job is to read the content of the table
    from a CSV file and add the support of iterator to the table. See the
    specification in README.md for detailed information.
    """

    def de_repeat(self):
        temp = list()
        for row in self.__m_rows:
            flag = False
            for temp_row in temp:
                if row.get_data() == temp_row.get_data():
                    flag = True
            if not flag:
                temp.append(row)
        self.__m_rows = temp

    def set_pk(self, pk):
        self.__pkey = pk
        for row in self.__m_rows:
            row.set_pk(pk)

    def avg(self, key, key2=None, value=None):
        if key2 is None or value is None:
            avg = 0
            for row in self.__m_rows:
                avg += row[key]
            return avg / len(self.__m_rows)
        else:
            avg = 0
            count = 0
            for row in self.__m_rows:
                if row[key2] == value:
                    avg += row[key]
                    count += 1
            return avg / count

    def sum(self, key, key2=None, value=None):
        if key2 is None or value is None:
            sum_ = 0
            for row in self.__m_rows:
                sum_ += row[key]
            return sum_
        else:
            sum_ = 0
            for row in self.__m_rows:
                if row[key2] == value:
                    sum_ += row[key]
            return sum_

    def max(self, key, key2=None, value=None):
        if key2 is None or value is None:
            max_ = self.__m_rows[0][key]
            for row in self.__m_rows:
                if row[key] >= max_:
                    max_ = row[key]
            return max_
        else:
            max_ = None
            for row in self.__m_rows:
                if row[key2] == value:
                    if max_ is None:
                        max_ = row[key]
                    elif row[key] >= max_:
                        max_ = row[key]
            return max_

    def min(self, key, key2=None, value=None):
        if key2 is None or value is None:
            min_ = self.__m_rows[0][key]
            for row in self.__m_rows:
                if row[key] <= min_:
                    min_ = row[key]
            return min_
        else:
            min_ = None
            for row in self.__m_rows:
                if row[key2] == value:
                    if min_ is None:
                        min_ = row[key]
                    elif row[key] <= min_:
                        min_ = row[key]
            return min_

    def calculate(self, agg, key, key2=None, value=None):
        if agg == 'AVG':
            return self.avg(key, key2, value)
        elif agg == 'SUM':
            return self.sum(key, key2, value)
        elif agg == 'MIN':
            return self.min(key, key2, value)
        elif agg == 'MAX':
            return self.max(key, key2, value)

    def add_keys(self, newkey, value, num=None):
        if newkey not in self.__m_keys:
            self.__m_keys.append(newkey)
        if num is not None:
            self.__m_rows[num].add_key(newkey, value)
        else:
            for i in range(len(self.__m_rows)):
                self.__m_rows[i].add_key(newkey, value)

    def add_key(self, key):
        if key not in self.__m_keys:
            self.__m_keys.append(key)

    def set_name(self, name):
        self.__tablename = name

    def set_keys(self, keys):
        self.__m_keys = keys
        res = []
        for row in self.__m_rows:
            value = list(row.get_data().values())
            temp = Row(self.__m_keys, value)
            res.append(temp)
        self.__m_rows = res

    def get_rows(self):
        return self.__m_rows

    def get_row(self, num):
        return self.__m_rows[num]

    def __init__(self, filename, rows=None, keys=None, primary_key=None):
        self.__tablename = filename
        self.__m_rows = list()
        if rows is None:
            with open(filename) as f:
                reader = csv.reader(f)
                data = []
                flag = 0
                for row in reader:
                    if row == [] or row[0] == '':
                        continue
                    temp_row = []
                    if flag == 0:
                        for item in row:
                            item=str(item).strip()
                            temp_row.append(item)
                        flag=1
                    else:
                        for item in row:
                            item = str(item).strip()
                            if re.match(r"\d+\.\d*", item):
                                item = float(item)
                            elif re.match(r"\d+\d*", item):
                                item = int(item)
                            temp_row.append(item)
                    data.append(temp_row)
                new_data0 = []
                for key in data[0]:
                    new_data0.append(str(key))
                self.__m_keys = new_data0
                data.pop(0)
                for row in data:
                    self.__m_rows.append(Row(self.__m_keys, row, primary_key))
        else:
            self.__m_keys = keys
            self.__m_rows = rows

        if primary_key is None:
            if self.__m_keys:
                self.__pkey = self.__m_keys[0]
        else:
            if self.__m_keys:
                self.__pkey = primary_key
        if self.__m_keys:
            if self.__pkey not in self.__m_keys:
                raise KeyError

        self.__m_keys.sort()
        self.__m_rows.sort(key=lambda x: x[self.__pkey])

        self.__start_num = -1
        self.__num = len(self.__m_rows)

        ## YOUR CODE HERE ##

    def __iter__(self):
        return self
        ## YOUR CODE HERE ##

    ''''deep copy'''

    def __next__(self):
        self.__start_num += 1
        if self.__start_num >= self.__num:
            self.__start_num = -1
            raise StopIteration()
        res = copy.deepcopy(self.__m_rows[self.__start_num])
        return res
        ## YOUR CODE HERE ##

    def __getitem__(self, key):
        for i in range(len(self.__m_rows)):
            if self.__m_rows[i][self.__pkey] == key:
                return self.__m_rows[i]
        raise ValueError
        ## YOUR CODE HERE ##

    def __len__(self):
        return len(self.__m_rows)
        ## YOUR CODE HERE ##

    ''''deep copy'''

    def get_table_name(self):
        res = copy.deepcopy(self.__tablename)
        return res
        ## YOUR CODE HERE ##

    ''''deep copy'''

    def keys(self):
        res = copy.deepcopy(self.__m_keys)
        return res
        ## YOUR CODE HERE ##

    ''''deep copy'''

    def get_primary_key(self):
        res = copy.deepcopy(self.__pkey)
        return res
        ## YOUR CODE HERE ##

    def export(self, filename=None):
        if filename is None:
            name = self.__tablename
        else:
            name = filename
        fileheader = self.__m_keys
        data = []
        for row in self.__m_rows:
            temp = row.get_data()
            res = sorted(temp.items(), key=lambda x: x[0])
            temp = []
            for value in res:
                temp.append(value[1])
            data.append(temp)
        with open(name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fileheader)
            writer.writerows(data)
        ## YOUR CODE HERE ##


class Query():
    """
    The `Query` class.
    """

    # def judge(self, left, right, operator):
    #     if operator == '==':
    #         return left == right
    #     elif operator == '>':
    #         return left > right
    #     elif operator == '<':
    #         return left < right
    #     elif operator == '>=':
    #         return left >= right
    #     elif operator == '<=':
    #         return left <= right
    #     elif operator == '!=':
    #         return left != right

    def __init__(self, query, table=None):
        self.__keys = query['select']
        self.__where = query['where']
        if table is None:
            self.__table = Table(query['from'])
            if self.__table.get_primary_key() not in self.__keys:
                self.__keys.append(self.__table.get_primary_key())
        else:
            self.__table = table
            if self.__table.get_primary_key() not in self.__keys:
                self.__keys.insert(0,self.__table.get_primary_key())

        self.__tablename = self.__table.get_table_name()

        for key in self.__keys:
            if key not in self.__table.keys():
                raise KeyError

        self.res_list=list()

        for row in self.__table.get_rows():
            counter=0

            for where in self.__where:
                ttemp_key=where[0]
                operator=where[-1]
                condition=where[1]
                if operator == '==':
                    if row[ttemp_key] == condition:
                        counter+=1
                elif operator == '>':
                    if row[ttemp_key] > condition:
                        counter+=1
                elif operator == '>=':
                    if row[ttemp_key] >= condition:
                        counter+=1
                elif operator == '<':
                    if row[ttemp_key] < condition:
                        counter+=1
                elif operator == '<=':
                    if row[ttemp_key] <= condition:
                        counter+=1
                elif operator == '!=':
                    if row[ttemp_key] != condition:
                        counter+=1
            if counter == len(self.__where):
                temp=copy.deepcopy(row)
                self.res_list.append(temp)

            res_rows = []
            for row in self.res_list:
                data = []
                for key in self.__keys:
                    data.append(row[key])
                res_rows.append(Row(self.__keys, data))
            self.__table=Table(query['from'],res_rows,self.__keys,self.__table.get_primary_key())



        # if not self.__where:
        #     rows = [row for row in self.__table]
        #     res_rows = []
        #     for row in rows:
        #         data = []
        #         for key in self.__keys:
        #             data.append(row[key])
        #         res_rows.append(Row(self.__keys, data))
        #     self.__table = Table(query['from'], res_rows, self.__keys, self.__table.get_primary_key())
        # else:
        #     for key in self.__where:
        #         if key[0] not in self.__table.keys():
        #             raise KeyError
        #
        #     rows = []
        #     for row in self.__table:
        #         count = 0
        #         for where in self.__where:
        #             key = where[0]
        #             left = where[1]
        #             operator = where[2]
        #             if self.judge(row[key], left, operator):
        #                 count += 1
        #         if count == len(self.__where):
        #             rows.append(row)
        #

        #     self.__table = Table(query['from'], res_rows, self.__keys, self.__table.get_primary_key())

        ## YOUR CODE HERE ##

    def as_table(self):
        res = copy.deepcopy(self.__table)
        if isinstance(res.get_table_name(), list):
            res.set_name('join.csv')
        return res
        ## YOUR CODE HERE ##

    def set_table(self, table):
        self.__table = table


class JoinQuery(Query):

    def __init__(self, query):
        self.mquery = query

    def as_table(self):
        query1 = self.mquery['from'][0]
        query2 = self.mquery['from'][1]

        self.table1 = Table(query1)
        t1_keys = self.table1.keys()

        self.table2 = Table(query2)
        t2_keys = self.table2.keys()

        title1 = (self.table1.get_table_name()).replace('csv', '')
        title2 = (self.table2.get_table_name()).replace('csv', '')

        product_res = list(product(self.table1.get_rows(), self.table2.get_rows()))
        temp_keys = list()
        if self.table1.get_table_name() <= self.table2.get_table_name():
            temp_keys = [title1 + str(key1) for key1 in t1_keys] + [title1 + str(key2) for key2 in t2_keys]
        else:
            temp_keys = [title2 + str(key2) for key2 in t2_keys] + [title1 + str(key1) for key1 in t1_keys]

        self.m_rows = list()
        for pair in product_res:
            if pair[0][pair[0].get_primary_key()] == pair[1][pair[1].get_primary_key()]:
                if self.table1.get_table_name() <= self.table2.get_table_name():
                    temp_values = [pair[0][key1] for key1 in pair[0]] + [pair[1][key2] for key2 in pair[1]]
                else:
                    temp_values = [pair[1][key1] for key1 in pair[1]] + [pair[0][key2] for key2 in pair[0]]
                self.m_rows.append(Row(temp_keys, temp_values, title1 + self.table1.get_primary_key()))

        temp_res = Table(self.mquery['from'], self.m_rows, temp_keys, title1 + self.table1.get_primary_key())
        res = Query(self.mquery, temp_res)
        return res.as_table()

    """
    The `JoinQuery` class
    """


class AggQuery(Query):
    """
    The `AggQuery` class
    """

    def __init__(self, query):
        filename = query['from']

        operator_list = list()
        aim_key_list = list()
        new_keys = list()
        for agg in query['select']:
            if re.findall(r'MIN|MAX|AVG|SUM', agg):
                operator = re.findall(r'MIN|MAX|AVG|SUM', agg)[0]
                aim_key = re.findall(r'[(](.*?)[)]', agg)[0]
                new_key = agg
                operator_list.append(operator)
                aim_key_list.append(aim_key)
                new_keys.append(new_key)
        agg_dict = dict(zip(operator_list, aim_key_list))

        temp_query = copy.deepcopy(query)
        temp_query['select'] = Table(query['from']).keys()
        useful_table = Query(temp_query).as_table()

        if 'group_by' in list(query.keys()):
            key_ = query['group_by']
            temp = Table(filename, useful_table.get_rows(), useful_table.keys(), primary_key=key_)
            temp_table = copy.deepcopy(temp)
        else:
            temp = Table(filename, useful_table.get_rows(), useful_table.keys(),
                         primary_key=useful_table.get_primary_key())
            temp_table = copy.deepcopy(temp)

        for key, value in zip(new_keys, agg_dict):
            if 'group_by' in list(query.keys()):
                key_ = query['group_by']
                for i in range(len(temp_table.get_rows())):
                    calcul_res = temp_table.calculate(value, agg_dict[value], key_, temp_table.get_row(i)[key_])
                    temp_table.add_keys(key, calcul_res, i)
                    temp_table.add_key(key)
            else:
                calcul_res = temp_table.calculate(value, agg_dict[value])
                temp_table.add_keys(key, calcul_res)
        self.__table = copy.deepcopy(temp_table)
        self.__table = Query(query, table=self.__table).as_table()
        self.__table.de_repeat()

    def as_table(self):
        res = copy.deepcopy(self.__table)

        return res
