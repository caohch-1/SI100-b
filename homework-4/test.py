#=============================================================================#
#                             Homework 4: yourSQL                             #
#         SI 100B: Introduction to Information Science and Technology         #
#                     Spring 2020, ShanghaiTech University                    #
#                     Author: Diao Zihao <hi@ericdiao.com>                    #
#                         Last motified: 05/01/2020                           #
#=============================================================================#
# test.py - test your implementation.
# PLEASE NOTE: Teatcases here are different from those in the auto-grader. Only
# results of the testcases in the auto-grader will be considered valid and will
# count for your final score.

import unittest
import yoursql

# Feel Free to motify the testcases below to create your own testcases.


class TestTask0Row(unittest.TestCase):

    def setUp(self):
        self.row = yoursql.Row(['a', 'b', 'c'], [1, 2, 3])

    def testInit(self):
        self.assertEqual(self.row.get_primary_key(), 'a')
        self.assertEqual(self.row.keys(), ['a', 'b', 'c'])
        self.assertEqual([self.row[i] for i in self.row], [1, 2, 3])


class TestTask0Table(unittest.TestCase):

    def setUp(self):
        self.table = yoursql.Table('testcases/student.csv')

    def testInit(self):
        self.assertEqual(self.table.get_table_name(), 'testcases/student.csv')
        self.assertEqual(self.table[23456123]['name'], 'Li Xiaoming')


class TestTask1(unittest.TestCase):

    def setUp(self):
        self.q = {
            'select': ['id', 'name', 'school', 'major', 'gpa'],
            'from': 'testcases/student.csv',
            'where': [('id', 45280742, '=='), ]
        }

    def testInit(self):
        table = yoursql.Query(self.q).as_table()
        self.assertEqual(table[45280742]['gpa'], 3.76)


class TestTask2(unittest.TestCase):

    def setUp(self):
        self.q = {
            'select': ['testcases/student_name.id', 'testcases/student_name.school', 'testcases/student_name.name', 'testcases/student_name.major'],
            'from':  ['testcases/student_name.csv', 'testcases/student_gpa.csv'],
            'where': [('testcases/student_gpa.gpa', 4.0, '==')],
        }

    def testInit(self):
        table = yoursql.JoinQuery(self.q).as_table()
        self.assertEqual(table[12567923]['testcases/student_name.school'], 'SLST')


class TestTask3(unittest.TestCase):

    def setUp(self):
        self.table = yoursql.Table('testcases/student.csv')
        self.table[45280742]['gpa'] = 3.99

    def testInit(self):
        self.table.export('new_student.csv')
        new_table = yoursql.Table('new_student.csv', primary_key='id')
        self.assertEqual(new_table[45280742]['gpa'], 3.99)


class TestBonus(unittest.TestCase):

    def setUp(self):
        self.q = {
            'select': ['school', 'AVG(gpa)'],
            'from': 'testcases/student.csv',
            'group_by': 'school',
            'where': [],
        }

    def testInit(self):
        table = yoursql.AggQuery(self.q).as_table()
        self.assertEqual(table['SLST']['AVG(gpa)'], 4.00)


if __name__ == '__main__':
    unittest.main()
