## Feel free to import handy built-in modules here
import re
import copy


## Self-defined exceptions
## Please **DO NOT MODIFITY** the following exceptions!
class MatrixKeyError(Exception):
    pass


class MatrixValueError(Exception):
    pass


class MatrixSizeError(Exception):
    pass


class ObjectError(Exception):
    pass


class SizeUnmatchedError(Exception):
    pass


class ScalarTypeError(Exception):
    pass


class NotSquaredError(Exception):
    pass


class NotInvertibleError(Exception):
    pass


class IndexSyntaxError(Exception):
    pass


## Feel free to define your own functions here


## Your implementations start here
class Matrix:
    def __init__(self, m, n, expr):
        '''
        Task 2.1: Load the Matrix in
        Input:
            m: postive integer
            n: postive integer
            expr: string
        Output:
            No output for this task
        '''

        '''SizeError'''
        if (not isinstance(m, int)) or (not isinstance(n, int)) or m <= 0 or n <= 0:
            raise MatrixSizeError

        matrix = [[0] * n for i in range(m)]

        '''Process key list is totally empty eg: [2; ]'''
        expr = re.sub(r';\s*\]', '; [;]', expr)

        '''Process keys list is [;] or [ ; ]'''
        expr = expr.replace('[;]', '[(pass)]').replace('[ ; ]', '[(pass)]').replace('[; ]', '[(pass)]').replace('[ ;]',
                                                                                                                '[(pass)]').replace(
            '[]', '[(pass)]').replace('[ ]', '[(pass)]').replace('[  ]', '[(pass)]')

        '''Get Values: !!!can't get values in strange data type (like str...)'''
        values = re.findall(r'\[+\-?\d+\.?\d*e?[-+]?\d*\;', expr)

        '''Check whether exists [value '''
        for i in range(len(values)):
            temp = re.findall(r'\[', values[i])
            if (len(temp) != 1 and i != 0) or (len(temp) != 2 and i == 0):
                raise MatrixValueError

        values = [value.strip(';').strip('[') for value in values]

        '''ValueError0: Value like 5+4 or 5-4 or +5'''
        for value in values:
            if re.fullmatch(r'\d*\++\d*', value) or re.fullmatch(r'\d+\-+\d*', value):
                raise MatrixValueError

        '''Change value string into int or float'''
        for i in range(len(values)):
            if re.fullmatch(r'\-?\d+', values[i]):
                values[i] = int(values[i])
            else:
                values[i] = float(values[i])

        '''Get Keys'''
        keys = re.findall(r'\(.*\)', expr)
        for i in range(len(keys)):
            keys[i] = keys[i].split(';')
            for j in range(len(keys[i])):
                keys[i][j] = keys[i][j].replace('(', '').replace(')', '').replace(' ', '').replace('\n', '').replace(
                    '\t', '').replace('\r', '')
                keys[i][j] = keys[i][j].split(',')

        '''ValueError1:Strange Data Type'''
        if len(keys) > len(values):
            raise MatrixValueError
        if len(keys) < len(values):
            raise MatrixKeyError

        '''KeyError1&&2: !!!not include out of range'''
        for row in keys:
            for location in row:
                '''Doesn't have 2 elements or (have one element but the element isn't pass)'''
                if len(location) > 2 or len(location) < 1 or (len(location) == 1 and location[0] != 'pass'):
                    raise MatrixKeyError
                else:
                    for char in location:
                        '''Strange Data Type'''
                        if not re.fullmatch(r'\-?\d+', char) and char != 'pass':
                            raise MatrixKeyError

        '''Translate key's string into int or float'''
        for i in range(len(keys)):
            '''Process pass'''
            for j in range(len(keys[i])):
                if keys[i][j][0] == 'pass':
                    keys[i].clear()
                    continue

                if re.fullmatch(r'\-?\d+', keys[i][j][0]):
                    keys[i][j][0] = int(keys[i][j][0])
                else:
                    keys[i][j][0] = float(keys[i][j][0])

                if re.fullmatch(r'\-?\d+', keys[i][j][1]):
                    keys[i][j][1] = int(keys[i][j][1])
                else:
                    keys[i][j][1] = float(keys[i][j][1])

                location = (keys[i][j][0], keys[i][j][1])
                keys[i][j] = location

        '''Change Matrix and KeyError3:Out Of Range'''
        for i in range(len(values)):
            value = values[i]
            for j in range(len(keys[i])):
                loc_row = keys[i][j][0]
                loc_col = keys[i][j][1]
                if (-1 >= loc_row >= -m or 0 <= loc_row < m) and (-1 >= loc_col >= -n or 0 <= loc_col < n):
                    matrix[loc_row][loc_col] = value
                else:
                    raise MatrixKeyError

        self.matrix = matrix
        self.size=(m,n)


    def __str__(self):
            '''
            Task 2.2: Print out the Matrix
            Input:
                No input for this task
            Output:
                expr: string
            '''
            res = "["
            for i in range(len(self.matrix)):
                for j in range(len(self.matrix[i])):
                    temp = str(self.matrix[i][j])
                    '''Float should keep 5 digits'''
                    if isinstance(self.matrix[i][j], float):
                        temp = '%.5f' % self.matrix[i][j]
                    res += temp
                    if j != len(self.matrix[i]) - 1:
                        res += ','
                if i != len(self.matrix) - 1:
                    res += ';'
            res += ']'

            return res

    def __add__(self, other):
        '''
		Task 3.1: Matrix Addition and Subtraction
		Input:
			other: class Matrix
		Output:
			res: class Matrix
		'''

        '''ObjectError'''
        if not isinstance(other, Matrix):
            raise ObjectError

        '''SizeUnmatchedError'''
        if self.size != other.size:
            raise SizeUnmatchedError

        res = Matrix(self.size[0], self.size[1], '')

        '''Calculation'''
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                res.matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]

        return res

    def __sub__(self, other):
        '''
		Task 3.1: Matrix Addition and Subtraction
		Input:
			other: class Matrix
		Output:
			res: class Matrix
		'''

        '''ObjectError'''
        if not isinstance(other, Matrix):
            raise ObjectError

        '''SizeUnmatchedError'''
        if self.size != other.size:
            raise SizeUnmatchedError

        res = Matrix(self.size[0], self.size[1], '')

        '''Calculation'''
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                res.matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]

        return res

    def __mul__(self, other):
        '''
		Task 3.2: Matrix Multiplication
		Input:
			other: class Matrix or scalar (int or float)
		Output:
			res: class Matrix
		'''

        '''ObjectError'''
        if (not isinstance(other, Matrix)) and (not isinstance(other, int)) and (not isinstance(other, float)):
            raise ObjectError

        '''Matrix Mult'''
        if isinstance(other, Matrix):
            '''SizeUnmatchedError'''
            if self.size[1] != other.size[0]:
                raise SizeUnmatchedError

            '''Calculation'''
            res = Matrix(self.size[0], other.size[1], '')

            for i in range(self.size[0]):
                for j in range(other.size[1]):
                    for k in range(other.size[0]):
                        res.matrix[i][j] += (self.matrix[i][k] * other.matrix[k][j])
            return res
        else:
            '''Scalar Mult'''
            res = Matrix(self.size[0], self.size[1], '')
            for i in range(self.size[0]):
                for j in range(self.size[1]):
                    res.matrix[i][j] = other * self.matrix[i][j]
            return res

    def __truediv__(self, other):
        '''
		Task 3.3: Matrix Scalar Division
		Input:
			other: scalar (int or float)
		Output:
			res: class Matrix
		'''

        '''ObjectError'''
        if (not isinstance(other, int)) and (not isinstance(other, float)):
            raise ObjectError

        '''Calculation: What Error Should be raised when other equals to 0???'''
        res = Matrix(self.size[0], self.size[1], '')
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                res.matrix[i][j] = self.matrix[i][j] / other
        return res

    def __eq__(self, other):
        '''
		Task 3.4: Matrix Equalization
		Input:
			other: class Matrix
		Output:
			isEqual: bool
		'''

        '''Object Error'''
        if not isinstance(other, Matrix):
            raise ObjectError

        '''Calculation'''
        res = False
        if self.matrix == other.matrix:
            res = True
        return res

    def transpose(self):
        '''
		Task 3.4: Matrix Transpose
		Input:
			No input for this task
		Output:
			res: class Matrix
		'''

        '''Calculation: Not change self'''
        temp = zip(*self.matrix)
        res_matrix = [list(pair) for pair in temp]
        res = Matrix(len(res_matrix), len(res_matrix[0]), '')
        res.matrix = res_matrix
        return res

    def det(self):
        '''
		Task 4.1: Matrix Determinant
		Input:
			No input for this task
		Output:
			det: float
		'''

        '''NotSquaredError'''
        if self.size[0] != self.size[1]:
            raise NotSquaredError

        '''If size equals to 1'''
        if self.size[0] == 1 and self.size[1] == 1:
            return self.matrix[0][0]

        '''Calculation:recursion'''
        if len(self.matrix) == 1:
            return self.matrix
        else:
            res = 0
            for i in range(len(self.matrix)):
                '''nxt is Residual form'''
                nxt_list = [[row[a] for a in range(len(self.matrix)) if a != i] for row in self.matrix[1:]]
                nxt_matrix = Matrix(len(nxt_list), len(nxt_list[0]), '')
                nxt_matrix.matrix = nxt_list
                if i % 2 == 0:
                    res += self.matrix[0][i] * nxt_matrix.det()
                else:
                    res -= self.matrix[0][i] * nxt_matrix.det()
            return res

    def adj(self):
        row_num = self.size[0]
        if row_num == 1:
            return [[1]]
        res = []
        for i in range(row_num):
            temp_list = []
            res.append(temp_list)
            for j in range(row_num):
                _list = copy.deepcopy(self.matrix)
                _list = [_l for _l in _list if _list.index(_l) != i]
                [_l.pop(j) for _l in _list]
                _det = Matrix(len(_list), len(_list), '')
                _det.matrix = _list
                _det = _det.det()
                temp_list.append(((-1) ** (i + j)) * _det)
        _matrix = Matrix(row_num, row_num, '')
        _matrix.matrix = res

        return _matrix.transpose()

    def inv(self):
        '''
		Task 4.1: Matrix Determinant
		Input:
			No input for this task
		Output:
			mat_inv: class Matrix
		'''

        '''NotSquaredError'''
        if self.size[0] != self.size[1]:
            raise NotSquaredError

        '''NotInvertibleError'''
        if self.det() == 0:
            raise NotInvertibleError

        '''Calculation'''
        _det = self.det()
        _adj = self.adj()
        return _adj / _det

    def __getitem__(self, key):
        '''
		Task 5: Matrix Indexing
		Input:
			key: integer or tuple
		Output:
			value: scalar (int or float) or class Matrix
		'''

        if isinstance(key, tuple):
            '''GET an element'''
            if not isinstance(key[0], slice):
                '''IndexSyntaxError1'''
                if (not isinstance(key[0], int)) or (not isinstance(key[1], int)):
                    raise IndexSyntaxError

                '''IndexSyntaxError2'''
                if (key[0] >= self.size[0] and key[0] >= 0) or (key[0] < -self.size[0] and key[0] < 0) or (
                        key[1] >= self.size[1] and key[1] >= 0) or (key[1] < -self.size[1] and key[1] < 0):
                    raise IndexSyntaxError

                return self.matrix[key[0]][key[1]]

            # Get Slice
            elif isinstance(key[0], slice) and isinstance(key[1], slice):
                start1 = key[0].start
                if start1 is None:
                    start1 = 0

                stop1 = key[0].stop
                if stop1 is None:
                    stop1 = self.size[0]

                step1 = key[0].step
                if step1 is None:
                    step1 = 1

                start2 = key[1].start
                if start2 is None:
                    start2 = 0

                stop2 = key[1].stop
                if stop2 is None:
                    stop2 = self.size[1]

                step2 = key[1].step
                if step2 is None:
                    step2 = 1

                loc_row = list(range(start1, stop1, step1))
                loc_col = list(range(start2, stop2, step2))

                '''IndexSyntaxError'''
                if (step1 > 0 and start1 > stop1) or (step2 > 0 and start2 > stop2) or (
                        step1 < 0 and start1 < stop1) or (step2 < 0 and start2 < stop2):
                    raise IndexSyntaxError

                res = Matrix(len(loc_row), len(loc_col), '')
                for i in range(len(loc_row)):
                    for j in range(len(loc_col)):
                        res[i, j] = self[start1 + i * step1, start2 + j * step2]

                return res

        # Get a row
        elif isinstance(key, int):
            '''IndexSyntaxError1'''
            if (key >= self.size[0] and key >= 0) or (key < -self.size[0] and key < 0):
                raise IndexSyntaxError

            res = Matrix(1, self.size[1], '')
            res.matrix = [self.matrix[key]]

            return res
        else:
            raise IndexSyntaxError

    def __setitem__(self, key, value):
        '''
		Task 5: Matrix Indexing
		Input:
			key: integer or tuple
			value: scalar (int or float) or class Matrix
		Output:
			No output for this task
		'''

        if isinstance(key, tuple):

            '''SET an element'''
            if not isinstance(key[0], slice):
                '''MatrixValueError'''
                if not isinstance(value, int) and not isinstance(value, float):
                    raise MatrixValueError

                '''Check Key'''
                check = self.__getitem__(key)

                self.matrix[key[0]][key[1]] = value
            # SET Slice
            elif isinstance(key[0], slice) and isinstance(key[1], slice):
                '''MatrixValueError'''
                if not isinstance(value, Matrix):
                    raise MatrixValueError
                else:
                    for i in range(value.size[0]):
                        for j in range(value.size[1]):
                            if (not isinstance(value[i, j], int)) and (not isinstance(value[i, j], float)):
                                raise MatrixValueError

                '''Check Key'''
                check = self.__getitem__(key)

                start1 = key[0].start
                if start1 is None:
                    start1 = 0

                stop1 = key[0].stop
                if stop1 is None:
                    stop1 = self.size[0]

                step1 = key[0].step
                if step1 is None:
                    step1 = 1

                start2 = key[1].start
                if start2 is None:
                    start2 = 0

                stop2 = key[1].stop
                if stop2 is None:
                    stop2 = self.size[1]

                step2 = key[1].step
                if step2 is None:
                    step2 = 1

                loc_row = list(range(start1, stop1, step1))
                loc_col = list(range(start2, stop2, step2))

                '''IndexSyntaxError'''
                if (value.size[0] != len(loc_row)) or (value.size[1] != len(loc_col)):
                    raise IndexSyntaxError

                for i in range(value.size[0]):
                    for j in range(value.size[1]):
                        self[start1 + i * step1, start2 + j * step2] = value[i, j]
        # Set A ROW
        elif isinstance(key, int):
            '''MatrixValueError'''
            if not isinstance(value, Matrix):
                raise MatrixValueError
            else:
                for i in range(value.size[0]):
                    for j in range(value.size[1]):
                        if (not isinstance(value[i, j], int)) and (not isinstance(value[i, j], float)):
                            raise MatrixValueError

            '''Check Key'''
            self.__getitem__(key)

            '''IndexSyntaxError'''
            if (value.size[1] != self.size[1]) or (value.size[0] != 1):
                raise IndexSyntaxError

            for i in range(self.size[1]):
                self[key, i] = value[0, i]
        else:
            raise IndexSyntaxError


## Self-defined methods
## Feel free to define your own methods for class Matrix here

class KF:
    def predict(self, x_pre, P_pre, u_k, F, B, Q):
        '''
		Task 6.1: Prediction
		Input:
			x_pre: n*1 Matrix
			P_pre: n*n Matrix
			u_k: m*1 Matrix
			F: n*n Matrix
			B: n*m Matrix
			Q: n*n Matrix
		Output: 
			x_predicted: n*1 Matrix
			P_predicted: n*n Matrix
		'''

        '''Calculation'''
        # step1
        x_predicted = F * x_pre + B * u_k
        P_predicted = F * P_pre

        # step2
        F_transpose = F.transpose()
        P_predicted = P_predicted * F_transpose + Q

        return x_predicted, P_predicted

    def update(self, x_pre, P_pre, z_k, H, R):
        '''
		Task 6.2: Update
		Input:
			x_pre: n*1 Matrix
			P_pre: n*n Matrix
			z_k: k*1 Matrix
			H: k*n Matrix
			R: k*k Matrix
		Output: 
			x_updated: n*1 Matrix
			P_updated: n*n Matrix
		'''

        '''Calculation'''
        # Step1
        H_transpose = H.transpose()
        inv_element = H * P_pre
        inv_element = inv_element * H_transpose + R
        inv_element = inv_element.inv()

        K = P_pre * H_transpose
        K = K * inv_element

        # Step2
        x_updated = x_pre + K * (z_k - H * x_pre)

        # Step3
        P_updated = K * H
        P_updated = P_pre - P_updated * P_pre

        return x_updated, P_updated


def postfix_eval(expr):
    '''
	Task 7 (Bonus Task): Matrix Postfix Evaluation 
	Input:
		expr: list
	Output:
		res: class Matrix
	'''

    i = int()

    while i < len(expr):
        if not isinstance(expr[i], Matrix):
            if expr[i] == '+':
                res = expr[i - 2] + expr[i - 1]
                expr = expr[i + 1:]
                expr.insert(0, res)
                i = 0
            elif expr[i] == '-':
                res = expr[i - 2] - expr[i - 1]
                expr = expr[i + 1:]
                expr.insert(0, res)
                i = 0
            elif expr[i] == '*':
                res = expr[i - 2] * expr[i - 1]
                expr = expr[i + 1:]
                expr.insert(0, res)
                i = 0
            elif expr[i] == '/':
                res = expr[i - 2] / expr[i - 1]
                expr = expr[i + 1:]
                expr.insert(0, res)
                i = 0

        i += 1

    return expr[0]


if __name__ == "__main__":

    ## Tests for Task 2
    ## normal test
    test_m1 = 5
    test_n1 = 5
    test_expr1 = \
        '''[[2; [(1, 2); (1, 0)]]; 
			[3; [(0, 2); (1, 1); (2, 0)]]; 
			[4; [(0, 1); (2, 1)]];
			[5; [(2, 2)]]]'''

    mat1 = Matrix(test_m1, test_n1, test_expr1)
    mat1_ans = str(mat1)

    if mat1_ans == '[0,4,3,0,0;2,3,2,0,0;3,4,5,0,0;0,0,0,0,0;0,0,0,0,0]':
        print('Pass normal test of Task 2.')
    else:
        print('Fail normal test of Task 2.')

    ## exception tests
    ## MatrixSizeError test
    test_m1_err = 3.0
    try:
        mat1 = Matrix(test_m1_err, test_n1, test_expr1)
    except MatrixSizeError:
        print('Pass MatrixSizeError test of Task 2.')
    except:
        print('Fail MatrixSizeError test of Task 2.')
    else:
        print('Fail MatrixSizeError test of Task 2.')

    ## MatrixKeyError test
    test_expr1_err1 = \
        '''[[2; [(0, 1); (1, 0)]]; 
			[3; [(0, 2); (1, 1); (2, 0)]]; 
			[4; [(1, 2.0); (2, 1)]];
			[5; [(2, 2)]]]'''
    try:
        mat1 = Matrix(test_m1, test_n1, test_expr1_err1)
    except MatrixKeyError:
        print('Pass MatrixKeyError test of Task 2.')
    except:
        print('Fail MatrixKeyError test of Task 2.')
    else:
        print('Fail MatrixKeyError test of Task 2.')

    ## MatrixValueError test
    test_expr1_err2 = \
        '''[[2; [(0, 1); (1, 0)]]; 
			[3; [(0, 2); (1, 1); (2, 0)]]; 
			[4+1j; [(1, 2); (2, 1)]];
			[5; [(2, 2)]]]'''
    try:
        mat1 = Matrix(test_m1, test_n1, test_expr1_err2)
    except MatrixValueError:
        print('Pass MatrixValueError test of Task 2.')
    except:
        print('Fail MatrixValueError test of Task 2.')
    else:
        print('Fail MatrixValueError test of Task 2.')

    ## Tests for Task 3
    add_res = mat1 + mat1
    sub_res = mat1 - mat1
    if str(add_res) == '[0,8,6,0,0;4,6,4,0,0;6,8,10,0,0;0,0,0,0,0;0,0,0,0,0]':
        print('Pass Addition Test.')
    else:
        print('Fail Addition Test')

    if str(sub_res) == '[0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0]':
        print('Pass Subtraction Test.')
    else:
        print('Fail Subtraction Test')

    mul_res1 = mat1 * mat1
    mul_res2 = mat1 * 1.5

    if str(mul_res1) == '[17,24,23,0,0;12,25,22,0,0;23,44,42,0,0;0,0,0,0,0;0,0,0,0,0]':
        print('Pass Multiplication Test 1.')
    else:
        print('Fail Multiplication Test 1.')

    if str(
            mul_res2) == '[0.00000,6.00000,4.50000,0.00000,0.00000;3.00000,4.50000,3.00000,0.00000,0.00000;4.50000,6.00000,7.50000,0.00000,0.00000;0.00000,0.00000,0.00000,0.00000,0.00000;0.00000,0.00000,0.00000,0.00000,0.00000]':
        print('Pass Multiplication Test 2.')
    else:
        print('Fail Multiplication Test 2.')

    div_res = mat1 / 1.5
    if str(
            div_res) == '[0.00000,2.66667,2.00000,0.00000,0.00000;1.33333,2.00000,1.33333,0.00000,0.00000;2.00000,2.66667,3.33333,0.00000,0.00000;0.00000,0.00000,0.00000,0.00000,0.00000;0.00000,0.00000,0.00000,0.00000,0.00000]':
        print('Pass Division Test.')
    else:
        print('Fail Division Test.')

    if not mat1 == div_res:
        print('Pass Equalization Test 1.')
    else:
        print('Fail Equalization Test 1.')

    if mat1 == mat1:
        print('Pass Equalization Test 2.')
    else:
        print('Fail Equalization Test 2.')

    mat1_2 = Matrix(3, 3, test_expr1)

    if str(mat1_2.transpose()) == '[0,2,3;4,3,4;3,2,5]':
        print('Pass Transpose Test.')
    else:
        print('Fail Transpose Test.')

    ## Test for Task 4
    if mat1_2.det() == -19:
        print('Pass Determinant Test.')
    else:
        print('Fail Determinant Test.')

    if str(mat1_2.inv()) == '[-0.36842,0.42105,0.05263;0.21053,0.47368,-0.31579;0.05263,-0.63158,0.42105]':
        print('Pass Inverse Test.')
    else:
        print('Fail Inverse Test.')

    ## Test for Task 5
    # print('mat1[1]:', mat1[1])
    if str(mat1[1]) == '[2,3,2,0,0]':
        print('Pass Test 1 for Task 5')
    else:
        print('Fail Test 1 for Task 5')

    if mat1[1, 1] == 3:
        print('Pass Test 2 for Task 5')
    else:
        print('Fail Test 2 for Task 5')

    if str(mat1[0:3, 1:4]) == '[4,3,0;3,2,0;4,5,0]':
        print('Pass Test 3 for Task 5')
    else:
        print('Fail Test 3 for Task 5')

    mat2 = Matrix(2, 3,
                  '''[[1; [(0, 0)]]; 
			[2; [(0, 1)]]; 
			[3; [(0, 2)]];
			[4; [(1, 0)]]; 
			[5; [(1, 1)]]; 
			[6; [(1, 2)]]]'''
                  )
    mat1[0:2, 1:4] = mat2
    if str(mat1) == '[0,1,2,3,0;2,4,5,6,0;3,4,5,0,0;0,0,0,0,0;0,0,0,0,0]':
        print('Pass Test 4 for Task 5')
    else:
        print('Fail Test 4 for Task 5')

    mat3 = Matrix(1, 5,
                  '''[[1; [(0, 0); (0, 1); (0, 2); (0, 3); (0, 4)]]]''')
    mat1[2] = mat3
    if str(mat1) == '[0,1,2,3,0;2,4,5,6,0;1,1,1,1,1;0,0,0,0,0;0,0,0,0,0]':
        print('Pass Test 5 for Task 5')
    else:
        print('Fail Test 5 for Task 5')

    mat1[3, 3] = 2.333
    if str(mat1) == '[0,1,2,3,0;2,4,5,6,0;1,1,1,1,1;0,0,0,2.33300,0;0,0,0,0,0]':
        print('Pass Test 6 for Task 5')
    else:
        print('Fail Test 6 for Task 5')

    ## Test for Task 6
    x_pre = Matrix(3, 1,
                   '''[[0; [(0, 0)]]; 
			[1; [(1, 0)]]; 
			[2; [(2, 0)]]]''')
    P_pre = Matrix(3, 3,
                   '''[[1; [(0, 0)]];
			[2; [(0, 1)]]; 
			[3; [(0, 2)]];
			[4; [(1, 0)]]; 
			[5; [(1, 1)]];
			[6; [(1, 2)]]; 
			[7; [(2, 0)]]; 
			[8; [(2, 1)]];
			[9; [(2, 2)]]]''')
    u = Matrix(3, 1,
               '''[[0.1; [(0, 0)]]; 
			[0.2; [(1, 0)]]; 
			[0.3; [(2, 0)]]]''')
    F = Matrix(3, 3,
               '''[[1; [(0, 0); (1, 1); (2, 2)]]]''')
    B = Matrix(3, 3,
               '''[[4; [(0, 0)]];
			[5; [(1, 1)]];
			[6; [(2, 2)]]]''')
    Q = F * 0.02
    z = Matrix(3, 1,
               '''[[1; [(0, 0)]]; 
			[3; [(1, 0)]]; 
			[4; [(2, 0)]]]''')
    H = F * 2
    R = F * 0.03

    kf = KF()
    x_predicted, P_predicted = kf.predict(x_pre, P_pre, u, F, B, Q)
    x_updated, P_updated = kf.update(x_predicted, P_predicted, z, H, R)

    test1 = str(x_predicted) == '[0.40000;2.00000;3.80000]'
    test2 = str(P_predicted) == '[1.02000,2.00000,3.00000;4.00000,5.02000,6.00000;7.00000,8.00000,9.02000]'
    test3 = str(x_updated) == '[0.53589;1.43717;2.02935]'
    test4 = str(P_updated) == '[0.00720,0.00069,-0.00036;0.00069,0.00614,0.00068;-0.00037,0.00067,0.00717]'

    if test1 and test2:
        print('Pass Prediction Test.')
    else:
        print('Fail Prediction Test.')

    if test3 and test4:
        print('Pass Update Test.')
    else:
        print('Fail Update Test.')

    A = Matrix(3, 3,
               '''[[1; [(0, 0)]];
			[2; [(0, 1); (1, 0)]]; 
			[3; [(0, 2); (1, 1); (2, 0)]];
			[4; [(1, 2); (2, 1)]]; 
			[5; [(2, 2)]]]''')
    B = 1.5
    C = Matrix(3, 3,
               '''[[1; [(0, 0); (1, 1); (2, 2)]]]''')
    D = Matrix(3, 3,
               '''[[1; [(0, 0); (1, 1); (2, 2)]]]''')

    test_postfix_expr = [A, B, '*', C, '+', D, '-']
    if postfix_eval(test_postfix_expr) == A * B + C - D:
        print('Pass Postfix Evaluation.')
    else:
        print('Fail Postfix Evaluation.')
