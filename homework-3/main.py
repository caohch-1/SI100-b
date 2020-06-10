import math
import re

keywords = ['lambda', 'apply', 'pow',
            'neg', 'cos', 'sin', 'exp',
            'numi', 'numf']
others = ['+', '-', '*']
unary = ['neg', 'cos', 'sin', 'exp']
binary = ['+', '-', '*', 'pow', 'lambda']
type = ['numi', 'numf']


class Syntax:
    # implement your syntaxes here
    # ... or derive subclasses and implement there
    def charsToSyntaxes(self, symbols):
        # implement your conversion from symbols to syntaxes here
        temp = ''
        res = []
        patt = '[a-zA-Z][a-zA-Z0-9]*( |\\))'
        nums = re.findall(r"\d+\.?\d*", symbols)
        for i in range(len(symbols)):
            temp += symbols[i]
            if ((temp in keywords) and (symbols[i+1] == ' ') and symbols[pre] == '(') or temp in others:
                res.append(temp)
                if res[-1] in type:
                    res.append(nums[0])
                    nums.pop(0)
                temp = ''
                pre = i
            elif re.match(patt, temp):
                """???"""
                if temp[0:-1] in keywords:
                    res.append(temp[0:-1]+'_')
                else:
                    res.append(temp[0:-1])
                temp = ''
                pre = i
            elif temp in [' ', '.', '(', ')'] or temp.isdigit():
                temp = ''
                pre = i
        return res

    def __hash__(self):
        # implement your hash algorithm for the symbols here
        # you may make use of the default hash algorithms of strings, numbers, etc.
        pass

    def __eq__(self, rhs):
        # implement your syntax equality determination here
        pass


class EvaluationContext:
    # implement your evaluation context here
    def __init__(self, prev=None):
        # implement your evaluation context initialization here
        self.prev_table = prev
        self.table = dict()

    def store(self, name, value):
        # implement your variable store logic here
        self.table[name] = value

    def load(self, name):
        if name in self.table:
            return self.table[name]
        elif self.prev_table != None:
            return self.prev_table.load(name)
        else:
            return None
        # implement your variable retrieve logic here
        # return None if no name is associated with the name

    def push(self):
        # implement your context stacking logic here
        return EvaluationContext(self)

    def pop(self):
        # implement your context destacking logic here
        return self.prev_table


class ASTNode:
    # implement your AST Nodes here
    # ... or derive subclasses and give actual implementations there
    def createAST(self, syntaxes):
        self.children = []
        if (syntaxes[0] in unary) or (syntaxes[0] in type):
            self.name = syntaxes.pop(0)
            new_child = ASTNode()
            self.children.append(new_child.createAST(syntaxes))
        elif syntaxes[0] in binary:
            self.name = syntaxes.pop(0)
            left_child = ASTNode()
            self.children.append(left_child.createAST(syntaxes))
            right_child = ASTNode()
            self.children.append(right_child.createAST(syntaxes))
        elif syntaxes[0] == 'apply':
            self.name = syntaxes.pop(0)
            left_child = ASTNode()
            self.children.append(left_child.createAST(syntaxes))
            mid_child = ASTNode()
            self.children.append(mid_child.createAST(syntaxes))
            right_child = ASTNode()
            self.children.append(right_child.createAST(syntaxes))
        elif re.match('[a-zA-Z][a-zA-Z0-9]*', syntaxes[0]):
            self.name = syntaxes.pop(0)
        elif re.fullmatch(r"\d+\.?\d*", syntaxes[0]):
            self.name = syntaxes.pop(0)
        return self

    def evaluate(self, eval_context=EvaluationContext(None)):
        # implement your AST Node evaluation according to given context here
        if self.name == 'sin':
            res = self.children[0].evaluate(eval_context)
            if res.name == 'numi':
                self.name = 'numf'
                self.children[0].name = math.sin(
                    int(self.children[0].children[0].name))
                self.children[0].children.clear()
            elif res.name == 'numf':
                self.name = 'numf'
                self.children[0].name = math.sin(
                    float(self.children[0].children[0].name))
                self.children[0].children.clear()
        elif self.name == 'cos':
            res = self.children[0].evaluate(eval_context)
            if res.name == 'numi':
                self.name = 'numf'
                self.children[0].name = math.cos(
                    int(self.children[0].children[0].name))
                self.children[0].children.clear()
            elif res.name == 'numf':
                self.name = 'numf'
                self.children[0].name = math.cos(
                    float(self.children[0].children[0].name))
                self.children[0].children.clear()
        elif self.name == 'exp':
            res = self.children[0].evaluate(eval_context)
            if res.name == 'numi':
                self.name = 'numf'
                self.children[0].name = math.exp(
                    int(self.children[0].children[0].name))
                self.children[0].children.clear()
            elif res.name == 'numf':
                self.name = 'numf'
                self.children[0].name = math.exp(
                    float(self.children[0].children[0].name))
                self.children[0].children.clear()
        elif self.name == 'neg':
            res = self.children[0].evaluate(eval_context)
            if res.name == 'numi':
                self.name = 'numi'
                self.children[0].name = -int(self.children[0].children[0].name)
                self.children[0].children.clear()
            elif res.name == 'numf':
                self.name = 'numf'
                self.children[0].name = - \
                    float(self.children[0].children[0].name)
                self.children[0].children.clear()
        elif self.name == '+':
            resl = self.children[0].evaluate(eval_context)
            resr = self.children[1].evaluate(eval_context)
            if resl.name in ['numi', 'numf'] and resr.name in ['numi', 'numf']:
                if resl.name == 'numi' and resr.name == 'numi':
                    self.name = 'numi'
                    self.children[0].name = int(
                        self.children[0].children[0].name) + int(self.children[1].children[0].name)
                    self.children[0].children.clear()
                    self.children.pop()
                else:
                    self.name = 'numf'
                    self.children[0].name = float(
                        self.children[0].children[0].name) + float(self.children[1].children[0].name)
                    self.children[0].children.clear()
                    self.children.pop()
        elif self.name == '-':
            resl = self.children[0].evaluate(eval_context)
            resr = self.children[1].evaluate(eval_context)
            if resl.name in ['numi', 'numf'] and resr.name in ['numi', 'numf']:
                if resl.name == 'numi' and resr.name == 'numi':
                    self.name = 'numi'
                    self.children[0].name = int(
                        self.children[0].children[0].name) - int(self.children[1].children[0].name)
                    self.children[0].children.clear()
                    self.children.pop()
                else:
                    self.name = 'numf'
                    self.children[0].name = float(
                        self.children[0].children[0].name) - float(self.children[1].children[0].name)
                    self.children[0].children.clear()
                    self.children.pop()
        elif self.name == '*':
            resl = self.children[0].evaluate(eval_context)
            resr = self.children[1].evaluate(eval_context)
            if resl.name in ['numi', 'numf'] and resr.name in ['numi', 'numf']:
                if resl.name == 'numi' and resr.name == 'numi':
                    self.name = 'numi'
                    self.children[0].name = int(
                        self.children[0].children[0].name) * int(self.children[1].children[0].name)
                    self.children[0].children.clear()
                    self.children.pop()
                else:
                    self.name = 'numf'
                    self.children[0].name = float(
                        self.children[0].children[0].name) * float(self.children[1].children[0].name)
                    self.children[0].children.clear()
                    self.children.pop()
        elif self.name == 'pow':
            resl = self.children[0].evaluate(eval_context)
            resr = self.children[1].evaluate(eval_context)
            if resl.name in ['numi', 'numf'] and resr.name in ['numi', 'numf']:
                if resl.name == 'numi' and resr.name == 'numi':
                    self.name = 'numi'
                    self.children[0].name = pow(int(self.children[0].children[0].name), int(
                        self.children[1].children[0].name))
                    self.children[0].children.clear()
                    self.children.pop()
                else:
                    self.name = 'numf'
                    self.children[0].name = pow(float(self.children[0].children[0].name), float(
                        self.children[1].children[0].name))
                    self.children[0].children.clear()
                    self.children.pop()
        elif self.name == 'apply':
            resl = self.children[0]
            resm = self.children[1].evaluate(eval_context)
            eval_context.push()
            eval_context.store(resl.name, [resm.name, resm.children[0].name])
            self.children[2].evaluate(eval_context)
            self.name = self.children[2].name
            self.children = self.children[2].children
            eval_context.pop()
        elif self.name == 'lambda':
            resl = self.children[0]
            loadname = eval_context.load(resl.name)
            self.children[1].evaluate(eval_context)
            if loadname != None:
                self.name = self.children[1].name
                self.children = self.children[1].children
        elif self.name == 'numi':
            self.children[0].name = int(self.children[0].name)
        elif self.name == 'numf':
            self.children[0].name = float(self.children[0].name)
        elif re.match('[a-zA-Z][a-zA-Z0-9_]*', self.name):
            loadname = eval_context.load(self.name)
            if loadname != None:
                self.name = loadname[0]
                new_node = ASTNode()
                new_node.name = loadname[1]
                self.children = [new_node]
        return self

    def outPut(self, res):
        keywords = ['pow',
                    'neg', 'cos', 'sin', 'exp',
                    'numi', 'numf', 'lambda', 'apply']
        if (isinstance(self.name, int) or isinstance(self.name, float) or re.match('[a-zA-Z][a-zA-Z0-9_]*', self.name) and self.name not in keywords) or re.match(r'\d+\.?\d*', self.name):
            if isinstance(self.name, float):
                res += '{0:.5f}'.format(self.name)
            elif not isinstance(self.name, int) and self.name[-1] == '_':
                res += str(self.name[:-1])
            else:
                res += str(self.name)
        else:
            res += '('
            res += self.name+' '
            for i in range(len(self.children)):
                self.children[i].outPut(res)
                if i != len(self.children)-1:
                    res += ' '
            res += ')'


class AST:
    # implement your AST here
    def syntaxesToAST(self, syntaxes):
        ast = AST()
        ast.root = ASTNode()
        ast.root.createAST(syntaxes)
        return ast

    def evaluate(self, eval_context):
        # implement your AST evaluation according to given context here
        self.root.evaluate(eval_context)
        self.root.evaluate(eval_context)
        return self


class Evaluator:
    # implement your evaluator here
    def getInputAsChars(self):
        with open('input.txt', 'r') as f:
            return f.read()
        # retrieve the input as characters from the input file here
        # ... generator is greatly recommended.

    def evaluate(self):
        chars = self.getInputAsChars()
        syntaxes = Syntax().charsToSyntaxes(chars)
        ast = AST().syntaxesToAST(syntaxes)
        ec = EvaluationContext(None)
        return ast.evaluate(ec)

    def stringifyResult(self, result):
        # implement your result stringify logic here
        res = []
        result.root.outPut(res)
        return ''.join(res)

    def writeOutput(self, s):
        # store your output to required file here
        with open('output.txt', 'w') as f:
            f.write(s)


if __name__ == "__main__":
    evaluator = Evaluator()
    result = evaluator.evaluate()
    s = evaluator.stringifyResult(result)
    evaluator.writeOutput(s)
