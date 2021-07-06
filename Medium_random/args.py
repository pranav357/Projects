# args and kwargs Unpacking operators, unpack values from iterable objects in python
# * for any iterable and ** for dictionaries

#Positional vs keyword argument
#Positional arguments are declared by a name only
#Keyword arguments are declared by a name and a default value
#Keyword arguments  must always follow after positional arguments(including args)
def addition(a, b=2): #a is positional, b is keyword argument
   return a + b

#args
nums = [1,2,3]
alphs = ['a', 'b', 'c']

nums_d = {1: 'one', 2: 'two', 3: 'three'}
alphs_d = {'a': 'First', 'b': 'Second', 'c' : 'Third'}

print(nums)#Prints the list
print(*nums)#Unpacks and prints values

#Using with functions
ex = 'Args and Kwargs'
print(*ex)#Unpacks even strings
print([*ex])#Convert to list

def sum_of_nums(n1, n2, n3):
    print(n1 + n2 + n3)

sum_of_nums(*nums)#Unpacks and passes to function
#allow a function to take any number of positional arguments

def concat_str(*args):
    res = ''
    for s in args:
        res += s
    print(res)

concat_str(*alphs)
#Unpacking operators allow us to use and manipulate as we need the individual element in any iterable

#Pass a combination of positional arguments and args
def arg_printer(a, b, *args):
   print(f'a is {a}')
   print(f'b is {b}')
   print(f'args are {args}')

arg_printer(3, 4, 5, 8, 3)

#kwargs
#**kwargs stand for keyword arguments and used with dictionaries
#allow a function to take any number of keyword arguments
#By defaults its an empty dictionary and each undefined keyword argument is stored as a key-value pair in the kwargs dictionary
def concat_str_2(**kwargs):
    result = ''
    for arg in kwargs:
        result += arg
    return result

print(concat_str_2(**alphs_d))#Concats the keys of the dictionary

# concatenating the values of the kwargs dictionary
def concat_str_3(**kwargs):
    result = ''
    for arg in kwargs.values():
        result += arg
    return result

print(concat_str_3(**alphs_d))#Concats the values of the dictionary

alphanum = {**nums_d, **alphs_d}#Concat two dictionaries
print(alphanum)

concat_str_3(a = 'Merge', b = ' this ', c = "dictionary's values")#Taken as key value pairs and concats them

# correct order of arguments in function
def my_cool_function(a, b, *args, **kwargs):
    '''my cool function body'''

from loguru import logger
@logger.catch

import snoop
@snoop

import heartrate 
heartrate.trace(browser=True)

    for _ in range(N):
        command, *value = input().split()
        if command !="print":
            getattr(lis,command)(*(map(int, value)))
        else:
            print(lis)

