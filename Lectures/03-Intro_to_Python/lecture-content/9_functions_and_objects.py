#!/usr/bin/env python

# ## FUNCTIONS
# def summing_two_nums(x, y): # make a new function by using def followed by the function name you will give it.
#     return x + y

# print(summing_two_nums(1, 4)) # call the function with inputs 1 and 4.
# print(summing_two_nums(3, 3)) # re-use the same function

# def appending_to_list(input_list, new_item):
#     input_list.append(new_item)
#     return input_list
    
# print(appending_to_list([1, 2, 3], 4))


# ## variable scope
# def my_function(my_variable):
#     my_variable = 'bar' # local variable with the same name as the global variable
#     for i in range(n): # accessing a global variable
#         print('my_variable inside the function: ' + my_variable)

# my_variable = 'foo'   # global variable
# n = 2
# print('my_variable outside the function: ' + my_variable)

# my_function(my_variable)

# print('my_variable outside the function again: ' + my_variable) # unchanged

# ## passing mutable objects to functions
# def change_list(my_list_inside):
#     my_list_inside.append([1,2,3,4]);
#     print("Values inside the function: ", my_list_inside)
#     return # note that we are not returning anything

# my_list = [10,20,30];
# change_list(my_list);
# print("Values outside the function: ", my_list) # changed


# ## importing libraries
# import math

# print(math.pi) # constant
# print(math.factorial(5))

# from math import factorial
# print(factorial(5))

# from IPython.display import YouTubeVideo #Importing the YouTubeVideo function from the IPython.display module
# YouTubeVideo("ml6VkmtLXpA",560,315,rel=0)

# ## OBJECTS
# class Dog:
#     # Class attribute - all objects
#     species = "Canis familiaris"
#     def __init__(self, name, age): #Specific to instance
#         self.name = name
#         self.age = age
#     def description(self):
#         return f"{self.name} is {self.age} years old"
        
# my_dog = Dog("Bonzo", 7)
# print(my_dog.name)
# print(my_dog.description())
# print(my_dog.species)
