#!/usr/bin/env python


# ## DICTIONARIES
# # keys can be any hashable immutable type, such as strings or integers...or even some tuples. Usually we use strings.
# fruits_available = {"apples": 3, "oranges": 9, "bananas": 12, "guanabana": 0}
# print(fruits_available["apples"]) # accessing the value associated to the "apples" key

# # we can store dictionaries inside dictionaries, which are called nested dictionaries.
# fruits_nutrition = {"apple": {"calories" : 54, "water_percent" : 86, "fibre_grams" : 2.4}, 
#                     "orange" : {"calories" : 60, "water_percent" : 86, "fibre_grams" : 3.0}}
# print(fruits_nutrition["apple"]["calories"]) # note the two key levels

# # updating a value in a dictionary
# fruits_nutrition["apple"]["calories"] = 52
# print(fruits_nutrition["apple"]["calories"])

# # adding an item to the dictionary
# fruits_nutrition["banana"] = {"calories" : 89, "water_percent" : 75, "fibre_grams" : 2.6}
# print(fruits_nutrition) # notice our dictionary now has a new fruit
# # this is different for lists, where accessing an inexistent item would cause an error

# # deleting an item from the dictionary
# del fruits_nutrition['apple']
# print(fruits_nutrition.keys()) #list the keys or values

# # some dictionary methods
# print(fruits_nutrition.keys()) # list the keys or values
# print(fruits_nutrition["apple"].keys()) # for the nested dictionary
# print(fruits_nutrition["apple"].values())

# print(fruits_nutrition.get("apple"))    # alternative way to obtain a value from a key
# print(fruits_nutrition.get("blahblah")) # can test if entry exists without causing an error if it doesn't
# print(fruits_nutrition["blahblah"])     # KeyError


# ## SETS
# list_with_duplicates = [1, 2, 3, 1, 2, 3]
# unique_items = list(set(list_with_duplicates)) # cast to set, then back to list
# print(list_with_duplicates)
# print(unique_items)
