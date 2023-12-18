#!/usr/bin/env python

# ## WHILE LOOPS
# i = 1 #initialize our counter
# while i < 6:
#     print(i)
#     i += 1 #note that we are incrementing our counter variable by 1 every time. i += 1 is the same as i = i + 1

# # use the break command to exit the loop - The break statement terminates the loop containing it
# i = 1
# while i < 6:
#     print(i)
#     if i == 3: # exit the loop when i takes the value of 3
#         break
#     i += 1

# # Using a while loop to iterate over a list
# my_list = ["orange", "apples", "bananas"]
# x = 0
# while x < len(my_list):
#     print(my_list[x])
#     x += 1 # increment the index


# ## FOR LOOPS
# my_list = ["orange", "apples", "bananas"]
# for x in my_list:
#     print(x)

# for y in range(3): # in range(n) - from 0 to n-1, so here it's from from 0 to 2
#     print(y)

# for y in range(3, 13, 3): # from 3 to 12, in steps of 3
#     print(y)

# for character in "string": # loop over a string's characters.
#     print(character)

# # iterating over a dictionary
# fruits_nutrition = {"apple": {"calories" : 54, "water_percent" : 86, "fibre_grams" : 2.4},
#                     "orange" : {"calories" : 60, "water_percent" : 86, "fibre_grams" : 3.0},
#                     "banana" : {"calories" : 89, "water_percent" : 75, "fibre_grams" : 2.6}}

# for key in fruits_nutrition: # loop over the keys
#     print(key)

# for item in fruits_nutrition.items(): # loop over keys and values
#     print(item)

# for key, value in fruits_nutrition.items(): # have access to both keys and values as you loop (unpack the tuple)
#     print(key, "-->", value["calories"])

# # nested loops
# my_list = ["orange", "apples", "bananas"]

# for item in my_list:
#     x = 1
#     while x <= 3: # note that the inner 'nested' loop has to finish before the next iteration of the outer loop.
#         print(item) # notice the double indentation below.
#         x += 1

# # Worth mentioning that a break statement inside a nested loop only terminates the inner loop

# ## LIST COMPREHENSION
# my_list = ["orange", "apples", "bananas"]

# new_my_list = [item for item in my_list if item[-1] == 's']
# print(new_my_list)
