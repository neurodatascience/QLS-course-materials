#!/usr/bin/env python

# ## LISTS
# my_list = [1, 2, 345, 42]
# print(my_list)

# # some list operations
# print(my_list[0])     # list indexing (just like strings)
# print(my_list[0:3])   # list slicing (just like strings)
# print(len(my_list))   # getting the number of items in a list
# print(345 in my_list) # checking if an item is in the list
# print(sum(my_list))   # computing the sum of the items

# # modifying a list
# print(my_list.append("hello")) # this does not return anything
# print(my_list)                 # my_list is changed
# my_list.append([3, "hi", 4])   # appending a list to a list. Now there is a list inside another list.
# print(my_list)
# print(my_list[5][0])           # access the first element ( [0] ) of the list within a list.

# my_list[0] = 22                # change the value, lists are mutable
# print(my_list)

# del my_list[0]                 # deleting an item
# print(my_list)

# # we can concatenate lists with the + operator (this creates a new list)
# list1 = [1,2,3]
# list2 = [4,5,6]
# list3 = list1 + list2
# print(list3)

# # different variables pointing to the same list
# listA = [0]
# listB = listA
# listB.append(1)
# print(listB) # we changed this
# print(listA) # this is also changed (they are the same list)

# # making a (shallow) copy of a list
# listA = [0]
# listB = listA[:]
# listB.append(1)
# print(listB) # we changed this
# print(listA) # this hasn't changed


# ## TUPLES
# fruit_tuple = ('apple', 'orange', 'banana', 'guanabana')
# print(fruit_tuple[3])    # tuple indexing (same as for lists/strings)
# fruit_tuple[3] = 'grape' # trying to modify a tuple will cause an error

# # typecasting a tuple to a list
# # we can convert a tuple into a list (and vice-versa)
# this_tuple = (1,2,3)
# this_list = list(this_tuple)
# print(this_list)
