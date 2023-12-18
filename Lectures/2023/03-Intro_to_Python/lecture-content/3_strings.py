#!/usr/bin/env python

# ## strings
# message = "Hello, I am a string"
# print(message)

# # string indexing
# print(message[0])  # first character
# print(message[1])  # second character
# print(message[-1]) # last character
# print(message[-2]) # penultimate character

# # string slicing
# print(message[7:])   # 8th all the way to last character
# print(message[7:11]) # 8th to 11th character

# print(message[7:-7]) # can use negative indices

# # strings are immutable
# message = "Hello, I am a string"
# message[1] = "Y"

# message = "Hello, I am a string"
# message = "Y" + message[1:] # we can make a new string and assign it to the same variable
# print(message)

# # some string methods/operations on strings
# message = "This is a string!"
# print(message)

# print(len(message)) # length of strings

# print(message + " And you can add stuff on.") # creates a new string

# print("string" in message) # True if "string" is inside the message variable

# print(message.count("i")) # Counts the number of times 'i' appears in the string
# print(message.find("s"))  # Finds the index of the first 's' it finds in the string
