#!/usr/bin/env python

# # in this piece of code, the variable w is not defined, so it throws an error
# try:
#     print(w) # this code inside the try block is tested for error
# except Exception:
#     # the code inside the except block is executed if there are errors. The program does not crash with an error
#     print("An exception occurred")

# # using multiple except blocks
# try:
# #     print(int('w')) # TypeError
#     print(w) # NameError
    
# # the code throws a name error when it fails outside a try block
# # so if we know this is a possibility, we catch it specifically.
# except NameError:
#     print("Variable w is not defined")
    
# # and this code catches more general errors, in case something else unexpected goes wrong.
# except Exception:
#     print("Something else went wrong")

# # the 'finally' block is always executed
# try:
#     print(w)
#     print("This is not executed")
# except NameError:
#     print("Something went wrong")
# finally:
#     print("This executes even if there is an error")

    