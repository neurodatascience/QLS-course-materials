## HINTS FOR
# Task 1: Track and share your own work

## Task
Create a local repository, link it to a remote repository on GitHub, and practice changing the repo and sharing your changes on GitHub.

## Add to your drawing
As you go through these steps, add to your drawing the commands that move changes between different parts of the git project. In this version of the file, these steps are **bolded**.

## Steps
1. Make a directory somewhere on your machine. Make its name unique so there isn't another with the same name on GitHub.\
`mkdir <your_creative_dir_name>`

2. Change into the directory\
`cd <your_creative_dir_name>`

3. **Initialize a git repository**\
`git init`

4. Make a python file with some contents\
`<text editor> <filename>`

5. (optional) Try inspecting what git is/isn't tracking\
`git status`

6. **Stage the change**\
`git add <filename>`

7. **Commit the change**\
`git commit -m "<your short, informative commit message>"`

8. Make a PUBLIC repository on GitHub with the same name (if it's private, you won't be able to do the collaboration task; you can delete it afterwards).    
   - Go to `www.github.com/<your-github-username>` and log in
   - Click "Repositories"
   - Click "New"
   - Enter the name of your repo
   - Choose to make your repo Public
   - Don't check any boxes
   - Click "Create repository"    

9. Once you create your repo on GitHub, it will bring you to a page with a heading *"... or push an existing repository from the command line"*. Use this code to tell git the shortname and address of your remote repository, rename your principal branch to 'main', and **push your local repo contents to the remote repo**.\
`git remote add <remote shortname> <remote address>`\
`git branch -M main`\
`git push -u <remote> <branch>`\
*Note that you only need the `-u` the first time*

10. If you have time before the next task, repeat these steps a few times:
      - Change your file
      - **Stage the change**
      - **Commit the change**
      - **Push the change to GitHub**