## HINTS FOR
# Task 2: Collaborate with a partner

## Task
Practice collaborating with a partner on git and github. This includes opening an issue on their repo, forking it, making a change, and integrating that change with a pull request.

## Add to your drawing
As you go through these steps, add to your drawing the commands that move changes between different parts of the git project. In this version of the file, these steps are **bolded**.

## Steps
1. Find a partner (the instructor will help with this)

2. Find your partner on slack, and in a direct message, tell them the address of the github repo that you made in task 1.

3. On Slack, decide with your partner what you'll change in each other's repo. E.g., you could change the file they uploaded or add a new file.

4. Open an issue on your partner's repo to say what you'll change.

5. Look at the issue your partner opened on your repo, and make a comment on it saying that they can go ahead and make the change. 

6. **Fork your partner's repo**

7. **Clone your fork of your partner's repo to your computer** (*make sure you're not still in your own repo's folder*).\
   `git clone https://github.com/<YOUR-USERNAME>/<YOUR-PARTNERS-REPO-NAME>`

8. Create a branch to work on\
   `git branch <branch name>`

9.  Move onto that branch\
    `git checkout <branch name>`

10. Make the change you agreed on

11. **Stage the change**\
    `git add <file(s)>`

12. **Commit the change, with the issue number in the commit message**\
    `git commit -m "commit message; addresses #X"`

13. **Push the change to your fork of your partner's repository**\
    `git push origin <branch-name>`

14. **Open a pull request**

15. Review and merge each others' pull requests