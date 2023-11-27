import os
import pandas as pd
import time
import tkinter as tk
from tkinter import filedialog
import project

def the_prompt():
    global projet_name, root_name
    str_aux = root_name[:]
    if projet_name is None:
        pass
    else:
        str_aux = str_aux + "@" + projet_name[:]
    return "{} >>>:".format(str_aux)

def warning(message):
    return "{} warning! {}".format(the_prompt(), message)


def confirm(prefix_message=None):
    if prefix_message is None:
        str_message = "{} proceed? (y/n): ".format(the_prompt())
    else:
        str_message = "{} {} -- proceed? (y/n): ".format(the_prompt(), prefix_message)
    while True:
        str_confirm = input(str_message).strip()
        if str_confirm in ["y", "n"]:
            break
        else:
            pass
    return str_confirm

def pick_folder(title="Select a folder"):
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    folder_path = filedialog.askdirectory(title=title)

    if folder_path:
        return folder_path
    else: # cancel
        return None

def func_a(params):
    print(params)
    time.sleep(2)

def func_b(params):
    print("wohooo")
    time.sleep(2)

def func_c(params):
    dict_menu = {
        "Option A": [func_a, None],
        "Option B": [func_b, None]
    }

    m2 = Menu(dict_actions=dict_menu, name="project setup")
    m2.loop()

def change_workplace(params=None):
    global prompt_name
    str_new_folder = pick_folder()
    if str_new_folder is None:
        print('{} cancel'.format(the_prompt()))
    else:
        global root
        root = str_new_folder
        print('{} new workplace: {}'.format(the_prompt(), root))
    time.sleep(2)
    return None

def new_project(params=None):
    global prompt_name, project_name, root, my_project

    def set_new_project(params=None):
        while True:
            str_name = input("{} enter new project name: ".format(the_prompt())).strip()
            if str_name in os.listdir(root):
                print(warning("project already exists"))
                break
            elif any(char.isspace() for char in str_name):
                print(warning("blank spaces in name not allowed"))
                break
            elif len(str_name) > 20:
                print(warning("name too long (max 20 characters)"))
                break
            else:
                str_confirm = confirm(prefix_message="new name: {}".format(str_name))
                if str_confirm == "n":
                    pass
                else:
                    my_project = project.Project(name=str_name, root=root)
                    my_project = None
                break
        return 0


    dict_menu = {
        "set project name": [set_new_project, None],
    }

    m2 = Menu(dict_actions=dict_menu, name="new project", message=get_workplace)
    m2.loop()


def project_session():
    global my_project

    dict_menu = {
        "data management": [func_a, None],
        "simulation tools": [func_a, None],
        "assessment tools": [func_a, None],
        "model diagnostics": [func_a, None],
    }
    m2 = Menu(dict_actions=dict_menu, name=my_project.name, message=get_workplace)
    m2.loop()



def open_project(params=None):
    global root, my_project, prompt_name

    def set_project(params=None):
        global my_project, projet_name, root
        my_project = project.Project(name=params[:], root=root)
        projet_name = params[:]

        project_session()

        my_project = None
        projet_name = None

        return 0


    list_projects = os.listdir(root)
    print(list_projects)
    dict_menu = dict()
    for p in list_projects:
        dict_menu[p] = [set_project, p]

    m2 = Menu(dict_actions=dict_menu, name="open project", message=get_workplace)
    m2.loop()


def get_workplace():
    global root, my_project
    if my_project is None:
        str_workplace = root[:]
    else:
        str_workplace = my_project.path_main[:]
    return "\nworkplace: {}\n".format(str_workplace)

class Menu:
    '''
    The Primitive Menu Object
    '''
    def __init__(self, dict_actions, name="Menu", exit_key="e", prompt_name="plans", message=None):
        """Instantiate the Menu object

        :param dict_actions: dictionary of actions (functions) and parameters (dict)
        :type dict_actions: dict
        :param name: menu title
        :type name: str
        :param exit_key: exite key
        :type exit_key: str
        :param prompt_name: prompt name
        :type prompt_name: str
        """
        self.header_size = 60
        self.header_char = "*"
        self.name = name
        self.message = message
        self.exit = exit_key
        self.dict_actions = dict_actions
        self.dict_actions["exit menu"] = self.exit
        # get lists
        self.list_options = [o for o in self.dict_actions]
        self.list_keys = [str(i + 1) for i in range(len(self.list_options))]
        # set exit on keys
        self.list_keys[-1] = self.exit
        # actions by keys
        self.dict_keys = dict()
        for i in range(len(self.list_keys)):
            self.dict_keys[self.list_keys[i]] = dict_actions[self.list_options[i]]
        # labels dict
        self.dict_labels = dict()
        for i in range(len(self.list_keys)):
            self.dict_labels[self.list_keys[i]] = self.list_options[i]

    def get_table(self):
        """Get the menu table

        :return: table dataframe
        :rtype: :class:`pandas.DataFrame`
        """
        # set header line

        list_options = self.list_options.copy()
        list_keys = self.list_keys.copy()
        # header line
        list_lines_sizes = [len(str_line) for str_line in list_options]
        n_line_size = max(list_lines_sizes)

        list_options.insert(0, "-" * (n_line_size + 2))
        list_keys.insert(0, "-" * 10)
        df_table = pd.DataFrame(
            {
                "options": list_options,
                "keys": list_keys
            }
        )
        return df_table

    def header(self):
        n_name = len(self.name)
        n_aux = int((self.header_size - n_name) / 2)
        str_aux = self.header_char * n_aux
        return "\n{} {} {}\n".format(str_aux, self.name.upper(), str_aux)

    def ask(self):
        print(self.header())
        if self.message is None:
            pass
        else:
            print(self.message())
        print(self.get_table().to_string(index=False))
        print("\n")
        while True:
            str_ans = input("{} enter key: ".format(the_prompt())).strip()
            if str_ans == "":
                pass
            else:
                break
        return str_ans

    def validade(self, str_answer):
        if str_answer in self.list_keys:
            print("{} OK. selected key: <{}>".format(the_prompt(), str_answer))
            return True
        else:
            print("{} <{}> key not found. options available: {}".format(
                the_prompt(),
                str_answer,
                self.list_keys))
            time.sleep(2)
            return False

    def loop(self):
        self.get_table()
        while True:
            str_answer = self.ask()
            b_valid = self.validade(str_answer=str_answer)
            if str_answer == self.exit:
                break
            else:
                if b_valid:
                    str_confirm = confirm(prefix_message=self.dict_labels[str_answer])
                    if str_confirm == "y":
                        # run function
                        output = self.dict_keys[str_answer][0](self.dict_keys[str_answer][1])
                        if output == 0:
                            break



if __name__ == '__main__':
    # define root
    root_name = "plans"
    root = "C:/" + root_name
    # handle root
    if os.path.isdir(root):
        pass
    else:
        os.mkdir(root)
    # define prompt
    prompt_name = "plans"

    # project
    projet_name = None
    my_project = None
    #
    dict_menu = {
        "open project": [open_project, None],
        "new project": [new_project, None],
        "change workplace": [change_workplace, None]
    }
    # set menu
    m = Menu(
        dict_actions=dict_menu,
        name="plans - Home",
        prompt_name=prompt_name,
        message=get_workplace


    )
    # loop
    m.loop()