'''
GUI for 2D Spacecraft Docking Simulation

Created by Kai Delsing
Mentor: Kerianne Hobbs

Description:
	A GUI for basic interaction with the Spacecraft Docking files, realtime plotting, and simulation recalls

'''
from RL_algorithms import viewer
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import Menu
import os

border_effects = {
    "flat": tk.FLAT,
    "sunken": tk.SUNKEN,
    "raised": tk.RAISED,
    "groove": tk.GROOVE,
    "ridge": tk.RIDGE
}
plot_list = [
    'Distance vs time',
    'relative velocity',
    'thrust vs time',
    'reward vs time',
    'cumulative thrust vs time',
    'backup controller vs RL'
]

def sim(filePath, custom):
	if custom == True:
		print("Custom Settings activated", custom)
		set = CustomSettings()
	else:
		print("Custom Settings not activated", custom)
		set = None
	#/home/kdelsing/spacecraftdockingrl/RL_algorithms/saved_models/VPGbaseline.dat
	viewer.view(env_name='spacecraft-docking-v0', hidden_sizes=[64,64], episodes=5, latest=False, algo='VPG', RTA=False, Path=filePath, custom_settings=set)


class Home():
	def __init__(self):
		window = Tk()
		window.title("Spacecraft Docking GUI")
		window.geometry('550x200')
		p = StringVar()


		def get_file(dialog, path):
			if dialog: #file dialog prompt
				dir = filedialog.askopenfilename(initialdir = "/home",title = "Select file",filetypes = (("data files","*.dat"),("all files","*.*")))
				##CONVERT DIR TO STRING##
				file = dir
			else: #if path was entered via the entry box
				file = path

			if file: #if file path isn't empty
				print('PATH: \"', file, '\"')
				PlotSet(file)
			else:
				print("Path empty")

		#create headers and text labels
		Label(window, text="Spacecraft Docking GUI").grid(column=0, row=0)
		Label(window, text="- - - - - - - - - -").grid(column=0, row=1)
		Label(window, text="Click the button to open a file dialog \n or enter a file path in the text box below!").grid(column=0, row=2)
		Label(window, text=" ").grid(column=0, row=3)

		#This button is disabled as the method of turing the file object returned from the file dialog
		#into a string (a string path is needed for the viewer) is currently unknown.
		#To enable the button, delete hte "state-tk.DISABLED," text from the following line
		Button(window, state=tk.DISABLED, text="File Dialog", command=(lambda: get_file(dialog=True, path="/"))).grid(column=0, row=4)
		e = tk.Entry(window, width = 30, textvariable=p).grid(column=0, row=5)
		Button(window, text="Submit", command=(lambda: get_file(dialog=False, path=p.get()))).grid(column=1, row=5)
		Button(window, text='Quit', command=window.quit).grid(column=1, row=7)
		window.mainloop()

class PlotSet():
	def __init__(self, filePath):
		plot_window = Tk()
		plot_window.title("Spacecraft Docking GUI")
		plot_window.geometry('280x220')
		Label(plot_window, text="Spacecraft Docking GUI").grid(column=0, row=0)
		Label(plot_window, text=" - Select which plots to display:").grid(column=0, row=1)
		self.vars = []
		for i in range(len(plot_list)):
			#Create a variable & checkbox, then link the checkbox to the variable
			var = BooleanVar()
			self.vars.append(var)
			chk = Checkbutton(plot_window, text=plot_list[i], variable=self.vars[len(self.vars)-1]).grid(column=0, row=(2+i))

		def print_state(self):
			#Print the respective plot titles and checkbox variables
			for i in range(len(plot_list)):
				print(plot_list[i], " - ", self.vars[i].get())

		def submit():
			#Start render
			print("Begin Render...")
			sim(filePath, self.custom.get())


			print("...Render Complete")

		self.custom = BooleanVar()
		Checkbutton(plot_window, text="\nCustom Settings?", variable=self.custom).grid(column=0, row=len(plot_list)+2)
		Button(plot_window, text='Peek', command=(lambda: print_state(self))).grid(column=0, row=len(plot_list)+3) #see the values of the checkbox variables
		Button(plot_window, text='Quit', command=plot_window.quit).grid(column=0, row=len(plot_list)+4) #exit window
		Button(plot_window, text='Submit', command=submit).grid(column=0, row=len(plot_list)+5) #begin render
		plot_window.mainloop()

class CustomSettings():
	def __init__(self):
		#(thrust='Block', trace=5, v_arrow=True, f_arrow=True, stars=200, a1=0, b1=0, a2=0, b2=0, e_qualtity=0)
		settings_window = Tk()
		settings_window.title("Spacecraft Docking GUI")
		settings_window.geometry('500x180')
		Label(settngs_window, text="Spacecraft Docking GUI").grid(column=0, row=0)
		Label(settngs_window, text=" - Select your render settings:").grid(column=0, row=1)
		Label(settngs_window, text=" ").grid(column=0, row=2)
		self.settings = [" ", 0, IntVar(), IntVar(), 0, 0, 0, 0, 0, 0]

		Label(settngs_window, text="Thrust  Visualization:").grid(column=0, row=3)
		Radiobutton(settings_window, text="Block", variable=self.settings[0], value="Block").grid(column=0, row=4)
		Radiobutton(settings_window, text="Particle", variable=self.settings[0], value="Particle").grid(column=1, row=4)
		Radiobutton(settings_window, text="None", variable=self.settings[0], value="None").grid(column=2, row=4)

		Label(settings_window, text="Trace Spacing:").grid(column=0, row=5)
		Entry(settings_window, width=10, textvariable=self.settings[1]).grid(column=0, row=6)

		Checkbutton(settngs_window, text="Velocity Arrow:", variable=self.settings[2]).grid(column=0, row=7)
		Checkbutton(settngs_window, text="Force Arrow:", variable=self.settings[3]).grid(column=1, row=7)

		Label(settings_window, text="Number of Stars:").grid(column=0, row=8)
		Entry(settings_window, width=10, textvariable=self.settings[4]).grid(column=1, row=8)

		Label(settings_window, text="Ellipse Quality: (0 for no ellipse)").grid(column=0, row=9)
		Entry(settings_window, width=10, textvariable=self.settings[9]).grid(column=1, row=9)
		Label(settings_window, text="Ellipse A: Semi-Major").grid(column=1, row=10)
		Entry(settings_window, width=10, textvariable=self.settings[1]).grid(column=0, row=10)
		Label(settings_window, text="Ellipse A: Semi-Minor").grid(column=1, row=11)
		Entry(settings_window, width=10, textvariable=self.settings[1]).grid(column=0, row=11)
		Label(settings_window, text="Ellipse B: Semi-Major").grid(column=1, row=12)
		Entry(settings_window, width=10, textvariable=self.settings[1]).grid(column=0, row=12)
		Label(settings_window, text="Ellipse B: Semi-Minor").grid(column=1, row=13)
		Entry(settings_window, width=10, textvariable=self.settings[1]).grid(column=0, row=13)

		Button(plot_window, text='Quit', command=plot_window.quit).grid(column=0, row=14) #exit window
		Button(plot_window, text='Submit', command=submit).grid(column=0, row=14) #begin render

		e = (
			self.settings[0],
			self.settings[1],
			self.settings[2].get(),
			self.settings[3].get(),
			self.settings[4],
			self.settings[5],
			self.settings[6],
			self.settings[7],
			self.settings[8],
			self.settings[9],
			)
		return e
'''
	def getSettings():
		e = (
			self.settings[0],
			self.settings[1],
			self.settings[2].get(),
			self.settings[3].get(),
			self.settings[4],
			self.settings[5],
			self.settings[6],
			self.settings[7],
			self.settings[8],
			self.settings[9],
			)
		return e
'''

#Run GUI
m = Home()



#viewer(path, env_name='spacecraft-docking-v0', hidden_sizes=[64,64], episodes=10, latest=False, algo='VPG', RTA = True)

#file = filedialog.askopenfilename()
#file = filedialog.askopenfilename(filetypes = (("Text files","*.txt"),("all files","*.*")))
#file object = open(file_name, r, [, buffering])
#file object = open(file_name, w+, [, buffering])
'''
Old version

window = Tk()

window.title("Welcome to LikeGeeks app")

menu = Menu(window)

new_item = Menu(menu)

new_item.add_command(label='New')

new_item.add_separator()

new_item.add_command(label='Edit')

#new_item.add_command(label='New', command=clicked)

menu.add_cascade(label='File', menu=new_item)

window.config(menu=menu)

window.mainloop()
'''

'''
class Home(Frame):
    def __init__(self, parent=None, side=BOTTOM, anchor=W):
        Frame.__init__(self, parent)
        self.vars = []


    def go(self):
        root = Tk()
        #Button(root, text='Begin', command=plot).pack(side=TOP)
        Label(root, text = "Spacecraft Docking GUI").pack(side=TOP)
        Button(root, text='Peek').pack(side=TOP) #, command=print_state)
        Button(root, text='Quit', command=root.quit).pack(side=BOTTOM)
        root.mainloop()
class Plotting_Active(Frame):
    ...

class Plotting_Stager(Frame):
    def __init__(self, parent=None, picks=[], side=BOTTOM, anchor=W):
        Frame.__init__(self, parent)
        self.vars = []
        for pick in picks:
            var = IntVar()
            chk = Checkbutton(self, text=pick, variable=var)
            chk.pack(side=side, anchor=anchor, expand=YES)
            self.vars.append(var)

    def state(self):
        return map((lambda var: var.get()), self.vars)


if __name__ == '__main__':
    root = Tk()
    plot_list = Plotting_Stager(root, [
        'Distance vs time',
        'relative velocity',
        'thrust vs time',
        'reward vs time',
        'cumulative thrust vs time',
        'backup controller vs RL'])
    tgl = Plotting_Stager(root, ['English','German'])
    tgl.pack(side=BOTTOM)
    plot_list.pack(side=LEFT,  fill=X)
    plot_list.config(relief=GROOVE, bd=2) #bd = borderwidth

    def print_state():
        print(list(plot_list.state()), list(tgl.state()))
    def plot():
        Frame.quit

    Button(root, text='Begin', command=plot).pack(side=TOP)
    Button(root, text='Peek', command=print_state).pack(side=TOP)
    Button(root, text='Quit', command=root.quit).pack(side=BOTTOM)
    root.mainloop()
'''
