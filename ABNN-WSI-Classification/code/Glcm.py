import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from skimage.feature import greycomatrix, greycoprops
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class ImageSelectionGUI:
    def __init__(self, image_path):
        self.root = tk.Tk()
        self.root.title("Image Selection")
        self.image = Image.open(image_path)
        self.patch_size = 10
        
        self.frame = tk.Frame(self.root)
        self.frame.pack()

        self.current_marker = 'red'  # Variable to keep track of current marker color
        
        self.canvas = tk.Canvas(self.frame, width=self.image.width - 200, height=self.image.height-200)
        self.canvas.grid(row=0, column=0, columnspan=2)
        
       
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.image_tk, anchor=tk.NW)
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
        self.button = tk.Button(self.frame, text='Get Locations', command=self.get_locations)
        self.button.grid(row=1, column=0, pady=10)
        
        self.clear_button = tk.Button(self.frame, text='Clear Selection', command=self.clear_selection)
        self.clear_button.grid(row=1, column=1, pady=10)
        
        self.gclm_button = tk.Button(self.frame, text='Perform GLCM', command=self.perform_gclm)
        self.gclm_button.grid(row=2, column=0, columnspan=2, pady=10)


        self.marker1_button = tk.Button(self.frame, text='Marker 1 (Red)', command=lambda: self.set_marker_color('red'))
        self.marker1_button.grid(row=2, column=0, pady=10)

        self.marker2_button = tk.Button(self.frame, text='Marker 2 (Blue)', command=lambda: self.set_marker_color('blue'))
        self.marker2_button.grid(row=2, column=1, pady=10)

        self.marker1_button.config(command=lambda: self.set_marker_color('red'))
        self.marker2_button.config(command=lambda: self.set_marker_color('blue'))

        self.selected_locations = []
        self.markers = []
 
        
        self.root.mainloop()
    
    def set_marker_color(self, color):
        self.current_marker = color

        
    def on_canvas_click(self, event):
        location = (event.x, event.y)
        # if len(self.selected_locations) % 2 == 1:  # Marker 2 click
        #     self.toggle_marker_color()
        self.selected_locations.append((location, self.current_marker))
        self.draw_marker(location[0], location[1], self.current_marker)

        
    def draw_marker(self, x, y, color):
        marker_size = 10
        marker = self.canvas.create_oval(x - marker_size, y - marker_size, x + marker_size, y + marker_size, outline=color)
        self.markers.append(marker)


    
    def select_marker1(self):
        self.canvas.unbind('<Button-1>')
        self.canvas.bind('<Button-1>', self.on_canvas_click)
        
    def select_marker2(self):
        self.canvas.unbind('<Button-2>')
        self.canvas.bind('<Button-2>', self.on_canvas_click)
        
    def get_locations(self):
        if self.selected_locations:
            print('Selected locations:')
            for location in self.selected_locations:
                print(location)
        else:
            print('No locations selected.')
            
    def clear_selection(self):
        for marker in self.markers:
            self.canvas.delete(marker)
        self.selected_locations = []
        self.markers = []
    
    def toggle_marker_color(self):
        if self.current_marker == 'red':
            self.current_marker = 'blue'
            self.toggle_button.config(text='Current Marker: Blue')
        else:
            self.current_marker = 'red'
            self.toggle_button.config(text='Current Marker: Red')



    def perform_gclm(self):

        if not self.selected_locations:
            print('No locations selected for GLCM analysis.')
            return
        
        red_locations = [location for location, marker in self.selected_locations if marker == 'red']
        blue_locations = [location for location, marker in self.selected_locations if marker == 'blue']

        if not red_locations:
            print('No red locations selected for GLCM analysis.')
            return

        if not blue_locations:
            print('No blue locations selected for GLCM analysis.')
            return

        # compute GLCM properties for red locations
        xs = []
        ys = []

        for location in (red_locations + blue_locations):
            x, y = location
            region = self.image.crop((x - self.patch_size, y - self.patch_size, x + self.patch_size, y + self.patch_size)).convert('L')
            glcm = greycomatrix(region, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
            xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
            ys.append(greycoprops(glcm, 'correlation')[0, 0])


        # create the figure
        fig = plt.figure(figsize=(8, 8))

        # display original image with locations of patches
        ax = fig.add_subplot(3, 2, 1)
        ax.imshow(self.image, cmap=plt.cm.gray, vmin=0, vmax=255)

        for (x, y) in red_locations:
            ax.plot(x + self.patch_size / 2, y + self.patch_size / 2, 'gs')
        for (x, y) in blue_locations:
            ax.plot(x + self.patch_size / 2, y + self.patch_size / 2, 'bs')
        ax.set_xlabel('Original Image')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('image')

            # for each patch, plot (dissimilarity, correlation)
        ax = fig.add_subplot(3, 2, 2)
        ax.plot(xs[:len(red_locations)], ys[:len(blue_locations)], 'go',
                label='Non Cored')
        ax.plot(xs[len(red_locations):], ys[len(blue_locations):], 'bo',
                label='Cored')
        # High correlation and low dissimilariy - Homogenous texture
        ax.set_xlabel('GLCM Dissimilarity')
        ax.set_ylabel('GLCM Correlation')
        ax.legend()

        # display the image patches
        for i, location in enumerate(red_locations):
            x, y = location
            ax = fig.add_subplot(3, len(red_locations), len(red_locations)*1 + i + 1)
            patch = self.image.crop((x - self.patch_size, y - self.patch_size, x + self.patch_size, y + self.patch_size)).convert('L')
            ax.imshow(patch, cmap=plt.cm.gray,
                    vmin=0, vmax=255)
            ax.set_xlabel(f"Non Cored {i + 1}")

        for i, patch in enumerate(blue_locations):
            x, y = location
            ax = fig.add_subplot(3, len(blue_locations), len(blue_locations)*2 + i + 1)
            patch = self.image.crop((x - self.patch_size, y - self.patch_size, x + self.patch_size, y + self.patch_size)).convert('L')
            ax.imshow(patch, cmap=plt.cm.gray,
                    vmin=0, vmax=255)
            ax.set_xlabel(f"Cored {i + 1}")


        # display the patches and plot
        fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
        plt.tight_layout()
        plt.show()

        

                        
# Example usage
image_path = "/mnt/new-nas/work/data/npsad_data/vivek/reports/Manuscript/XE15-039_1_AmyB_1_2070x_86690y_image.png"
gui = ImageSelectionGUI(image_path)
