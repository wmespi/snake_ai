import matplotlib.pyplot as plt
import  numpy as np

class SnakeGrid:

    # Grid label constants
    EMPTY = 0
    SNAKE = 1
    WALL = 2
    FOOD = 3

    def __init__(self, grid_size=15, seed=1234):

        # Initialize class attributes
        self.rng = np.random.default_rng(seed=seed)
        self.grid_size = grid_size
        
        # Initialize the grid
        grid = np.zeros( (self.grid_size, self.grid_size) ,dtype=np.uint8) + self.EMPTY
        grid[0,:] = self.WALL; grid[:,0] = self.WALL; 
        grid[self.grid_size-1,:] = self.WALL; grid[:,self.grid_size-1] = self.WALL

        # Initialize location of food
        food_x, food_y = self.place_point()
        grid[food_x, food_y] = self.FOOD

        # Initialize location of snake
        snake_x, snake_y = self.place_point()
        while food_x == snake_x and food_y == snake_y:
            snake_x, snake_y = self.place_point()
        grid[snake_x, snake_y] = self.SNAKE

        # Save grid
        self.init_grid = grid
        self.grid = grid
        self.snake_coordinates = [[snake_x, snake_y]]

        # Initialize distance to food
        self.head_dist_to_food = self.grid_distance(self.snake_coordinates, np.argwhere(self.grid==self.FOOD))

        # Display initial plot
        self.game_plot()
        

    def place_point(self):

        # Get x,y coordinates
        food_x = self.rng.integers(low=1, high=self.grid_size - 2)
        food_y = self.rng.integers(low=1, high=self.grid_size - 2)

        return food_x, food_y
    
    def grid_distance(self,pos1,pos2):
        return np.linalg.norm(np.array(pos1,dtype=np.float32)-np.array(pos2,dtype=np.float32))
    
    def game_plot(self):

        # Get indexes of grid labels
        wall_ind = (self.grid==self.WALL)
        snake_ind = (self.grid==self.SNAKE)
        food_ind = (self.grid==self.FOOD)

        # Create color array for plot
        color_array=np.zeros((self.grid_size,self.grid_size,3),dtype=np.uint8)+255 #default white
        color_array[wall_ind,:]= np.array([0,0,0]) #black walls
        color_array[snake_ind,:]= np.array([0,0,255]) #bluish snake
        color_array[food_ind,:]= np.array([0,255,0]) #green food  

        # Plot grid
        fig = plt.figure()
        plt.axis("off")
        plt.imshow(color_array)
        plt.show()

        return color_array