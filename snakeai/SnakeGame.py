import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt

class Snake_game(gym.Env):
    """
    Custom Environment for Stable Baseline 3 for the classic Snake 
    """
    metadata = {'render.modes': ['console','rgb_array']}
    #Direction constants
    n_actions = 3 #3 possible steps each turn
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    #Grid label constants
    EMPTY = 0
    SNAKE = 1
    WALL = 2
    FOOD = 3
    #Rewards
    #REWARD_PER_STEP = 0 # reward for every step taken, gets into infinite loops if >0
    #Define Max steps to avoid infinite loops
    REWARD_WALL_HIT = -20 #should be lower than -REWARD_PER_STEP_TOWARDS_FOOD to avoid hitting wall intentionally
    REWARD_PER_STEP_TOWARDS_FOOD = 1 #give reward for moving towards food and penalty for moving away
    REWARD_PER_FOOD = 50 
    MAX_STEPS_AFTER_FOOD = 200 #stop if we go too long without food to avoid infinite loops


    def __init__(self, grid_size=12):
        super(Snake_game, self).__init__()
        #Steps so far
        self.stepnum = 0; self.last_food_step=0
        # Size of the 2D grid (including walls)
        self.grid_size = grid_size
        # Initialize the snake
        self.snake_coordinates = [ (1,1), (2,1) ] #Start in lower left corner
        #Init the grid
        self.grid = np.zeros( (self.grid_size, self.grid_size) ,dtype=np.uint8) + self.EMPTY
        self.grid[0,:] = self.WALL; self.grid[:,0] = self.WALL; #wall at the egdes
        self.grid[int(grid_size/2),3:(grid_size-3)] = self.WALL; #inner wall to make the game harder
        self.grid[4:(grid_size-4),int(grid_size/2-1)] = self.WALL; #inner wall to make the game harder
        #self.grid[int(grid_size/2),2:(grid_size-2)] = self.WALL; #inner wall to make the game harder
        self.grid[self.grid_size-1,:] = self.WALL; self.grid[:,self.grid_size-1] = self.WALL
        for coord in self.snake_coordinates:
            self.grid[ coord ] = self.SNAKE  #put snake on grid
        self.grid[3,3] = self.FOOD  #Start in upper right corner
        #Init distance to food
        self.head_dist_to_food = self.grid_distance(self.snake_coordinates[-1],np.argwhere(self.grid==self.FOOD)[0] )
        #Store init values
        self.init_grid = self.grid.copy()
        self.init_snake_coordinates = self.snake_coordinates.copy()
        
        # The action space
        self.action_space = spaces.Discrete(self.n_actions)
        # The observation space, "position" is the coordinates of the head; "direction" is which way the sanke is heading, "grid" contains the full grid info
        self.observation_space = gym.spaces.Dict(
            spaces={
                "position": gym.spaces.Box(low=0, high=(self.grid_size-1), shape=(2,), dtype=np.int32),
                "direction": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32),
                "grid": gym.spaces.Box(low = 0, high = 3, shape = (self.grid_size, self.grid_size), dtype=np.uint8),
            })
    
    def grid_distance(self,pos1,pos2):
        return np.linalg.norm(np.array(pos1,dtype=np.float32)-np.array(pos2,dtype=np.float32))

    def reset(self):
        # Reset to initial positions
        self.stepnum = 0; self.last_food_step=0
        self.grid = self.init_grid.copy()
        self.snake_coordinates = self.init_snake_coordinates.copy()
        #Init distance to food
        self.head_dist_to_food = self.grid_distance(self.snake_coordinates[-1],np.argwhere(self.grid==self.FOOD)[0] )
        return self._get_obs()    
             
    def _get_obs(self):
            direction = np.array(self.snake_coordinates[-1]) - np.array(self.snake_coordinates[-2])
            #return observation in the format of self.observation_space
            return {"position": np.array(self.snake_coordinates[-1],dtype=np.int32),
                    "direction" : direction.astype(np.int32),
                    "grid": self.grid}                  
            
    def step(self, action):
        #Get direction for snake
        direction = np.array(self.snake_coordinates[-1]) - np.array(self.snake_coordinates[-2])
        if action == self.STRAIGHT:
            step = direction #step in the firection the snake faces
        elif action == self.RIGHT:
            step = np.array( [direction[1], -direction[0]] )  #turn right
        elif action == self.LEFT:
            step = np.array( [-direction[1], direction[0]] )   #turn left
        else:
            raise ValueError("Action=%d is not part of the action space"%(action))
        #New head coordinate
        new_coord = (np.array(self.snake_coordinates[-1]) + step).astype(np.int32)
        #grow snake     
        self.snake_coordinates.append( (new_coord[0],new_coord[1]) ) #convert to tuple so we can use it to index

        
        #Check what is at the new position
        new_pos = self.snake_coordinates[-1]
        new_pos_type = self.grid[new_pos]
        self.grid[new_pos] = self.SNAKE #this position is now occupied by the snake
        done = False; reward = 0 #by default the game goes on and no reward   
        if new_pos_type == self.FOOD:
            reward += self.REWARD_PER_FOOD
            self.last_food_step = self.stepnum
            #Put down a new food item
            empty_tiles = np.argwhere(self.grid==self.EMPTY)
            if len(empty_tiles):
                new_food_pos=empty_tiles[np.random.randint(0,len(empty_tiles))]
                self.grid[new_food_pos[0],new_food_pos[1]] = self.FOOD
            else:
                done = True #no more tiles to put the food to
        else:
            #If no food was eaten we remove the end of the snake (i.e., moving not growing)
            self.grid[ self.snake_coordinates[0] ] = self.EMPTY
            self.snake_coordinates = self.snake_coordinates[1:]
            if  (new_pos_type == self.WALL) or (new_pos_type == self.SNAKE):
                done = True #stop if we hit the wall or the snake
                reward += self.REWARD_WALL_HIT #penalty for hitting walls/tail
#             else:
#                 reward += self.REWARD_PER_STEP
                
        #Update distance to food and reward if closer
        head_dist_to_food_prev = self.head_dist_to_food
        self.head_dist_to_food = self.grid_distance( self.snake_coordinates[-1],np.argwhere(self.grid==self.FOOD)[0] )
        if head_dist_to_food_prev > self.head_dist_to_food:
            reward += self.REWARD_PER_STEP_TOWARDS_FOOD #reward for getting closer to food
        elif head_dist_to_food_prev < self.head_dist_to_food:
            reward -= self.REWARD_PER_STEP_TOWARDS_FOOD #penalty for getting further
        
        #Stop if we played too long without getting food
        if ( (self.stepnum - self.last_food_step) > self.MAX_STEPS_AFTER_FOOD ): 
            done = True    
        self.stepnum += 1

        return  self._get_obs(), reward, done, {}

    def render(self, mode='rgb_array'):
        if mode == 'console':
            print(self.grid)
        elif mode == 'rgb_array':
            return self.snake_plot()
        else:
            raise NotImplementedError()

    def close(self):
        pass
    
    def snake_plot(self, plot_inline=False):
        wall_ind = (self.grid==self.WALL)
        snake_ind = (self.grid==self.SNAKE)
        food_ind = (self.grid==self.FOOD)
        #Create color array for plot, default white color
        Color_array=np.zeros((self.grid_size,self.grid_size,3),dtype=np.uint8)+255 #default white
        Color_array[wall_ind,:]= np.array([0,0,0]) #black walls
        Color_array[snake_ind,:]= np.array([0,0,255]) #bluish snake
        Color_array[food_ind,:]= np.array([0,255,0]) #green food  
        #plot
        if plot_inline:
            fig=plt.figure()
            plt.axis('off')
            plt.imshow(Color_array, interpolation='nearest')
            plt.show()
        return Color_array