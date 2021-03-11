import pygame
from queue import PriorityQueue

#WIDTH = 780
WIDTH = 1530
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")



button = pygame.Rect(10, 10, 100, 30)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
YELLOW = (255,255,0)
WHITE = (255,255,255)
BLACK = (0,0,0)
PURPLE = (128,0,128)
ORANGE = (255,165,0)
GREY = (128,128,128)
TURQUOISE = (64,224,208)

#<----------------Square on Grid----------------------->
class Spot:
    def __init__(self,row,col,width,total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self): return self.row, self.col
    def is_closed(self): return self.color == RED
    def is_open(self): return self.color == GREEN
    def is_barrier(self): return self.color == BLACK
    def is_start(self): return self.color == ORANGE
    def is_end(self): return self.color == TURQUOISE
    def reset(self): self.color = WHITE
    def make_closed(self): self.color = RED
    def make_open(self): self.color = GREEN
    def make_barrier(self): self.color = BLACK
    def make_start(self): self.color = ORANGE
    def make_end(self): self.color = TURQUOISE
    def make_path(self): self.color = PURPLE
    def draw(self, win): pygame.draw.rect(win,self.color,(self.x,self.y,self.width,self.width))

    def update_neighbors(self,grid):
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row+1][self.col].is_barrier():#DOWN
            self.neighbors.append(grid[self.row+1][self.col])

        if self.row > 0 and not grid[self.row-1][self.col].is_barrier():#UP
            self.neighbors.append(grid[self.row-1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col+1].is_barrier(): # RIGHT
            self.neighbors.append(grid[self.row][self.col+1])

        if self.row > 0 and not grid[self.row][self.col-1].is_barrier():#LEFT
            self.neighbors.append(grid[self.row][self.col-1])

    def __lt__(self,other): return False

#<----------------H Function----------------------->
def h(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return abs(x1-x2) + abs(y1-y2)


#<----------------Create path from nodes----------------------->
def reconstruct_path(came_from, current, draw):
    while current in came_from:
        current = came_from[current]
        if not current.is_start() and not current.is_end(): current.make_path()
        draw()

#<----------------Create path from nodes----------------------->
def create_path(path,rows,grid):
    for cur in path:
        if not cur.is_start() and not cur.is_end(): cur.make_path()
        draw(WIN,grid,rows,WIDTH)

#<---------------- Breath First Search----------------------->
def bfs(graph_to_search, start, end, rows):
    queue = [[start]]
    visited = set()

    while queue:
        path = queue.pop(0)
        vertex = path[-1]
        for node in visited:
            if not node.is_start() and not node.is_end(): node.make_closed()
        if vertex == end:
            print("HOUNS")
            create_path(path,rows,graph_to_search)
            return path
        elif vertex not in visited:
            for current_neighbour in vertex.neighbors:
                if not current_neighbour.is_start() and not current_neighbour.is_end(): current_neighbour.make_open()
                new_path = list(path)
                new_path.append(current_neighbour)
                queue.append(new_path)
            visited.add(vertex)
        draw(WIN, graph_to_search, rows, WIDTH)
    print("NOT HOUNS")

#<---------------- Depth First Search----------------------->
def dfs(graph_to_search, start, end, rows):
    queue = [[start]]
    visited = set()
    a = True
    bf = False
    while queue:
        path = queue.pop()
        vertex = path[-1]
        for node in visited:
            if not node.is_start() and not node.is_end(): node.make_closed()
        if vertex == end:
            print("HOUNS")

            create_path(path,rows,graph_to_search)
            return path
        elif vertex not in visited:
            for current_neighbour in vertex.neighbors:
                if not current_neighbour.is_start() and not current_neighbour.is_end(): current_neighbour.make_open()
                new_path = list(path)
                new_path.append(current_neighbour)
                queue.append(new_path)
            visited.add(vertex)
        draw(WIN, graph_to_search, rows, WIDTH)
    print("NOT HOUNS")


#<---------------- Border around Screen----------------------->
def make_border(graph):
    for j in range(len(graph)):
        for i in range(len(graph[j])):
            if i == 0 or j == 0 or i == len(graph[j])-1 or j == len(graph)-1:
                graph[j][i].make_barrier()


#<---------------- A* Pathfinding----------------------->
def algorithm(draw, grid, start, end):
    count = 0 
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {sp: float("inf") for row in grid for sp in row}
    g_score[start] = 0

    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())

    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        current = open_set.get()[2]
        open_set_hash.remove(current)
        if current == end:
            reconstruct_path(came_from, end, draw)
            end.make_end()
            start.make_start()
            return True
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1
            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    if neighbor != end: 
                        neighbor.make_open()
        draw()
        if current != start and current != end:
            current.make_closed()
    return False


#<---------------- A* Pathfinding----------------------->
def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i,j,gap,rows)
            grid[i].append(spot)

    return grid


#<---------------- Draw the grid lines----------------------->
def draw_grid(win,rows,width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win,GREY,(0,i*gap),(width,i*gap))
        for j in range(rows):
            pygame.draw.line(win,GREY,(j*gap,0),(j*gap,width))


#<---------------- Draw The Grid and Squares----------------------->
def draw(win,grid,rows,width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win,rows, width)
    pygame.display.update()


#<----------------Draw the Grid and Squares without display ----------------------->
def draw_main(win,grid,rows,width):
    win.fill(WHITE)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win,rows, width)


#<----------------Get Mouse Clicked Tile ----------------------->
def get_clicked_pos(pos,rows,width):
    gap = width // rows
    y,x = pos
    row = y // gap
    col = x // gap
    return row, col

#<---------------- Main Function----------------------->
def main(win, width):
    ROWS = 30
    grid = make_grid(ROWS, width)
    make_border(grid)
    start = None
    end = None
    a = True
    bf = False  
    df = False
    run = True
    started = False
    while run:
        #Draw to the Screen
        draw_main(win, grid, ROWS, width)
        if a == True: pygame.draw.rect(WIN, TURQUOISE, button)
        elif bf == True: pygame.draw.rect(WIN, GREEN, button)
        else: pygame.draw.rect(WIN, PURPLE, button)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if started:
                continue

            #Place Tile
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row,col = get_clicked_pos(pos,ROWS, width)
                spot = grid[row][col]
                if not start and spot != end and not spot.is_barrier():
                    start = spot
                    start.make_start()
                elif not end and spot != start and not spot.is_barrier():
                    end = spot
                    end.make_end()
                elif spot != end and spot != start:
                    spot.make_barrier()
            
            #Delete Tile 
            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row,col = get_clicked_pos(pos,ROWS, width)
                spot = grid[row][col]
                spot.reset()
                if spot == start:
                    start = None
                elif spot == end:
                    end = None
                
            if event.type == pygame.KEYDOWN:
                #Start Algorithm
                if event.key == pygame.K_SPACE and not started:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    if a == True: algorithm(lambda: draw(win,grid,ROWS,width), grid, start, end)
                    elif bf == True: bfs(grid,start,end,ROWS)
                    elif df == True: dfs(grid,start,end,ROWS)

                #Reset Algorithm
                if event.key == pygame.K_c:
                    for row in grid:
                        for spot in row:
                            if not spot.is_barrier() and not spot.is_start() and not spot.is_end(): spot.reset()

                #Clear Board
                if event.key == pygame.K_x:
                    for row in grid:
                        for spot in row:
                            spot.reset()
                    start =  None
                    end = None
                    make_border(grid)

            #Switch Algorithm Button
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if button.collidepoint(mouse_pos):
                    if a == True:
                        a = False
                        bf = True
                        df = False
                    elif bf == True:
                        a = False
                        bf = False
                        df = True
                    else:
                        df = False
                        a = True
                        bf = False

    pygame.quit()

main(WIN, WIDTH)