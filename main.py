"""
In this assignment you will implement and compare different search strategies
for solving the n-Puzzle, which is a generalization of the 8 and 15 puzzle to
squares of arbitrary size (we will only test it with 8-puzzles for now). 
"""
import time
def state_to_string(state):
    row_strings = [" ".join([str(cell) if cell>0 else " "for cell in row]) for row in state]
    return "\n".join(row_strings)

def swap_cells(state, i1, j1, i2, j2):
    """
    Returns a new state with the cells (i1,j1) and (i2,j2) swapped. 
    """
    value1 = state[i1][j1]
    value2 = state[i2][j2]
    new_state = []
    for row in range(len(state)): 
        new_row = []
        for column in range(len(state[row])): 
            if row == i1 and column == j1: 
                new_row.append(value2)
            elif row == i2 and column == j2:
                new_row.append(value1)
            else: 
                new_row.append(state[row][column])
        new_state.append(tuple(new_row))
    return tuple(new_state)
    
def get_successors(state):
    child_states = []
    for row in range(len(state)):
        for column in range(len(state[row])):
            if state[row][column] == 0:
                if column < len(state)-1: # Left 
                    new_state = swap_cells(state, row,column, row, column+1)
                    child_states.append(("Left",new_state))
                if column > 0: # Right 
                    new_state = swap_cells(state, row,column, row, column-1)
                    child_states.append(("Right",new_state))
                if row < len(state)-1:   #Up 
                    new_state = swap_cells(state, row,column, row+1, column)
                    child_states.append(("Up",new_state))
                if row > 0: # Down
                    new_state = swap_cells(state, row,column, row-1, column)
                    child_states.append(("Down", new_state))
                break
    return child_states
            
def goal_test(state):   
    counter = 0
    for row in state:
        for cell in row: 
            if counter != cell: 
                return False 
            counter += 1
    return True

def bfs(state):      
  parents = {} 
  actions = {} 
  tabu = [] 
  tabu.append(state)
  queue = []
  queue.append(state)
  visited_state_count = 0 
  while len(queue) > 0:
    p = queue.pop(0) 
    visited_state_count += 1
    options = get_successors(p)
    for i in options:
      if i[1] not in tabu:
        actions[i[1]] = i[0] 
        parents[i[1]] = p 
        if(goal_test(i[1])):
          print("Total states visited", visited_state_count)
          print(state_to_string(i[1]))
          return get_solution(i[1], parents, actions)
        else:
          tabu.append(i[1])
          queue.append(i[1])
  return None 

def dfs(state):    
  parents = {} 
  actions = {} 
  tabu = set() 
  tabu.add(state)
  stack = []
  stack.append(state)
  visited_state_count = 0 
  while len(stack) > 0:
    p = stack.pop() 
    visited_state_count += 1
    options = get_successors(p)
    for i in options: #dfs is good
      if i[1] not in tabu:
        actions[i[1]] = i[0] 
        parents[i[1]] = p 
        if(goal_test(i[1])):
          #print(state_to_string(i[1]))
          print("Total states visited", visited_state_count)
          print(state_to_string(i[1]))
          return get_solution(i[1], parents, actions)
        else:
          tabu.add(i[1])
          stack.append(i[1])
  return None 

def misplaced_heuristic(state):
  count = 0
  num_equiv = 0
  for i in state:
    for num in i:
      #print(num, num_equiv) 
      if num != num_equiv and num != 0: 
        count+=1
      num_equiv +=1 
  return count

def manhattan_heuristic(state):
  #print(state)
  count = 0
  num_equiv = 0
  for i in state:
    for num in i:
      #print(num, num_equiv)
      #print(unweighted_single_shortest_path(graph,num,num_equiv))
      count += len(unweighted_single_shortest_path(graph,num_equiv,num)[0])
      num_equiv +=1 
  return count

graph = \
{0: [1, 3],
 1: [0, 2, 4],
 2: [1, 5],
 3: [0, 4, 6], 
 4: [1, 3, 5, 7],
 5: [2, 4, 8],
 6: [3, 7],
 7: [4, 6, 8],
 8: [5, 7]}

def unweighted_single_shortest_path(graph, source, target):
  prev = {}
  cost = {}
  q = []
  tabu = []
  q.append(source)
  tabu.append(source)
  cost[source] = 0
  while len(q) > 0:
    v = q.pop(0)
    value = list(graph.get(v))
    for w in value:
      if w not in tabu:
        prev[w] = v
        tabu.append(w)
        q.append(w) 
        cost[w] = cost[v] + 1 
  return find_path(prev, source, target)

def find_path(prev, source, target):
  current = target
  result = []
  while current!=source:
    result.append(current)
    current = prev[current]
  return result, current

def get_solution(state, parents, actions):
    solution = []
    while state in parents: 
      action = actions[state]
      solution.append(action)
      state = parents[state]
    return solution[::-1]

def best_first(state, heuristic = misplaced_heuristic):
  # You might want to use these functions to maintain a priority queue
  # You may also use your own heap class here
  from heapq import heappush
  from heapq import heappop

  parents = {}
  actions = {}
  tabu = [] 
  ##tabu.append([0, state])
  tabu.append(state)
  queue = []
  ##queue.append([0, state])
  queue.append(state)
  options = []
  visited_state_count = 0 
  while len(queue) > 0:
    p = queue.pop(0) 
   # print("QUEUE ITEM ", p, "QUEUE ", queue)
    visited_state_count += 1
   ## move_options = get_successors(p[1])
    move_options = get_successors(p)
    for i in move_options:
      #print(i[1])
      #print(heuristic(i[1]))
      h = heuristic(i[1])
      heappush(options,[h,i])
    options.sort()
    #print(options)
    #for i in options:
    ##val = heappop(options)
    val = heappop(options)
    #print(val[1][1])
      #if i[1] not in tabu:
    while (val[1])[1] in tabu:
      val = heappop(options)
      #print(val[1][1])
    ##if (val[1])[1] not in tabu:
    #else: 
     #   actions[i[1]] = i[0] 
     # actions[(val[1])[1]] = (val[1])[0]
    actions[(val[1])[1]] = (val[1])[0]
    #    parents[i[1]] = p 
     # parents[(val[1])[1]] = p
    parents[(val[1])[1]] = p
    #    if(goal_test(i[1])):
    if(goal_test((val[1])[1])):
      print("Total states visited", visited_state_count)
      print(state_to_string((val[1])[1]))
      return get_solution((val[1])[1], parents, actions)
        #return get_solution((val[1])[1], parents, actions)
        #return state_to_string((val[1])[1])
     #   else:
    else:
     #     tabu.append([h, i[1]])
        ##tabu.append([h,(val[1])[1]])
      tabu.append((val[1])[1])
     #     queue.append([h, i[1]])
       ## queue.append([h,(val[1])[1]])
      queue.append((val[1])[1])
    #options = []
  return None 


def astar(state, heuristic = misplaced_heuristic):
    # You might want to use these functions to maintain a priority queue
    # You may also use your own heap class here
  from heapq import heappush
  from heapq import heappop
  parents = {}
  actions = {}
  costs = {}
  costs[state] = 0
  tabu = [] 
  tabu.append(state)
  queue = []
  queue.append(state)
  options = []
  #distance = 0
  
  visited_state_count = 0 
  while len(queue) > 0:
    p = queue.pop(0) 
    visited_state_count += 1
    move_options = get_successors(p)
    for i in move_options:
      chCost = costs[p] +1
      costs[i[1]] = chCost
      #distance += 1 # are we implementing this correctly?
      h = heuristic(i[1])
      heappush(options,[chCost+h,i])
    options.sort()
    val = heappop(options)
    while (val[1])[1] in tabu:
      val = heappop(options) 
    actions[(val[1])[1]] = (val[1])[0]
    parents[(val[1])[1]] = p
    if(goal_test((val[1])[1])):
      print("Total states visited", visited_state_count)
      print(state_to_string((val[1])[1]))
      return get_solution((val[1])[1], parents, actions)
    else:
      tabu.append((val[1])[1])
      queue.append((val[1])[1])
  return None
 
def print_result(solution):
    """
    Helper function to format test output. 
    """
    if solution is None: 
        print("No solution found.")
    else: 
        print("Solution has {} actions.".format(len(solution)))

if __name__ == "__main__":
    #Easy test case
   # test_state = ((1, 4, 2),
    #              (0, 5, 8), 
    #              (3, 6, 7)) 
    # We created a medium example test case
    #test_state = ((1, 4, 2),
    #              (6, 5, 8), 
    #              (7, 3, 0)) 
    #More difficult test case
    test_state = ((7, 2, 4),
                  (5, 0, 6), 
                  (8, 3, 1))  
    print(state_to_string(test_state))
    print()
    print(manhattan_heuristic(test_state))
    #print(misplaced_heuristic(test_state))
   # print("====BFS====")
    #start = time.time()
    #print(start)
    #solution = bfs(test_state)
    #print(solution) # not in class
    #end = time.time()
    #print_result(solution)
    #print() # not in class
    #if solution is not None:
    #    print(solution)
    #print("Total time: {0:.3f}s".format(end-start))
  
    print() 
    print("====DFS====") 
    start = time.time()
    solution = dfs(test_state)
    end = time.time()
    #print(solution)
    print_result(solution)
    print("Total time: {0:.3f}s".format(end-start))
    print() 

    print("====Greedy Best-First (Misplaced Tiles Heuristic)====") 
    start = time.time()
    solution = best_first(test_state, misplaced_heuristic)
    print(solution)
    print_result(solution)
    end = time.time()
    print("Total time: {0:.3f}s".format(end-start))
    
    print() 
    print("====A* (Misplaced Tiles Heuristic)====") 
    start = time.time()
    solution = astar(test_state, misplaced_heuristic)
    end = time.time()
    print_result(solution)
    print("Total time: {0:.3f}s".format(end-start))

    print() 
    print("====A* (Total Manhattan Distance Heuristic)====") 
    start = time.time()
    solution, states_expanded, max_frontier = astar(test_state, manhattan_heuristic)
    end = time.time()
    print_result(solution, states_expanded, max_frontier)
    print("Total time: {0:.3f}s".format(end-start))