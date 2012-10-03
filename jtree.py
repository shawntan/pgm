import numpy as np

vertices = {
		1 : np.array([0.7,0.3]),
		3 : np.array([0.7,0.3]),
		5 : np.array([0.7,0.3]),
		2 : np.array([0.1,0.9]),
		4 : np.array([0.1,0.9]),
		6 : np.array([0.1,0.9])
	}

edges = {
		(1, 2): np.array([[1,0.5],[0.5,1]]),
		(1, 3): np.array([[1,0.5],[0.5,1]]),
		(2, 4): np.array([[1,0.5],[0.5,1]]),
		(2, 5): np.array([[1,0.5],[0.5,1]]),
		(3, 6): np.array([[1,0.5],[0.5,1]])
	}

def edge(s,t):
	global edges
	if edges.has_key((s,t)): return edges[s,t]
	elif edges.has_key((t,s)): return edges[t,s]

def construct_tree(node,parent = None):
	global edges,vertices
	children = {}
	for i in vertices:
		if i == node or i == parent: continue
		elif edge(node,i) != None:
			children[i] = construct_tree(i, parent=node)
	return children

def sum_product(root):
	global vertices
	#root_node = construct_tree(root)
	messages  = {}
	for e in neighbours(root):
		collect(messages,root,e)
	for e in neighbours(root):
		distribute(messages,root,e)
	
	return {i:compute_marginal(messages,i) for i in vertices}

def neighbours(j, exclude = None):
	global vertices
	N = [k for k in vertices if edge(k,j) != None and k != exclude]
	return N
def collect(m,i,j):
	global vertices
	for k in neighbours(j,exclude=i):
		collect(m,j,k)
	dim_i = vertices[i].shape[0]
	for x_i in range(dim_i): send_message(m,j,i,x_i)

def distribute(m,i,j):
	global vertices
	dim_j = vertices[j].shape[0]
	for x_j in range(dim_j): send_message(m,i,j,x_j)

	for k in neighbours(j,exclude=i):
		distribute(m,j,k)

def send_message(m,j,i,x_i):
	global vertices
	dim_j = vertices[j].shape[0]
	total = 0
	for x_j in range(dim_j): #sum over all values of x_j
		phi_j  = vertices[j][x_j]
		phi_ij = edge(i,j)[x_i][x_j]
		m_ij = 1
		for k in neighbours(j,exclude=i): m_ij *= m[k,j,x_j]
	
		total += phi_j * phi_ij * m_ij
	
	m[j,i,x_i] = total

def compute_marginal(m,i):
	global vertices
	p = np.copy(vertices[i])
	dim_i = p.shape[0]
	for x_i in range(dim_i):
		for j in neighbours(i):
			p[x_i] *= m[j,i,x_i]

	total = sum(p)
	for i in range(dim_i):
		p[i] = p[i]/total
	return p




