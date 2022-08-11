#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import powerlaw
from statistics import stdev
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import itertools
import scipy.stats as stats
from itertools import combinations
import math
import random


# In[2]:


# ! pip install powerlaw


# In[3]:


pathway_ID = ['03320', '04010', '04012', '04014', '04015', '04020', '04022', '04024', '04062', '04064',
              '04066', '04068', '04071', '04072', '04115', '04150', '04151', '04152', '04310', '04330',
              '04340', '04350', '04370', '04371', '04390', '04392', '04620', '04621', '04622', '04630',
              '04660', '04662', '04722', '04910', '04912', '04915', '04917', '04919', '04920', '04921',
              '04926', '04933']
len(pathway_ID)


# ## Function

# In[4]:


def EGtest(DS_input):
    if np.sum(DS_input) % 2 == 1: 
        return 'failure'
    if np.sum(DS_input) != 0 & 0 in DS_input: # ensure 'delete' function could work
        DS_input = np.delete(DS_input, np.where(DS_input == 0)) # remove '0' in DS
    if np.sum(DS_input) > len(DS_input)*(len(DS_input) - 1):  # exclude multiple edge scenario
        return 'failure'
    DS_input = np.sort(DS_input)[::-1]
    for index in range(len(DS_input)): # check EG theorem
        k = index + 1
        tmp_DS_input = np.array([k if DS_input[x] > k and x >= k else DS_input[x] for x in range(len(DS_input))]) # replace element if it's larger than k
        if sum(tmp_DS_input[0:k]) > k*(k-1) + np.sum(tmp_DS_input[k:len(DS_input)]):
            return 'failure'
    return 'success' 

def rightmost_adj(DS_input):
    non_zero_degree_node_index = np.array([i for i, x in enumerate(DS_input) if x > 0])
    if 0 in DS_input:
        DS_input = np.delete(DS_input, np.where(DS_input == 0))
    order_index = np.argsort(-DS_input) # save index for adjacency matrix
    DS_input = np.sort(DS_input)[::-1]
    rightmost_adj_set = np.array([], dtype = int)
    for non_leading_node in range(1,len(DS_input))[::-1]: # run from last-second node to second node 
        tmp_DS = np.copy(DS_input)
        # connect 'non_leading_node' with leading node
        tmp_DS[0] -= 1
        tmp_DS[non_leading_node] -= 1
        # run EG test for checking connection is ok
        if tmp_DS[0] != 0:
            # connect leading node with first i nodes
            DS_for_test = np.array([tmp_DS[i] - 1 if i <= tmp_DS[0] else tmp_DS[i] for i in range(1, len(tmp_DS))])
        else:
            DS_for_test = np.copy(tmp_DS)
        if EGtest(DS_for_test) == 'success':
            rightmost_adj_set = np.append(rightmost_adj_set, non_leading_node)
            if tmp_DS[0] == 0:
                break  #stop when degree of leading node is 0
            DS_input = np.copy(tmp_DS)
    rightmost_adj_set = np.array(non_zero_degree_node_index[order_index[rightmost_adj_set]])   #re-order
    return rightmost_adj_set

def left_adj(current_adj_mat, DS_input_original, DS_input_current, rightmost_adj, leading_node_index):
    leading_node_degree = DS_input_current[leading_node_index]
    tmp_DS = np.copy(DS_input_current)
    tmp_DS[leading_node_index] = 0
    order_index = np.argsort(-tmp_DS)
    tmp_DS = np.sort(tmp_DS)[::-1]
    # remove node withh zero degree
    non_zero_DS = np.delete(tmp_DS, np.where(tmp_DS == 0))
    number_of_min = len(np.where(non_zero_DS == min(non_zero_DS))[0])
    non_min_node_index = np.where(non_zero_DS != min(non_zero_DS))[0]
    number_of_non_min = len(non_min_node_index)
    # parameters for for-loop
    tmp_left_adj_set = [[0]*leading_node_degree] 
    i_start = max(leading_node_degree - number_of_min, 0) # 0要檢查 因為python index 0 = R inedx 1
    i_end = max(min(number_of_non_min, leading_node_degree), 0) + 1
    # deal with duplicated structure
    duplicate_marker = [mat.copy() for mat in current_adj_mat]
    [duplicate_marker[i].append(DS_input_original[i]) for i in range(len(DS_input_original))]
    duplicated_index = [0]*len(duplicate_marker)
    for i in range(len(duplicate_marker)):
        for j in range(len(duplicate_marker)):
            if duplicate_marker[j] == duplicate_marker[i]:
                duplicated_index[i] = j
    duplicated_index = [duplicated_index[order_index[i]] for i in range(len(order_index))]
    for i in range(i_start, i_end):
        # first part (for non-min degree node)
        if i == 1 and number_of_non_min == 1:
            first_part = [list(non_min_node_index)]
        elif i != 0:
            first_part = [list(l) for l in combinations(non_min_node_index,i)]
            duplicated_mat = [0]*len(first_part)
            for j in range(len(first_part)):
                duplicated_mat[j] = [duplicated_index[first_part[j][k]] for k in range(len(first_part[j]))]
            unique_index = []
            unique_value = []
            for j in range(len(duplicated_mat))[::-1]:
                x = duplicated_mat[j]
                if x not in unique_value:
                    unique_value.append(x)
                    unique_index.append(j)
            first_part = [first_part[m] for m in range(len(first_part)) if m in unique_index]
        # second part (for min degree node)
        if i != leading_node_degree:
            min_degree_node = np.where(non_zero_DS == min(non_zero_DS))[0]
            if len(min_degree_node) == 1:
                second_part = [list(min_degree_node)]
            else:
                second_part = [list(l) for l in combinations(min_degree_node,leading_node_degree-i)]
            
            duplicated_mat = [0]*len(second_part)
            for j in range(len(second_part)):
                duplicated_mat[j] = [duplicated_index[second_part[j][k]] for k in range(len(second_part[j]))]
            unique_index = []
            unique_value = []
            for j in range(len(duplicated_mat))[::-1]:
                x = duplicated_mat[j]
                if x not in unique_value:
                    unique_value.append(x)
                    unique_index.append(j)
            second_part = [second_part[m] for m in range(len(second_part)) if m in unique_index]
        #combine first part & second part
        if i == 0:
            combine_two_part = second_part
        elif i == leading_node_degree:
            combine_two_part = first_part
        else:
            combine_two_part = [x + y for x in first_part for y in second_part]
        tmp_left_adj_set = tmp_left_adj_set + combine_two_part
    tmp_left_adj_set.remove(tmp_left_adj_set[0])   # delete offset
    ## check colex order
    # calculate colex score (ex: colex_order_rightmost = [0 2 3] -> clex_score_rightmost = [2^0 2^2 2^3] = [1 4 8])
    # ex: tmp_left_adj_set = [[1,2,3],[0,2,3]] -> score = 14, 13
    #     colex_order_rightmost = [0, 2, 3] -> score = 13
    # then exclude [1,2,3]
    mapping_index = [np.where(tmp_DS == tmp_DS[k])[0][0] for k in range(len(tmp_DS))]

    colex_order_rightmost = [np.where(order_index == rightmost_adj[i])[0][0] for i in range(len(rightmost_adj))]
    colex_score_rightmost = np.sum([2**(mapping_index[colex_order_rightmost[i]]) for i in range(len(colex_order_rightmost))])
    colex_score_left = []
    for i in range(len(tmp_left_adj_set)):
        colex_score_i = np.sum([2**(mapping_index[tmp_left_adj_set[i][j]]) for j in range(len(colex_order_rightmost))])
        colex_score_left.append(colex_score_i)
    check_to_the_left = np.array(colex_score_left <= colex_score_rightmost, dtype=bool)
    # filter
    tmp_left_adj_set = np.asarray(tmp_left_adj_set)
    left_adj_set = tmp_left_adj_set[check_to_the_left]
    left_adj_set = [list(order_index[k]) for k in left_adj_set]
    return left_adj_set

#adjacency matrix
def connect_adj_set(leading_node_index, current_adj_mat, adj_set):
    output_mat = []
    for ii in range(len(adj_set)):
        tmp_mat = [mat.copy() for mat in current_adj_mat]
        for jj in range(len(adj_set[0])):
            tmp_mat[leading_node_index][adj_set[ii][jj]] = tmp_mat[adj_set[ii][jj]][leading_node_index] = 1
        output_mat.append([mat.copy() for mat in tmp_mat])
    return output_mat

##generate network(main algorithm)
def net_gen (original_DS):
    sum_DS = np.sum(original_DS)
    rows = cols = len(original_DS)
    imcomplete_adj_mat = [[[0]*cols for i in range(rows)]]
    # imcomplete_adj_mat.remove(imcomplete_adj_mat[-1])     # approach to remove matrix
    complete_adj_mat = []
    while len(imcomplete_adj_mat) != 0 or len(complete_adj_mat) == 0:
        last_matrix = imcomplete_adj_mat[-1]
        imcomplete_adj_mat.remove(imcomplete_adj_mat[-1])
        current_DS = original_DS - np.array([sum(last_matrix[i]) for i in range(len(original_DS))])
        if sum(current_DS != 0) > 1:  # probelm: sequence with one value only
            leading_node = np.where(current_DS == max(current_DS))[0][0]      # first [0] represents all max value in thhe sequence
            # righhtmost adjacency set
            rightmost_adj_set = rightmost_adj(DS_input = current_DS)
            # left adjacency set
            left_adj_set = left_adj(current_adj_mat = last_matrix, 
                                     DS_input_original = original_DS, 
                                     DS_input_current = current_DS, 
                                     rightmost_adj = rightmost_adj_set, 
                                     leading_node_index = leading_node)
            new_matrix = connect_adj_set(leading_node_index = leading_node, 
                                          current_adj_mat = last_matrix, 
                                          adj_set = left_adj_set)
            for kk in range(len(new_matrix)):
                # complete matrix
                if sum([(sum(x)) for x in new_matrix[kk]]) == sum_DS:
                    complete_adj_mat.append(new_matrix[kk])
                else:
                    imcomplete_adj_mat.append(new_matrix[kk])
    return complete_adj_mat

##Draw
def show_graph_with_labels(adjacency_matrix, input_label=""):
    gr=nx.from_numpy_matrix(adjacency_matrix)
    graph_pos=nx.spring_layout(gr,k=0.50,iterations=50)
    # nodes
    nx.draw_networkx_nodes(gr, graph_pos,
                           node_color='#1f78b4',
                           node_size=220,
                           alpha=0.6)
    # edges
    nx.draw_networkx_edges(gr, graph_pos, width=2.0, alpha=0.3)
    # labels
    labels={}
    if len(input_label):
        for label_index in range(len(input_label)):
            labels[label_index] = str(input_label[label_index])
        nx.draw_networkx_labels(gr,graph_pos,labels)
    else:
        nx.draw_networkx_labels(gr,graph_pos)
    plt.show()

##permutation
def Network_score_with_cor(Candidate_network, Structure_score, Correlation_mat, Gene_list, Network_ranking_size, Permute_size = 2):
    # parameter
    tmp_structure_score = Structure_score.copy()
    tmp_structure_score.sort()
    final_gene_label = []
    final_net_score = []
    final_structure_list = []
    Graph_list = []
    unique_network_count = 0
    structure_count = 0
    permutation_time = Permutation_time_fun(len(Gene_list), Permute_size)
    # execution
    # output first 500 network by default
    while unique_network_count < Network_ranking_size and structure_count < len(tmp_structure_score):
        # ignore duplicated networks 
        net_index = np.where(Structure_score == tmp_structure_score[structure_count])[0][0]
        structure_count += 1
        current_net = np.array(Candidate_network[net_index])
        # check isomorphism
        isomorphic_result = False
        if unique_network_count != 0:
            #backward checking
            current_graph = nx.from_numpy_matrix(current_net)
            for iso_test in range(unique_network_count)[::-1]:
                isomorphic_result = nx.is_isomorphic(Graph_list[iso_test], current_graph)
                if isomorphic_result == True: 
                    break
    
        # calculate network score if network is unique
        if isomorphic_result == False: 
            Graph_list.append(nx.from_numpy_matrix(current_net))
            final_structure_list.append(current_net)
            unique_network_count += 1
            current_degree_seq = sum(np.array(current_net))

            # labeling
            # start with max degree node
            max_degree_node_index = np.where(current_degree_seq == max(current_degree_seq))[0][0]
            # print('max degree node is located in "{}" index'.format(max_degree_node_index))
            cor_sum = sum(Correlation_mat)
            # print('sum of correlation for each gene: {}'.format(cor_sum))
            max_cor_sum_gene = Gene_list[np.where(cor_sum == max(cor_sum))[0][0]]
            # print('Labeling max degree node by gene ({}) with max sum of correlation.'.format(max_cor_sum_gene))

            # (1) assign gene with max sum of correlation to node with max degree
            gene_label = np.array(Gene_list.copy())
            permute_candidate = np.array(np.where(gene_label == max_cor_sum_gene)[0][0])
            permute_candidate = np.append(permute_candidate, max_degree_node_index)
            # permute candidate one and candidate two
            gene_label[permute_candidate] = gene_label[permute_candidate[::-1]]
            # new gene co-expression matrix
            gene_cor = np.copy(Correlation_mat)
            gene_cor[:,permute_candidate] = gene_cor[:,permute_candidate[::-1]]
            gene_cor[permute_candidate,:] = gene_cor[permute_candidate[::-1],:]

            # (2) calculate initial score
            net_score = 0
            for jj in range(len(Correlation_mat)):
                net_score = net_score + sum(gene_cor[jj]*current_net[jj])

            # (3) permutation (100 times by default)
            # select two index to execute permutation
            for i in range(permutation_time):
                tmp_gene_label = np.copy(gene_label)
                permute_candidate = random.sample(range(len(Gene_list)), Permute_size)
                after_permute = permute_candidate.copy() 
                while after_permute == permute_candidate:  
                    after_permute = random.sample(permute_candidate, Permute_size) 
                # permute label
                tmp_gene_label[permute_candidate] = tmp_gene_label[after_permute]
                # new gene co-expression matrix
                tmp_gene_cor = np.copy(gene_cor)
                tmp_gene_cor[:, permute_candidate] = tmp_gene_cor[:, after_permute]
                tmp_gene_cor[permute_candidate,:] = tmp_gene_cor[after_permute,:]
                
                # calculate new score
                tmp_score = 0
                for jj in range(len(Correlation_mat)):
                    tmp_score = tmp_score + sum(tmp_gene_cor[jj]*current_net[jj])

                # if score is higher than updating
                if (tmp_score > net_score):
                    gene_label = tmp_gene_label.copy()
                    net_score = tmp_score.copy()

            # (4) save result to a list
            final_gene_label.append(gene_label)
            final_net_score.append(net_score)
    
    return final_structure_list, final_gene_label, final_net_score

def Permutation_time_fun(total_gene, candidate_number):
    comb = math.factorial(total_gene) // (math.factorial(candidate_number) * math.factorial(total_gene - candidate_number))
    return 3*comb


def truncated_power_law(alpha, maximum_value):
    x = np.arange(1, maximum_value+1, dtype='float')
    pmf = 1/x**alpha
    pmf /= pmf.sum()
    return stats.rv_discrete(values=(range(1, maximum_value+1), pmf))
## cannot exist 0


# In[5]:


# Extract the information we need from KGML file (KEGG human biological pathway)

def KGML_to_Matrix(pathway_name, KGML_file_path = "", save_path = ""):
    # First, import the KGML file and rearrange it into 2 parts (node information and relation)
    # The xml.etree.ElementTree package is used to read the KGML file.
    
    # Read KGML file
    tree = ET.parse(KGML_file_path)
    root = tree.getroot()
    
    # Extract information from KGML file including 2 parts
    data_raw = []
    for child in root:
        data = {}
        data[child.tag] = child.attrib
        data["graphics"] = []
        data["component"] = []
        data["subtype"] = []
        for children in child:
            # Third layer of the KGML file
            data[children.tag].append(children.attrib)
        data_raw.append(data)
    
    # Remove the empty set
    for ii in range(len(data_raw)):
        for key in list(data_raw[ii].keys()):
            if not data_raw[ii].get(key): ##Empty set = False
                del data_raw[ii][key]
    
    # rearrange into 2 parts (node information and relation)
    # Extract entry part (node information) and produce "node_detail" file in output.
    data_entry_raw = []
    # Extract information about !§relation!‥ between nodes, and produce "adj_matrix" file in output.
    data_relation_raw = []
    for ii in range(len(data_raw)):
        if "entry" in data_raw[ii].keys():
            data_entry_raw.append(data_raw[ii])
        if "relation" in data_raw[ii].keys():
            data_relation_raw.append(data_raw[ii])
            
            
    # Part 1: processing entry part (node information)
    # Extract useful variables (id / name / type / node_name) from entry part and transfer to data.frame
    # id         -> index of nodes in pathway (node id)
    # name       -> entryID of all genes/compounds inside the node (such as 7583, 9641, 255,...)
    # type       -> type of components inside the node (such as gene, compound or group)
    # node_name  -> symbol of all genes/compounds inside the node (such as ESR1, ESR2,...)
    
    # Replace the repeated names in <graphics> with other name
    for ii in range(len(data_entry_raw)):
        if "name" in data_entry_raw[ii]["graphics"][0]:
            data_entry_raw[ii]["graphics"][0]["node_name"] = data_entry_raw[ii]["graphics"][0].pop("name")
        if "type" in data_entry_raw[ii]["graphics"][0]:
            data_entry_raw[ii]["graphics"][0]["type1"] = data_entry_raw[ii]["graphics"][0].pop("type")
    
    # Merge each dictionary in data_entry_raw
    data_entry = []
    for ii in range(len(data_entry_raw)):
        data_entry.append({**data_entry_raw[ii]["entry"], **data_entry_raw[ii]["graphics"][0]})
    
    # Transfer data type to dataframe
    data_entry = pd.DataFrame(data_entry)
    data_entry = data_entry[["id","name","type","node_name"]]
    
    # "component.id", a variable for the index of nodes inside this group.
    # "number", the variable for the number of nodes inside this group.
    # "data_entry_new", a data-frame with an additional variable, "component", 
    # saves as (id / name / type / node_name / component)
    number = []
    for ii in range(len(data_entry_raw)):
        if "component" in data_entry_raw[ii]:
            number.append(len(data_entry_raw[ii]["component"]))
        else:
            number.append(int(1))
    component_id = []
    for ii in range(len(number)):
        for jj in range(number[ii]):
            component_id.append(range(len(number))[ii])  
            
    data_entry_component = pd.DataFrame(data_entry.iloc[component_id ,:])
    
    component = []
    for ii in range(len(data_entry_raw)):
        if "component" not in data_entry_raw[ii]:
            component.append(float("NaN"))
        if "component" in data_entry_raw[ii]:
            for jj in data_entry_raw[ii]["component"]:
                component.append(jj["id"])
    
    component = pd.DataFrame(component, index = component_id)
    component.columns = ["component"]
    data_entry_new = pd.concat([data_entry_component, component], axis = 1)
    
    # Filter out "map" nodes and preserve "gene" nodes and "compound" nodes only
    data_entry = data_entry.loc[data_entry["type"] != "map"].reset_index()
    
    # Processing repeated nodes
    unique_correspond = None
    unique_entryname = None
    unique_entryid = None

    unique_entryname = data_entry["name"].unique()
    unique_erntryid = data_entry["id"][data_entry["name"].duplicated() == False]
    
    match = lambda a, b: [ b.index(x) if x in b else None for x in a ]
    
    position_repeat = match(list(data_entry["name"][data_entry["name"].duplicated()]), list(data_entry["name"]))
    
    Var1 = list(data_entry["id"][data_entry["name"].duplicated()])
    Var2 = list(data_entry["id"][position_repeat])
    unique_correspond = {"Var1":Var1, "Var2":Var2}
    unique_correspond = pd.DataFrame(unique_correspond)
    
    # Part 2: processing relationship part (relationship between nodes)
    # Extract useful variables (entry1 / entry2) from data_relation and transfer to data.frame
    # For example, (entry1 / entry2) = (A node / B node) represents there is an edge from A node to B 
    # node. (A -> B) where entry1 and entry2 record node id
    data_relation = []
    for ii in data_relation_raw:
        data_relation.append(ii["relation"])
    data_relation = pd.DataFrame(data_relation)[["entry1", "entry2"]]
    
    # processing group
    # Break group into separate nodes and make each node connect with the others.
    # If some nodes inside the pathways do not have !§relation!‥, then remove them and return the 
    # pathway names. 
    if len(data_relation) == 0:
        return print("There are no relation in the ", pathway_name, "!")
    else:
        # separate_from records group id
        # separate_to records component id    
        separate_from = list(data_entry_new[data_entry_new["type"] == "group"]["id"])
        # transfer data type to numeric
        separate_to = list(data_entry_new["component"][data_entry_new["component"].isnull() == False])
    
        # Break group into separate nodes in the relationship part  (need to use "separate_from" & 
        # "separate_to" variable)
        # replace group with its component
        relation_new = []
        for ii in range(len(data_relation.index)):
            # for relation 1
            if [i in data_relation.iloc[ii,][0] for i in separate_from].count(True) > 0:
                relation1 = [separate_to[i] for i, v in enumerate([i in data_relation.iloc[ii,][0] for i in separate_from]) if v == True]
            else:
                relation1 = [data_relation.iloc[ii,][0]]
            # for relation 2
            if [i in data_relation.iloc[ii,][1] for i in separate_from].count(True) > 0:
                relation2 = [separate_to[i] for i, v in enumerate([i in data_relation.iloc[ii,][1] for i in separate_from]) if v == True]

            else:
                relation2 = [data_relation.iloc[ii,][1]]
            relation_new.append([(x, y) for x in relation1 for y in relation2])
        # transfer from list to dataframe    
        relation_new = pd.concat(list(map(pd.DataFrame, relation_new)))
        relation_new.columns = ["Var1", "Var2"]
        relation_new = relation_new.reset_index(drop = True)
        # make each node connected with the rest in the same group.
        # "relation_group_fn", a function combines the nodes within the same group.
        def relation_group_fn(xx):
            sub_group = [separate_to[i] for i, v in enumerate([i in xx for i in separate_from]) if v == True]
            return pd.DataFrame(list(itertools.combinations(sub_group, 2)))
        
        if len(separate_from) > 0:
            relation_group = pd.concat(list(map(pd.DataFrame, list(map(relation_group_fn, sorted(list(set(separate_from))))))))
            relation_group.columns = ["Var1", "Var2"]
            relation_group = relation_group.reset_index(drop = True)
            # reverse relation between nodes inside the same group because they should be undirected (A <-> B)
            Reverse_relation = relation_group[["Var2", "Var1"]]
            Reverse_relation.columns = ["Var1", "Var2"]
            relation_group = pd.concat([relation_group, Reverse_relation]).reset_index(drop = True)
            # Finally, combine "relation_new" and "relation_group"
            relation_new = pd.concat([relation_new, relation_group]).reset_index(drop = True)
            
        # Create an adjacency matrix
        # In part 2, the relationship part, it uses node id to record the relation.
        # In order to save the relationship into the matrix, both the row name and column name in this 
        # adjacency matrix should be node id.
        relationship = pd.DataFrame(np.zeros((len(data_entry["id"]), len(data_entry["id"])), dtype = int), index = list(data_entry["id"]), columns = list(data_entry["id"]))
        # Save the connection of nodes.
        position1 = match(relation_new["Var1"], list(data_entry["id"]))
        position2 = match(relation_new["Var2"], list(data_entry["id"]))
        relation_position = pd.DataFrame({"position1" :position1, "position2" :position2})
        for ii in range(len(relation_position.index)):
            x = relation_position.iloc[ii, 0]
            y = relation_position.iloc[ii, 1]
            relationship.iloc[x, y] = 1
            
        # Deal with repeated nodes
        # Find the position of repeated nodes in the matrix.
        pos1 = match(unique_correspond["Var1"], list(data_entry["id"])) #will be deleted the end
        pos2 = match(unique_correspond["Var2"], list(data_entry["id"])) #preserved part
        # In order to avoid that pos1 does not include anything, add this "if else" constraint
        if len(pos1) != 0:
            # marge the relationship of each node if they are the same node
            for ii in range(len(pos1)):
                relationship.iloc[pos2[ii], ] = relationship.iloc[pos2[ii], ] + relationship.iloc[pos1[ii], ]
                relationship.iloc[:, pos2[ii]] = relationship.iloc[:, pos2[ii]] + relationship.iloc[:, pos1[ii]]
        # And, delete the id in column 1 to preserve unique nodes
        if len(pos1) > 0:
            pos1_relationship = [relationship.columns[ii] for ii in pos1]
            relationship = relationship.drop(pos1_relationship, axis = 1)
            relationship = relationship.drop(pos1_relationship, axis = 0)
            # the other output (node_detail) -> record all information about unique nodes.
            data_entry.index = list(data_entry["id"])
            entry_pos1 = [data_entry["id"][ii] for ii in pos1]
            node_detail = data_entry.drop(entry_pos1, axis = 0)
            node_detail = node_detail[["name", "type", "node_name"]]
            # remove the column and row of group node in the matrix
            if list(data_entry["type"] == "group").count(True) != 0:
                delete_group = data_entry.drop(entry_pos1, axis = 0)
                delete_group_drop = [delete_group.index[ii] for ii, vv in enumerate(delete_group["type"] == "group") if vv == True]
                relationship = relationship.drop(delete_group_drop, axis = 0)
                relationship = relationship.drop(delete_group_drop, axis = 1)
                node_detail = node_detail.drop(delete_group_drop, axis = 0) 
        else:
            data_entry.index = list(data_entry["id"])
            node_detail = data_entry[["name", "type", "node_name"]]
        # Change the value which is larger than 1 to 1
        for ii in range(len(relationship.index)):
            for jj in range(len(relationship.columns)):
                if relationship.iloc[ii, jj] > 1:
                    relationship.iloc[ii, jj] = 1
        adj_matrix = relationship
            
        # save adj_matrix & node_detail    
        file_name = save_path + pathway_name + "(directed)"
        adj_matrix.to_pickle(file_name)    
        file_name = save_path + pathway_name + "(node_detail)"    
        node_detail.to_pickle(file_name)
        return "Success"


# # Simulation

# In[6]:


######################################
# section 1 --- Library establishment 
######################################
# population: 100 genes (multivariate normal)
N = 100
seed = 20200413
random.seed(seed)
# create correlation matrix
# (no need to check correlation matrix for whole library)
whole_cor = np.zeros(shape=(N,N))
component = np.repeat([0.1, 0.3, 0.7],[2970, 990, 990])
component = component.tolist()
cor_input = random.sample(component, 4950)

# generating correlation matrix
i = 0
for row in range(N):
    for col in range(row,N):
        if(row == col):
            whole_cor[row][col] = 1
        else:
            whole_cor[row][col] = cor_input[i]
            whole_cor[col][row] = cor_input[i]
            i += 1

# sample network in library
sample_net_num = 0
sample_gene_num = 10
cor_threashold = 0.3
# original_node_index: original 10 genes for each network (before removing isolated nodes)
original_node_index = []
# removing isolated nodes
node_index = []
sample_network = []
while sample_net_num < 100:
    tmp_node_index = random.sample(range(100), 10)
    tmp_node_index.sort()
    if not(tmp_node_index in original_node_index):
        tmp_cor = whole_cor[np.ix_(tmp_node_index, tmp_node_index)]
        semi_pos = np.all(np.linalg.eigvals(tmp_cor) > 0)
        
        edge_matrix = tmp_cor > cor_threashold
        edge_rate = sum(sum(edge_matrix))/(len(edge_matrix)**2)
        # if correlation matrix is semi-positive and edge count > threshold,
        # then record node index and corresponding network
        if (semi_pos == True and edge_rate > 0.15 and edge_rate < 0.2):
            sample_net_num +=1
            # record node index
            original_node_index.append(tmp_node_index)

            # corresponding network (10x10 sub-matrix for 10 sample genes)
            mean = [0]*sample_gene_num
            sample_gene_exp = np.random.multivariate_normal(mean, tmp_cor, 100)
            sample_cor_from_gene_exp = np.corrcoef(sample_gene_exp, rowvar = False)
            #print(sample_network_cor_version)
            sample_network_tmp = np.zeros(shape=(sample_gene_num, sample_gene_num))
            for row in range(sample_gene_num):
                for col in range(sample_gene_num):
                    # default threshold = 0.3
                    sample_network_tmp[row,col] = sample_cor_from_gene_exp[row, col] > cor_threashold
            # diag = 0
            sample_network_tmp[range(sample_gene_num), range(sample_gene_num)] = 0
            
            # remove isolated nodes from network and node index
            isolated_node = np.where(sum(sample_network_tmp) == 0)[0]
            isolated_node = isolated_node[::-1]
            if (len(isolated_node) > 0):
                for iso in isolated_node:
                    sample_network_tmp = np.delete(sample_network_tmp, iso, 0)
                    sample_network_tmp = np.delete(sample_network_tmp, iso, 1)
                    tmp_node_index = np.delete(tmp_node_index, iso)
            node_index.append(tmp_node_index)
            sample_network.append(sample_network_tmp)

######################################
# section 3 --- leave-one-out
######################################
# sample_network record all of library network 
for leave_one_out_index in range(10):
    leave_one_out_net = sample_network[leave_one_out_index]
    # sub correlation matrix
    leave_one_out_node_index = node_index[leave_one_out_index]
    leave_one_out_correlation = whole_cor[np.ix_(leave_one_out_node_index, leave_one_out_node_index)]
    # remove isolated nodes
    isolated_index = []
    for i in range(len(leave_one_out_correlation)):
        if sum(leave_one_out_correlation[i] > cor_threashold) < 1:
            isolated_index.append(i)
    leave_one_out_correlation = np.delete(leave_one_out_correlation, isolated_index, 0)
    leave_one_out_correlation = np.delete(leave_one_out_correlation, isolated_index, 1)

    library_network = []
    library_node_index = []
    for index in range(sample_net_num):
        if index != leave_one_out_index:
            library_network.append(sample_network[index])
            library_node_index.append(node_index[index])

    ######################################
    ## Library information
    ######################################
    # extract relationship in each sample network
    val = 1   # find 1 in matrix (relationship)
    relation_pos = []
    for i in range(len(library_network)):
        mat = library_network[i]
        for row_index in range(len(mat)):
            if val in mat[row_index]:
                col_index = np.where(mat[row_index] == val)[0]
                origin_row_index = library_node_index[i][row_index]
                origin_col_index = [library_node_index[i][col_index[j]] for j in range(len(col_index))]
                [relation_pos.append((origin_row_index, x)) for x in origin_col_index]


    # Create library information matrix (L)
    lib_info = np.zeros(shape=whole_cor.shape)
    for row_index, col_index in relation_pos:
        lib_info[row_index, col_index] = lib_info[row_index, col_index] + 1
    ######################################
    # section 4 --- estimate alpha in the power-law distribution
    ######################################
    alpha = []
    for index in range(sample_gene_num):
        current_mat = sample_network[index]
        degree_seq = [sum(row) for row in current_mat]
        degree_seq_non_zero = [i for i in degree_seq if i != 0]
        degree_seq_non_zero = np.array(degree_seq_non_zero)
        fit = powerlaw.Fit(degree_seq_non_zero, xmin=1, discrete=True)
        alpha.append(fit.power_law.alpha)

    # sampling from target power-law distribution
    estimated_alpha = sum(alpha)/len(alpha)
    sd_alpha = stdev(alpha)

    ######################################
    # section 5 --- potential network construction
    ######################################
    potential_net_size = len(leave_one_out_net)
    potential_net = []
    DS_index_for_sample_net = []
    for DS_count in range(200):
        # degree sequence generation
        EG_result = "failure"
        while(EG_result == "failure"):
            input_alpha = np.random.normal(loc = estimated_alpha, scale = sd_alpha, size = 1)
            d = truncated_power_law(alpha = input_alpha, maximum_value = potential_net_size - 1)
            sample_DS = d.rvs(size = potential_net_size)
            sample_DS.sort()
            sample_DS = sample_DS[::-1]
            EG_result = EGtest(sample_DS)
        # network construction
        sample_net = net_gen(np.array(sample_DS))
        for net_index in range(len(sample_net)):
            potential_net.append(sample_net[net_index])
            # record the DS_index for each sample_net
            DS_index_for_sample_net.append(DS_count)

    ######################################
    # section 6 --- rank by network properties 
    ######################################
    # network properties for potential network
    ave_path_len_list = []
    max_degree_centrality_list = []
    transitivity_list = []
    for net in potential_net:
        G_net = nx.from_numpy_matrix(np.array(net))
        dis_mat = nx.floyd_warshall_numpy(G_net)
        dis_mat_without_inf = np.where(np.isinf(dis_mat), len(dis_mat), dis_mat) 
        ave_path_len_list.append(dis_mat_without_inf.sum()/len(dis_mat)/len(dis_mat))
        max_degree_centrality_list.append(max(nx.degree_centrality(G_net).values()))
        transitivity_list.append(nx.transitivity(G_net))
    ave_path_len_list = np.array(ave_path_len_list)
    max_degree_centrality_list = np.array(max_degree_centrality_list)
    transitivity_list = np.array(transitivity_list)

    # network properties in library (library_network)
    library_ave_path_len = []
    library_max_degree_centrality = []
    library_transitivity = []
    for lib in library_network:
        target_net = nx.from_numpy_matrix(np.array(lib))
        # average path length
        dis_mat = nx.floyd_warshall_numpy(target_net) # distance matrix
        dis_mat_without_inf = np.where(np.isinf(dis_mat), len(dis_mat), dis_mat) 
        library_ave_path_len.append(dis_mat_without_inf.sum()/len(dis_mat)/len(dis_mat))
        # max degree centrality
        library_max_degree_centrality.append(max(nx.degree_centrality(target_net).values()))
        # transitivity
        library_transitivity.append(nx.transitivity(target_net))

    # mean of network properties
    mean_library_ave_path_len = sum(library_ave_path_len)/len(library_ave_path_len)
    mean_library_max_degree_centrality = sum(library_max_degree_centrality)/len(library_max_degree_centrality)
    mean_library_transitivity = sum(library_transitivity)/len(library_transitivity)

    # network selection by network properties
    dis_mat = nx.floyd_warshall_numpy(target_net)
    dis_mat_without_inf = np.where(np.isinf(dis_mat), len(dis_mat), dis_mat) 
    ave_path_len_measure = ave_path_len_list - mean_library_ave_path_len
    max_degree_centrality_list = max_degree_centrality_list - mean_library_max_degree_centrality
    transitivity_measure = transitivity_list - mean_library_transitivity
    # dissimilarity
    measure = ave_path_len_measure * ave_path_len_measure + max_degree_centrality_list * max_degree_centrality_list + transitivity_measure * transitivity_measure
    
    # convergence plot
    DS_index_np = np.array(DS_index_for_sample_net)
    measure_np = np.array(measure)
    DS_index_unique = np.unique(DS_index_np)
    min_dissimilarity = min(measure_np[np.where(DS_index_np == 0)[0]])
    all_min_dissimilarity = min_dissimilarity.copy()
    each_DS_min_dissimilarity = min_dissimilarity.copy()
    for i in DS_index_unique[1:]:
        tmp_dissimilarity = measure_np[np.where(DS_index_np == i)[0]]
        min_dissimilarity = min(np.append(tmp_dissimilarity, min_dissimilarity))
        all_min_dissimilarity = np.append(all_min_dissimilarity, min_dissimilarity)
        each_DS_min_dissimilarity = np.append(each_DS_min_dissimilarity, min(tmp_dissimilarity))
    fig, axs = plt.subplots(2, figsize=(8, 6), dpi=800, facecolor='w', edgecolor='k')
    fig.suptitle('Convergence plot', fontsize=12)
    axs[0].plot(DS_index_unique, each_DS_min_dissimilarity)
    axs[1].plot(DS_index_unique, all_min_dissimilarity)
    plt.show()

    
    
    ######################################
    # Labeling by permutation and Calculating network score
    ######################################
    final_gene_label = []
    final_net_score = []
    Graph_list = []
    unique_network_index = []
    structure_count = 0


    Sturcture_output, Gene_label_output, Score_output = Network_score_with_cor(Candidate_network = potential_net, 
                                                                               Structure_score = measure, 
                                                                               Correlation_mat = leave_one_out_correlation, 
                                                                               Gene_list = leave_one_out_node_index,
                                                                               Network_ranking_size = len(potential_net)/10)  # consider 10% network structure with lower dissimilarity  

    sorted_index = np.argsort(Score_output)
    sorted_index = sorted_index[::-1]
    sorted_sturcture_output = []
    sorted_score_output = []
    sorted_gene_label_output = []
    for i in sorted_index:
        sorted_sturcture_output.append(Sturcture_output[i])
        sorted_score_output.append(Score_output[i])
        sorted_gene_label_output.append(Gene_label_output[i])

    ######################################
    # Compare leave-one-out network with potential networks
    ######################################
    sensitivity = []
    for i in range(len(sorted_sturcture_output)):
        match_index = []
        tmp_gene_label = np.array(sorted_gene_label_output[i])
        for j in range(len(leave_one_out_node_index)):
            match_index.append(np.where(tmp_gene_label == leave_one_out_node_index[j])[0][0])
        shuffle_structure = np.copy(sorted_sturcture_output[i])
        shuffle_structure = shuffle_structure[:, match_index]
        shuffle_structure = shuffle_structure[match_index,:]
        Sturcture_output[i] = shuffle_structure
        # sensitivity
        array_target = leave_one_out_net.reshape(-1)
        array_potential = shuffle_structure.reshape(-1)
        edge_pos = np.where(array_target == 1)[0]
        sensitivity.append(sum(array_potential[edge_pos])/len(array_potential[edge_pos]))
    print('Total number of potential network structure: {}'.format(len(potential_net)))
    print('The {}-th network structure has the highest sensitivity, {}.'.format(np.argmax(np.array(sensitivity)), round(max(sensitivity), 2)))
    plt.figure(figsize=(8, 6), dpi=800)
    plt.plot(sensitivity)
    plt.suptitle('sensitivity', fontsize=12)
    plt.show()


# # Applicaion

# In[7]:


# import KEGG pathway (KGML files)
for ii in range(len(pathway_ID)):
    pathway_name = "hsa" + pathway_ID[ii]
    KGML_directory = "/Users/liaochenpo/Desktop/Meeting/Network construction/KEGG application/pathway/hsa" + pathway_ID[ii] + ".xml"
    # KGML_directory = "C:/Users/User/Desktop/Network construction/KEGG application/pathway/hsa" + pathway_ID[ii] + ".xml"
    save_directory = "//Users/liaochenpo/Desktop/Meeting/Network construction/KEGG application/pathway/hsa"
    KGML_to_Matrix(pathway_name = pathway_name, KGML_file_path = KGML_directory, save_path = save_directory)


# In[8]:


library_network = []
for ii in range(len(pathway_ID)):
    file_name = "/Users/liaochenpo/Desktop/Meeting/Network construction/KEGG application/pathway/hsa" + pathway_ID[ii] + "(directed)"
    directed_adjmatrix = pd.read_pickle(file_name).to_numpy()
    ##directed to undirected
    reverse = (np.nonzero(directed_adjmatrix)[1], np.nonzero(directed_adjmatrix)[0])
    directed_adjmatrix[reverse] = 1
    library_network.append(directed_adjmatrix)


# In[9]:


alpha = []
for ii in range(len(pathway_ID)):
    current_mat = library_network[ii]
    degree_seq = [sum(row) for row in current_mat]
    degree_seq_non_zero = [i for i in degree_seq if i != 0]
    degree_seq_non_zero = np.array(degree_seq_non_zero)
    fit = powerlaw.Fit(degree_seq_non_zero, xmin=1, discrete=True)
    alpha.append(fit.power_law.alpha)


# In[10]:


estimated_alpha = round(sum(alpha)/len(alpha), 2)
estimated_alpha


# In[11]:


sd_alpha = round(stdev(alpha), 2)
sd_alpha


# In[12]:


input_alpha = np.random.normal(loc = estimated_alpha, scale = sd_alpha, size = 1)
input_alpha


# In[13]:


print("Total reference network: {}".format(len(library_network)))


# ### Potential network construction

# In[14]:


seed = 20211208
np.random.seed(seed)
potential_net_size = 13
potential_net = []
sample_DS_data = []
for DS_count in range(20):
    # degree sequence generation
    EG_result = "failure"
    while(EG_result == "failure"):
        input_alpha = np.random.normal(loc = estimated_alpha, scale = sd_alpha, size = 1)
        d = truncated_power_law(alpha = input_alpha, maximum_value = potential_net_size - 1)
        sample_DS = d.rvs(size = potential_net_size)
        sample_DS.sort()
        sample_DS = sample_DS[::-1]
        EG_result = EGtest(sample_DS)
    # network construction
    sample_net = net_gen(np.array(sample_DS))
    sample_DS_data.append(sample_DS)
    for net_index in range(len(sample_net)):
        potential_net.append(sample_net[net_index])
len(potential_net)


# In[15]:


sample_DS_data


# ### Dissimilarity

# In[16]:


# network properties in library (library_network)
library_ave_path_len = []
library_max_degree_centrality = []
library_transitivity = []
for lib in library_network:
    target_net = nx.from_numpy_matrix(np.array(lib))
    # average path length
    dis_mat = nx.floyd_warshall_numpy(target_net) # distance matrix
    dis_mat_without_inf = np.where(np.isinf(dis_mat), len(dis_mat), dis_mat) 
    library_ave_path_len.append(dis_mat_without_inf.sum()/len(dis_mat)/len(dis_mat))
    # max degree centrality
    library_max_degree_centrality.append(max(nx.degree_centrality(target_net).values()))
    # transitivity
    library_transitivity.append(nx.transitivity(target_net))


# In[17]:


# mean of network properties
mean_library_ave_path_len = sum(library_ave_path_len)/len(library_ave_path_len)
mean_library_max_degree_centrality = sum(library_max_degree_centrality)/len(library_max_degree_centrality)
mean_library_transitivity = sum(library_transitivity)/len(library_transitivity)
print('average path length: {}'.format(mean_library_ave_path_len))
print('max degree centrality: {}'.format(mean_library_max_degree_centrality))
print('transitivity: {}'.format(mean_library_transitivity))


# In[18]:


# network properties for potential network
ave_path_len_list = []
max_degree_centrality_list = []
transitivity_list = []
for net in potential_net:
    G_net = nx.from_numpy_matrix(np.array(net))
    dis_mat = nx.floyd_warshall_numpy(G_net)
    dis_mat_without_inf = np.where(np.isinf(dis_mat), len(dis_mat), dis_mat) 
    ave_path_len_list.append(dis_mat_without_inf.sum()/len(dis_mat)/len(dis_mat))
    max_degree_centrality_list.append(max(nx.degree_centrality(G_net).values()))
    transitivity_list.append(nx.transitivity(G_net))
ave_path_len_list = np.array(ave_path_len_list)
max_degree_centrality_list = np.array(max_degree_centrality_list)
transitivity_list = np.array(transitivity_list)
print("Network properties for potential network")
print("(1) Average path length")
n_bar = int(len(potential_net)/1000)
plt.hist(ave_path_len_list, n_bar,
         histtype ='bar') 
plt.title('Average path length for potential network\n', 
          fontweight ="bold") 
plt.show()
print("(2) Maximum centrality")
plt.hist(max_degree_centrality_list, n_bar,
         histtype ='bar') 
plt.title('Maximum centrality for potential network\n', 
          fontweight ="bold") 
plt.show()
print("(3) Transitivity")
plt.hist(transitivity_list, n_bar,
         histtype ='bar') 
plt.title('Transitivity for potential network\n', 
          fontweight ="bold") 
plt.show()


# In[19]:


# network selection by network properties
dis_mat = nx.floyd_warshall_numpy(target_net)
dis_mat_without_inf = np.where(np.isinf(dis_mat), len(dis_mat), dis_mat) 
ave_path_len_measure = ave_path_len_list - mean_library_ave_path_len
max_degree_centrality_list = max_degree_centrality_list - mean_library_max_degree_centrality
transitivity_measure = transitivity_list - mean_library_transitivity
# dissimilarity
measure = ave_path_len_measure * ave_path_len_measure + max_degree_centrality_list * max_degree_centrality_list + transitivity_measure * transitivity_measure


# In[20]:


seed = 20211208
np.random.seed(seed)
tmp = measure.copy()
tmp.sort()
count = 0
for i in range(5):
    count += 1
    aa = np.where(measure == tmp[i])[0][0]
    print(count)
    show_graph_with_labels(np.array(potential_net[aa]))


# ### Data preprocessing

# In[21]:


breast_cancer = pd.read_csv("/Users/liaochenpo/Desktop/Meeting/RBDD network construction/GSE69240_Preprocessed_datamatrix.csv",index_col="GeneID")
ovarian_cancer = pd.read_csv("/Users/liaochenpo/Desktop/Meeting/RBDD network construction/gene_expression(with GeneSymbol).csv" ,index_col="gene_symbol")
breast_genes = ["STAT1","FHL1","ESR1","TFF1","BCL2","IGF1R","IKBKB","TSC1","IGF1","SFN","CDK4","CCND1","CDKN2A"]
ovarian_genes = ["CDH1","PGR","NF1","PTEN","TP53","STK11","ATM","ATR","CHEK2","MAP3K1","FGFR2","CDK1","TGFB1"]
breast_cancer = breast_cancer.loc[["STAT1","FHL1","ESR1","TFF1","BCL2","IGF1R","IKBKB","TSC1","IGF1","SFN","CDK4","CCND1","CDKN2A"]]
ovarian_cancer = ovarian_cancer.loc[["CDH1","PGR","NF1","PTEN","TP53","STK11","ATM","ATR","CHEK2","MAP3K1","FGFR2","CDK1","TGFB1"]]
ovarian_cancer = ovarian_cancer.loc["CDH1":"TGFB1","TCGA.04.1331":"TCGA.61.2111"]
breast_cancer = breast_cancer.loc["STAT1":"CDKN2A","N1":"T25"]


# In[22]:


breast_cancer = breast_cancer.T
breast_cancer.index = list(range(1,36))
breast_cancer = breast_cancer[10:36]
breast_cancer.head()


# In[23]:


ovarian_cancer = ovarian_cancer.T
ovarian_cancer.index = list(range(1,283))
ovarian_cancer.head()


# ### Breast cancer

# In[24]:


breast_cancer_cor = breast_cancer.corr()
breast_cancer_cor


# In[25]:


######################################
# Labeling by permutation and Calculating network score
######################################
seed = 20211208
np.random.seed(seed)
Structure_score = measure.copy()
Structure_score.sort()
final_gene_label = []
final_net_score = []
Graph_list = []
unique_network_index = []
structure_count = 0


Sturcture_output, Gene_label_output, Score_output = Network_score_with_cor(Candidate_network = potential_net, 
                                                                            Structure_score = measure, 
                                                                           Correlation_mat = breast_cancer_cor.values, 
                                                                           Gene_list = breast_genes,
                                                                           Network_ranking_size = len(potential_net) / 10) 
sorted_index = np.argsort(Score_output)
sorted_index = sorted_index[::-1]
sorted_sturcture_output = []
sorted_score_output = []
sorted_gene_label_output = []
for i in sorted_index:
    sorted_sturcture_output.append(Sturcture_output[i])
    sorted_score_output.append(Score_output[i])
    sorted_gene_label_output.append(Gene_label_output[i])


# In[26]:


len(potential_net)


# In[27]:


len(sorted_index)


# In[28]:


seed = 20211208
np.random.seed(seed)
print(show_graph_with_labels(sorted_sturcture_output[0]))
print(sorted_gene_label_output[0])


# In[29]:


sorted_sturcture_output[0]


# ### Ovarian cancer

# In[30]:


ovarian_cancer_cor = ovarian_cancer.corr()
ovarian_cancer_cor


# In[31]:


######################################
# Labeling by permutation and Calculating network score
######################################
Structure_score = measure.copy()
Structure_score.sort()
final_gene_label = []
final_net_score = []
Graph_list = []
unique_network_index = []
structure_count = 0


Sturcture_output, Gene_label_output, Score_output = Network_score_with_cor(Candidate_network = potential_net, 
                                                                            Structure_score = measure, 
                                                                           Correlation_mat = ovarian_cancer_cor.values, 
                                                                           Gene_list = ovarian_genes,
                                                                           Network_ranking_size = len(potential_net) / 10) 
sorted_index = np.argsort(Score_output)
sorted_index = sorted_index[::-1]
sorted_sturcture_output = []
sorted_score_output = []
sorted_gene_label_output = []
for i in sorted_index:
    sorted_sturcture_output.append(Sturcture_output[i])
    sorted_score_output.append(Score_output[i])
    sorted_gene_label_output.append(Gene_label_output[i])


# In[32]:


seed = 20211208
np.random.seed(seed)
print(show_graph_with_labels(sorted_sturcture_output[0]))
print(sorted_gene_label_output[0])


# In[33]:


sorted_sturcture_output[0]

