import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from copy import deepcopy

global seller_label, buyer_label
seller_label='seller'
buyer_label='buyer'

pd.options.mode.chained_assignment = None

# returns a maximal matching where the matched pairs are in (seller,buyer) order
# you can read more about maximal matching in Section 10.6 but basically: 
#     - In the case where preferences are binary (y/n): it is a matching that includes as many nodes as possible
#     - In the case where valuations are given (weighted graph): it is the same as the optimal matching 
#                                                                (total valuation is as high as as possible)
# that means that: 
#     - if you pass in a weighted graph, `make_matching` will return the optimal matching 
#     - if you pass in an unweighted graph and there's a perfect matching, `make_matching` will return a perfect matching 
#     - if you pass in an unweighted graph and there's no perfect matching, `make_matching` will return a matching that 
#       doesn't include all nodes but includes as many as it can. 
def make_matching(graph):
    matching = nx.max_weight_matching(graph)
    matching = [(n1,n2) if graph.nodes[n1]['bipartite']==seller_label else (n2,n1) for n1,n2 in matching]
    return matching


# returns a list of unique sellers and buyers in the graph
def find_sellers_and_buyers(graph):
    sellers = [node for node in graph.nodes if graph.nodes[node]['bipartite']==seller_label]
    buyers = [node for node in graph.nodes if graph.nodes[node]['bipartite']==buyer_label]
    return sellers, buyers


# finds all neighbors of the nodes in the graph
def find_neighbors(nodes, graph):
    return set([neighbor for node in nodes for neighbor in graph.neighbors(node)])

    
# checks whether a specific node in a graph is included in a given matching
def is_matched(node, matching):
    return any(node in pair for pair in matching)


# takes in a bipartite graph with edges representing buyer preferences, and tells you whether a perfect matching exists in that graph
def has_perfect_matching(graph):
    matching = nx.max_weight_matching(graph) #find the best possible matching for the graph
    if all(is_matched(node,matching) for node in graph.nodes): #returns True if all nodes are matched
        return True #function returns True if matching is perfect
    else:
        return False #function returns True if matching is not perfect


# returns a constricted set if one exists
def find_constricted_set(graph, matching):
    sellers, buyers = find_sellers_and_buyers(graph)
    constricted_set = []
    neighbors = []
    latest_set = [node for node in buyers if not is_matched(node,matching)] #unmatched buyers
    
    # Alternating BFS (see section 10.6)
    while True:
        constricted_set.extend(latest_set)
        
        # Find all sellers who are neighbors of the buyers
        latest_neighbors = [node for node in find_neighbors(latest_set,graph) if node not in neighbors]
        if len(latest_neighbors) == 0:
            break

        # Find only the buyers that are matched to these sellers in the matching
        neighbors.extend(latest_neighbors)
        neighbor_matches = [match for node in latest_neighbors for match in matching if node in match]
        neighbor_matches = [n1 if n2 in neighbors else n2 for n1,n2 in neighbor_matches]
        latest_set = neighbor_matches
    
    return sorted(constricted_set), sorted(neighbors)
        

# finds the preferred sellers for each buyer, given their valuations and a set of prices
def find_preferred_sellers(graph, prices):
    for seller,buyer in graph.edges:
        graph.edges[(seller,buyer)]['payoff'] = graph.edges[(seller,buyer)]['weight'] - prices[seller]
    
    preferred_sellers = {}
    for buyer in [node for node in graph.nodes if node not in prices.keys()]:
        max_payoff = max(graph.edges[(seller,buyer)]['payoff'] for seller in graph.neighbors(buyer))
        preferred_sellers[buyer] = [seller for seller in graph.neighbors(buyer) 
                                    if max_payoff==graph.edges[(buyer,seller)]['payoff']]
    
    return preferred_sellers

    
# finds a set of market-clearing prices that allow a matching to be made
def find_market_clearing_prices(graph, matching):
    sellers, buyers = find_sellers_and_buyers(graph)
    prices = {s:0 for s in sellers}

    while True:
        preferred_sellers = find_preferred_sellers(graph, prices)
        matches_not_preferred_sellers = []
        for seller,buyer in matching: 
            if seller not in preferred_sellers[buyer]:
                matches_not_preferred_sellers.append((seller,buyer))

        if len(matches_not_preferred_sellers)==0:
            break

        #augment prices so matches are preferred sellers
        for seller,buyer in matches_not_preferred_sellers:
            pref_seller = preferred_sellers[buyer][0]
            pref_payoff = graph.edges[(pref_seller,buyer)]['weight'] - prices[pref_seller]
            correct_payoff = graph.edges[(seller,buyer)]['weight'] - prices[seller]
            prices[pref_seller] += (pref_payoff - correct_payoff)
                    
            preferred_sellers = find_preferred_sellers(graph, prices)
            matches_not_preferred_sellers = [match for match in matching if match not in preferred_sellers]
            
            if len(matches_not_preferred_sellers)==0:
                break
    return prices


# returns a perfect matching for a bipartite graph
# if one does not exist, returns a constricted set
def perfect_match(graph):
    matching = make_matching(graph)
    has_perfect_match = all(is_matched(node,matching) for node in graph.nodes)
    
    if has_perfect_match: 
        return has_perfect_match, matching, ()
    else: 
        return has_perfect_match, [], find_constricted_set(graph, matching)


def find_perfect_matching(data, seller_column_name, buyer_column_name):

    # Define an empty graph
    graph = nx.Graph()

    # Create lists for the sellers and buyers
    sellers = sorted(data[seller_column_name])
    buyers = sorted(data[buyer_column_name])

    # Add the sellers and buyers as nodes in a bipartite graph
    graph.add_nodes_from(sellers, bipartite=seller_label)
    graph.add_nodes_from(buyers, bipartite=buyer_label)

    # Create a list of (seller,buyer) tuples, representing preferences 
    preferences = [[data[seller_column_name][i],
                    data[buyer_column_name][i]] for i in data.index]

    # Add these preferences as unweighted edges in the graph
    graph.add_edges_from(preferences)

    # Calculate whether there's a perfect matching of buyers to sellers
    has_perfect_match, perfect_matching, constricted_set =\
        perfect_match(graph)

    return graph, has_perfect_match, perfect_matching, constricted_set


# returns an optimal matching for a bipartite graph, weighted by the buyer valuations
def optimal_match(graph):
    if 'weight' not in list(graph.edges(data=True))[0][2].keys():
        raise Exception('Please make sure the graph is weighted')
    
    matching = make_matching(graph)
    market_clearing_prices = find_market_clearing_prices(graph, matching)
    
    return matching, market_clearing_prices


def find_optimal_matching(data, seller_column_name, buyer_column_name, valuation_column_name):

    # Define an empty graph
    graph = nx.Graph()

    # Create lists for the sellers and buyers
    sellers = sorted(data[seller_column_name])
    buyers = sorted(data[buyer_column_name])

    # Add the sellers and buyers as nodes in a bipartite graph
    graph.add_nodes_from(sellers, bipartite=seller_label)
    graph.add_nodes_from(buyers, bipartite=buyer_label)

    # Create a list of (seller,buyer,valuation) triples, representing mutual preferences 
    valuations = [[data[seller_column_name][i],
                   data[buyer_column_name][i],
                   data[valuation_column_name][i]] for i in data.index]

    # Add these valuations as edges in the graph
    graph.add_weighted_edges_from(valuations)

    # Calculate the optimal matching, or the matching that will maximize the total valuation
    optimal_matching, market_clearing_prices = optimal_match(graph)

    return graph, optimal_matching, market_clearing_prices


# prints the number of buyers and sellers that are matched vs. unmatched in the matching
def count_matched_and_unmatched(graph, matching):
    sellers, buyers = find_sellers_and_buyers(graph)
    matched_sellers = [x[0] for x in matching]
    matched_buyers = [x[1] for x in matching]

    print("Sellers (matched,unmatched):", len(matched_sellers), len(sellers)-len(matched_sellers))
    print("Buyers (matched,unmatched):", len(matched_buyers), len(buyers)-len(matched_buyers))
    

# calculates the social welfare (or sum of buyer and seller payoffs) under a matching
def social_welfare(graph, matching):
    welfare = 0
    for edge in matching:
        welfare += graph.edges[edge]['weight']
    return welfare


# plots the bipartite graph, with weighted edges
# if a matching is provided the matched edges are highlighted in blue
def plot_matching_market(graph, matching=[], constricted_set=[]):
    sellers,buyers = find_sellers_and_buyers(graph)
    nodes_in_constricted_set = [x for lst in constricted_set for x in lst]
    
    colors = ['orange' if node in nodes_in_constricted_set else 
              'yellow' if node in sellers else 'lightblue' for node in graph.nodes] 
    
    pos = nx.bipartite_layout(graph, sellers)
    pos.update( (n, (1, len(sellers)-i)) for i, n in enumerate(sellers)) 
    pos.update( (n, (2, len(buyers)-i)) for i, n in enumerate(buyers)) 
    
    try:
        edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges]
    except:
        edge_weights = [2 for edge in graph.edges]
        
    if len(matching)>0:
        print(1)
        color_by_matching = ['blue' if edge in matching else 'lightgrey' for edge in graph.edges] 
    else:
        color_by_matching = ['grey' for edge in graph.edges] 
    
    node_outline = [5 if node in nodes_in_constricted_set else 0 for node in graph.nodes]
    
    return nx.draw_networkx(graph, pos=pos, 
                            node_color = colors, node_size=1000, 
                            linewidths = node_outline,
                            edge_color = color_by_matching, width=edge_weights) 


def match_in_order(kidney_exchange, matching_column):
    donors_taken = []
    candidates_seen = []
    candidates_matched = []
    matching = []
    kidney_exchange = kidney_exchange.sort_values(matching_column) #WaitingListOrder, SurvivalWithoutTransplant
    kidney_exchange = kidney_exchange.reset_index(drop=True)
    for i in kidney_exchange.index:
        candidate = kidney_exchange.Candidate[i]
        if candidate not in candidates_seen:
            donor = kidney_exchange.Donor[i]
            df = kidney_exchange[(kidney_exchange.Candidate==candidate) &
                                 (~kidney_exchange.Donor.isin(donors_taken))]
            if df.shape[0] > 0:
                match = (df[df.LifeExtendedWithTransplant==max(df.LifeExtendedWithTransplant)].Donor.iloc[0], 
                         candidate)
                donors_taken.append(match[0])
                candidates_matched.append(match[1])
                matching.append(match)
        candidates_seen.append(candidate)
    return matching


def plot_donor_prices(market_clearing_prices):
    donors = pd.read_csv('Data/kidney_donors_synthetic_data.csv')
    donors['MarketClearingPrice'] = [market_clearing_prices[d] if d in market_clearing_prices 
                                     else 0 for d in donors.Donor]
    plt.boxplot([donors.MarketClearingPrice[donors.BloodType==b] for b in ['O','A','B','AB']],
                labels=['O','A','B','AB'])
    plt.xlabel('Donor Blood Type')
    plt.ylabel('Market-Clearing Price for Kidney')
    plt.show()


def plot_kidney_exchange(kidney_exchange, matching, title=''):
    matched_candidates = [x[1] for x in matching]
    waiting_list = pd.read_csv('Data/kidney_waiting_list_synthetic_data.csv')
    waiting_list['Matched'] = [1 if c in matched_candidates else 0 for c in waiting_list.Candidate]
    waiting_list['AgeGroup'] = pd.cut(waiting_list.Age, [0, 25, 45, 65, 85], labels=['25 and Under','26-45','46-65','66+'])
    waiting_list['SurvivalWithoutTransplant'] = pd.cut(waiting_list.SurvivalWithoutTransplant, 
                                                       [0, 18, 100], labels=['1.5 Years or Less','1.5+ Years'])

    kidney_exchange['Matched'] = [1 if (kidney_exchange.Donor[i],kidney_exchange.Candidate[i]) in 
                                  matching else 0 for i in kidney_exchange.index]
    matchings = kidney_exchange[kidney_exchange.Matched==1]
    matchings['AgeGroup'] = pd.cut(matchings.Age, [0, 25, 45, 65, 85], labels=['25 and Under','26-45','46-65','66+'])
    matchings['SurvivalWithoutTransplant'] = pd.cut(matchings.SurvivalWithoutTransplant, 
                                                    [0, 18, 100], labels=['1.5 Years or Less','1.5+ Years'])

    plt.rcParams['figure.figsize'] = [15, 3] #make the figure bigger or smaller
    fig, axes = plt.subplots(nrows=1, ncols=3)

    waiting_list.groupby('AgeGroup').agg({'Matched':'mean'}).\
    plot(kind='bar', legend=False, rot=0, ax=axes[0],
         color = ['#4363d8'],
         xlabel='Candidate\'s Age',
         ylabel='Fraction of Waiting List Matched')
    waiting_list.groupby('HasMedicalConditions').agg({'Matched':'mean'}).\
    plot(kind='bar', legend=False, rot=0, ax=axes[1],
         color = ['#ffe119'],
         xlabel='Does the Candidate have\nOther Medical Conditions?',
         ylabel='Fraction of Waiting List Matched')
    waiting_list.groupby('SurvivalWithoutTransplant').agg({'Matched':'mean'}).\
    plot(kind='bar', legend=False, rot=0, ax=axes[2],
         color = ['#a9a9a9'],
         xlabel='How long would the Candidate Survive\nwithout a Transplant?',
         ylabel='Fraction of Waiting List Matched')
    fig.suptitle(title)

#     plt.rcParams['figure.figsize'] = [15, 3] #make the figure bigger or smaller
#     fig, axes = plt.subplots(nrows=1, ncols=3)

#     matchings.groupby('SurvivalWithoutTransplant').agg({'Compatibility':'mean'}).\
#     plot(kind='bar', legend=False, rot=0, ax=axes[1],
#          title='Compatibility')
#     matchings.groupby('AgeGroup').agg({'Compatibility':'mean'}).\
#     plot(kind='bar', legend=False, rot=0, ylim=[6,8.5], ax=axes[1],
#          title='Quality of Match (histocompatibility)')
#     matchings.groupby('HasMedicalConditions').agg({'Compatibility':'mean'}).\
#     plot(kind='bar', legend=False, rot=0, ylim=[6,8.5], ax=axes[0],
#          title='Quality of Match (histocompatibility)')


def plot_kidney_markets(kidney_exchange, kidney_match, optimal_kidney_matching):
    # matching under the two procedures described

    ## give priority to people who are higher on the waiting list
    ## i.e., assigns kidneys first come first served (waiting list order)
    waitlist_kidney_matching = match_in_order(kidney_exchange, 'WaitingListOrder') 

    ## give priority to people who who would have lower survival without the transplant
    ## i.e., assigns kidneys to sickest people first (lowest survival without transplant)
    survival_kidney_matching = match_in_order(kidney_exchange, 'SurvivalWithoutTransplant') 


    print('How many years of life do we extend by assigning kidneys?')
    print()
    print('    Via Matching Market:',social_welfare(kidney_match, optimal_kidney_matching))
    print('    First Come First Served (waiting list order):',social_welfare(kidney_match, waitlist_kidney_matching))
    print('    Sickest People First (lowest survival without transplant):',social_welfare(kidney_match, survival_kidney_matching))
    print()
    print()
    print()


    print('Who gets a kidney?')
    plot_kidney_exchange(kidney_exchange, matching = optimal_kidney_matching, 
                         title='Disparities when Assigning Via Matching Market')
    plot_kidney_exchange(kidney_exchange, matching = waitlist_kidney_matching, 
                         title='Disparities when Assigning First Come First Served (waiting list order)')
    plot_kidney_exchange(kidney_exchange, matching = survival_kidney_matching,
                         title = 'Disparities when Assigning Sickest People First (lowest survival without transplant)')

    
def plot_hospital_prices(market_clearing_prices):
    hospitals = pd.read_csv('Data/hospitals_synthetic_data.csv')
    hospitals['MarketClearingPrice'] = [market_clearing_prices[hosp] 
                                        if hosp in market_clearing_prices 
                                        else 0 for hosp in hospitals.HospitalPosition]
    plt.rcParams['figure.figsize'] = [6, 4] #make the figure bigger or smaller
    plt.scatter(hospitals.HospitalQuality, hospitals.MarketClearingPrice)
    plt.xlabel('Hospital Reputation\n(higher=better)')
    plt.ylabel('Hospital\'s Market-Clearing Price')
    plt.show()

    
def plot_residency_match_rates(matching, outcome='quality'):
    matched_hospitals = [x[0] for x in matching]
    matched_residents = [x[1] for x in matching]
    
    residents = pd.read_csv('Data/residents_synthetic_data.csv')
    residents['Matched'] = [1 if res in matched_residents else 0 for res in residents.Resident]
    residents['NumHospitalsRankedBin'] = pd.cut(residents.NumHospitalsRanked, 
                                                [5.988,9,12,19], labels=['6-9','10-12','13-20'])
    residents['ResidentQualityBin'] = pd.cut(residents.ResidentQuality, 
                                             [40,49,52,100], labels=['Low','Mid','High'])

    
    hospitals = pd.read_csv('Data/hospitals_synthetic_data.csv')
    hospitals['Matched'] = [1 if hosp in matched_hospitals else 0 for hosp in hospitals.HospitalPosition]
    hospitals['NumResidentsRankedBin'] = pd.cut(hospitals.NumResidentsRanked/hospitals.NumPositionsOffered, 
                                                [0,5,10,15], labels=['3-5','6-10','11-15'])
    hospitals['HospitalQualityBin'] = pd.cut(hospitals.HospitalQuality, 
                                             [20,45,60,100], labels=['Low','Mid','High'])

    if outcome=='num_ranked':
        plt.rcParams['figure.figsize'] = [12, 3] #make the figure bigger or smaller
        fig, axes = plt.subplots(nrows=1, ncols=2)
        
        residents.groupby('NumHospitalsRankedBin').\
        agg({'Matched':'mean'}).\
        plot(kind='bar', legend=False, rot=0, ax=axes[0], 
             color = ['#ffe119','#a9a9a9'],
             xlabel='# Hospitals Ranked', 
             ylabel='Fraction of Residents Matched', title='Resident')

        hospitals.groupby('NumResidentsRankedBin').\
        agg({'Matched':'mean'}).\
        plot(kind='bar', legend=False, rot=0, ax=axes[1], 
             color = ['#4363d8','#ffe119','#a9a9a9'],
             xlabel='# Residents Ranked\n(Per Position Offered)', 
             ylabel='Fraction of Positions Filled', title='Hospital')
        plt.show()

    if outcome=='quality':
        plt.rcParams['figure.figsize'] = [12, 3] #make the figure bigger or smaller
        fig, axes = plt.subplots(nrows=1, ncols=2)

        residents.groupby('ResidentQualityBin').\
        agg({'Matched':'mean'}).\
        plot(kind='bar', legend=False, rot=0, ax=axes[0],
             color = ['#ffe119','#a9a9a9'],
             xlabel='Strength of Resident\'s Application\n(test scores,med school ranking,etc.)',
             ylabel='Fraction of Residents Matched', title='Resident')

        hospitals.groupby('HospitalQualityBin').\
        agg({'Matched':'mean'}).\
        plot(kind='bar', legend=False, rot=0, ax=axes[1],
             color = ['#4363d8','#ffe119','#a9a9a9'],
             xlabel='Hospital Reputation\n(skill of doctors,novelty of research,rating of department,etc.)', 
             ylabel='Fraction of Positions Filled', title='Hospital')
        plt.show()


def plot_residency_rankings(residency_ranks, graph, matching):
    residency_ranks['Institution'] = [x[:7] for x in residency_ranks.HospitalPosition]

    all_matched_residents = []
    all_matched_positions = []
    offer_matching = []

    while True:
        num_matched_residents = len(all_matched_residents)
        offers = []
        # available pairs
        df = residency_ranks[(~residency_ranks.Resident.isin(all_matched_residents)) & 
                             (~residency_ranks.HospitalPosition.isin(all_matched_positions))]
        if df.shape[0]>0:
            # hospital makes offers to their top unmatched residents
            for inst in set(df['Institution'].tolist()):
                n_pos = len(df[df.Institution==inst].HospitalPosition.value_counts())
                offers.append(df[df.HospitalPosition==df[df.Institution==inst].HospitalPosition.iloc[0]].\
                              sort_values('HospitalsRank').iloc[:(n_pos)])
            offers = pd.concat(offers).sort_values('ResidentsRank')
            # resident selects their top offer
            for res in set(offers['Resident'].tolist()):
                if res in offers.Resident.tolist():
                    match = (offers[offers.Resident==res].HospitalPosition.iloc[0],
                             offers[offers.Resident==res].Resident.iloc[0])
                    offer_matching.append(match)
                    all_matched_positions.append(match[0])
                    all_matched_residents.append(match[1])
                    offers = offers[offers.HospitalPosition != match[0]]
        # stop once there are no more matches to be made
        if len(all_matched_residents)==num_matched_residents:
            break

    def make_plotting_df(metric):
        matches = residency_ranks[[(residency_ranks.HospitalPosition[i], 
                                    residency_ranks.Resident[i]) in offer_matching
                                   for i in residency_ranks.index]]
        matches[metric+"Group"] = pd.cut(matches[metric], 
                                         [0,5,10,50], labels=['High\n(1-5)','Mid\n(6-10)','Low\n(11+)'])
        df1 = matches.groupby(metric+"Group").agg({metric:'count'}).\
                rename(columns={metric:'Hospital Makes Offers'})
        df1 = df1.reset_index(drop=False)

        matches = residency_ranks[[(residency_ranks.HospitalPosition[i], 
                                    residency_ranks.Resident[i]) in matching
                                  for i in residency_ranks.index]]
        matches[metric+"Group"] = pd.cut(matches[metric], 
                                         [0,5,10,50], labels=['High\n(1-5)','Mid\n(6-10)','Low\n(11+)'])
        df2 = matches.groupby(metric+"Group").agg({metric:'count'}).\
                rename(columns={metric:'Matching Market'})
        df2 = df2.reset_index(drop=False)

        return df1.merge(df2, on=metric+"Group")

    print('Social Welfare when Hospitals Make Initial Offers:', 
          social_welfare(graph=graph, matching=offer_matching))
    print('Social Welfare with a Matching Market:', 
          social_welfare(graph=graph, matching=matching))


    plt.rcParams['figure.figsize'] = [12, 3] #make the figure bigger or smaller
    fig, axes = plt.subplots(nrows=1, ncols=2)
    make_plotting_df("ResidentsRank").\
        plot.bar(x = 'ResidentsRankGroup', rot=0, legend=False, ax=axes[0],
                 color = ['#4363d8','#ffe119','#a9a9a9'],
                 xlabel="How High did Residents Rank\nthe Hospital they Work For",
                 ylabel="# Positions",
                 title="Resident's Satisfaction with Matching")
    make_plotting_df("HospitalsRank").\
        plot.bar(x = 'HospitalsRankGroup', rot=0, legend=True, ax=axes[1],
                 color = ['#4363d8','#ffe119','#a9a9a9'],
                 xlabel="How High did Hospitals Rank\nthe Resident they Work With",
                 ylabel="# Positions",
                 title="Hospital's Satisfaction with Matching")
    plt.show()


def plot_school_prices(school_ranks, market_clearing_prices):
    school_ranks['MarketClearingPrice'] = [market_clearing_prices[x] for x in school_ranks.SchoolPosition]
    school_ranks['Institution'] = [x[:4] for x in school_ranks.SchoolPosition]

    agg_ranks = school_ranks.\
          groupby('Institution').\
          agg({'StudentsRanking':'mean', 'MarketClearingPrice':'mean'}).\
          rename(columns={'Institution':'SchoolPosition', 
                          'StudentsRanking':'Ave. Ranking by Students', 
                          'MarketClearingPrice':'Ave. Market Clearing Price'})

    plt.figure(figsize=(16,4))

    # plot table
    ax1 = plt.subplot(121)
    plt.axis('off')
    tbl = pd.plotting.table(ax1, agg_ranks, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1, 2)

    # scatterplot
    ax2 = plt.subplot(122)
    agg_ranks.plot(ax=ax2, rot=0, legend=False, 
         #color = ['#4363d8','#ffe119','#a9a9a9'],
         x='Ave. Ranking by Students', y='Ave. Market Clearing Price', style="o", ms=8, 
         xlabel='Average Ranking by Students\n(1 is highest, 5 is lowest)',
         ylabel='Average Market Clearing Price\nfor Spots in the School')

    plt.show()


def plot_public_school_matching(school_ranks, matching):
    first_choice_ranking = []
    placed_students = []
    filled_positions = []
    for rank in range(1,6):
        for pos in set(school_ranks.SchoolPosition):
            if pos not in filled_positions:
                students = school_ranks[(~school_ranks.Student.isin(placed_students)) & 
                                        (school_ranks.SchoolPosition==pos) & 
                                        (school_ranks.StudentsRanking==rank)].Student
                if len(students) > 0:
                    student = np.random.choice(students.tolist())
                    placed_students.append(student)
                    filled_positions.append(pos)
                    first_choice_ranking.append((pos,student))

    school_ranks['Matched'] = [1 if (school_ranks.SchoolPosition[i], school_ranks.Student[i]) in first_choice_ranking 
                               else 0 for i in school_ranks.index]
    first_choice_matches = school_ranks[school_ranks.Matched==1]

    school_ranks['Matched'] = [1 if (school_ranks.SchoolPosition[i], school_ranks.Student[i]) in matching 
                               else 0 for i in school_ranks.index]
    market_matches = school_ranks[school_ranks.Matched==1]

    plt.rcParams['figure.figsize'] = [8, 3] #make the figure bigger or smaller
    
    first_choice_matches.groupby('StudentsRanking').agg({'Matched':'count'}).\
    rename(columns={'Matched':'Assign by Highest Rank'}).\
    merge(market_matches.groupby('StudentsRanking').agg({'Matched':'count'}).\
          rename(columns={'Matched':'Assign by Matching Market'}), on='StudentsRanking', how='outer').\
    reset_index(drop=False).\
    plot(kind='bar', x='StudentsRanking',rot=0,
         color = ['#4363d8','#ffe119','#a9a9a9'],
         xlabel='How Highly did Students Rank\nthe School they were Matched to?\n(1 is highest, 5 is lowest)',
         ylabel='# Students', title='Comparing Different Ways of\nAssigning Students to Schools')
    plt.show()

    
def plot_school_by_race(school_ranks, matching):
    
    school_ranks['Matched'] = [1 if (school_ranks.SchoolPosition[i], school_ranks.Student[i]) in matching 
                               else 0 for i in school_ranks.index]
    school_ranks['SchoolRanking'] = [6-int(s[3]) for s in school_ranks.SchoolPosition]
    market_matches = school_ranks[school_ranks.Matched==1]
    df1 = market_matches.groupby('Race').agg({'SchoolRanking':'mean'}).\
    rename(columns={'SchoolRanking':'Assigned in Matching Market'})
    df1 = df1.reset_index()

    school_ranks['School'] = [x[:4] for x in school_ranks['SchoolPosition']]
    market_matches = school_ranks[school_ranks['ClosestSchool']==school_ranks['School']]
    school_ranks['SchoolRanking'] = [6-int(s[3]) for s in school_ranks.SchoolPosition]
    df2 = market_matches.groupby('Race').agg({'SchoolRanking':'mean'}).\
    rename(columns={'SchoolRanking':'Assigned to Closest School'})
    df2 = df2.reset_index()

    df1.merge(df2).\
    melt(id_vars=['Race'], value_vars=['Assigned to Closest School','Assigned in Matching Market']).\
    pivot(index='variable', columns='Race', values='value').\
    rename({'Race/Ethnicity':'Race'}).\
    reset_index(drop=False).\
    plot(kind='bar', x='variable', rot=0, figsize=(8,4),
         color = ['#4363d8','#ffe119','#a9a9a9'],
         xlabel='How Students get Assigned to Schools',
         ylabel='Average School Quality\n(test scores, student:teacher ratio, etc.)',
         title='Do matching markets eliminate racial disparities in school assignment?'
         )
    plt.show()


def life_extended(graph, matching):
    life_years = 0
    for edge in matching:
        life_years += graph.edges[edge]['weight']
    return life_years
    
